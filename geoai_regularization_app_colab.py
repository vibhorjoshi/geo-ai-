import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
import cv2
import tempfile
import math
import os
from datetime import datetime

# Config
st.set_page_config(page_title="Geo Footprint Regularization", layout="wide")
# Initialize session state
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
# ----------------- Functions -----------------

def get_orientation(polygon):
    try:
        mrr = polygon.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [math.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1]) for i in range(4)]
        idx = np.argmax(edges)
        dx = coords[(idx+1)%4][0] - coords[idx][0]
        dy = coords[(idx+1)%4][1] - coords[idx][1]
        return math.degrees(math.atan2(dy, dx))
    except:
        return 0

def regularize_polygon(polygon, threshold, angle_thresh):
    try:
        orientation = get_orientation(polygon)
        cos_a, sin_a = math.cos(math.radians(-orientation)), math.sin(math.radians(-orientation))
        rotate = lambda x, y: (x*cos_a - y*sin_a, x*sin_a + y*cos_a)
        rotated = Polygon([rotate(x, y) for x, y in polygon.exterior.coords])
        simplified = rotated.simplify(threshold, preserve_topology=True)
        ortho_coords = []
        coords = list(simplified.exterior.coords)

        for i in range(len(coords) - 1):
            x1, y1, x2, y2 = *coords[i], *coords[i+1]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            nearest = min([0, 90, 180], key=lambda a: abs(angle - a))
            if abs(angle - nearest) <= angle_thresh:
                ortho_coords.append((x1, y1))
                if i == len(coords) - 2:
                    if nearest == 0:
                        ortho_coords.append((x2, y1))
                    else:
                        ortho_coords.append((x1, y2))
            else:
                ortho_coords.append((x1, y1))
                if i == len(coords) - 2:
                    ortho_coords.append((x2, y2))

        ortho_coords.append(ortho_coords[0])
        ortho_polygon = Polygon(ortho_coords)
        cos_a, sin_a = math.cos(math.radians(orientation)), math.sin(math.radians(orientation))
        unrotate = lambda x, y: (x*cos_a - y*sin_a, x*sin_a + y*cos_a)
        final_coords = [unrotate(x, y) for x, y in ortho_polygon.exterior.coords]
        final_geom = Polygon(final_coords).buffer(0)
        if final_geom.is_empty or final_geom is None:
            return polygon
        return final_geom
    except Exception as e:
        print(f"Regularization failed: {e}")
        return polygon

def adaptive_regularize_polygon(polygon, base_threshold=1.0):
    vertex_count = len(list(polygon.exterior.coords)) - 1
    adaptive_threshold = base_threshold * (1 + vertex_count / 100)
    return regularize_polygon(polygon, adaptive_threshold, 5)

def hybrid_regularize_polygon(polygon, threshold=1.0, angle_thresh=5):
    simplified = regularize_polygon(polygon, threshold, angle_thresh)
    return regularize_polygon(simplified, threshold * 0.5, angle_thresh * 0.8)

def regularize_gdf(gdf, thresh, angle, method="standard"):
    gdf_out = gdf.copy()
    if method == "adaptive":
        gdf_out['geometry_reg'] = gdf.geometry.apply(
            lambda geom: adaptive_regularize_polygon(geom) if isinstance(geom, Polygon) else MultiPolygon([adaptive_regularize_polygon(p) for p in geom.geoms])
        )
    elif method == "hybrid":
        gdf_out['geometry_reg'] = gdf.geometry.apply(
            lambda geom: hybrid_regularize_polygon(geom, thresh, angle) if isinstance(geom, Polygon) else MultiPolygon([hybrid_regularize_polygon(p, thresh, angle) for p in geom.geoms])
        )
    else:
        gdf_out['geometry_reg'] = gdf.geometry.apply(
            lambda geom: regularize_polygon(geom, thresh, angle) if isinstance(geom, Polygon) else MultiPolygon([regularize_polygon(p, thresh, angle) for p in geom.geoms])
        )
    return gdf_out

def calculate_metrics(gdf):
    df = pd.DataFrame(index=gdf.index)

    def safe_vertex_count(g):
        try:
            if g and hasattr(g, 'exterior'):
                return len(list(g.exterior.coords)) - 1
        except:
            return np.nan
        return np.nan

    def safe_area(g):
        try:
            return g.area if g else np.nan
        except:
            return np.nan

    def safe_hausdorff(row):
        try:
            if row.geometry and row.geometry_reg:
                return row.geometry.hausdorff_distance(row.geometry_reg)
        except:
            return np.nan
        return np.nan

    df['vertices_before'] = gdf.geometry.apply(safe_vertex_count)
    df['vertices_after'] = gdf.geometry_reg.apply(safe_vertex_count)
    df['vertex_reduction'] = ((df['vertices_before'] - df['vertices_after']) / df['vertices_before']) * 100
    df['area_before'] = gdf.geometry.apply(safe_area)
    df['area_after'] = gdf.geometry_reg.apply(safe_area)
    df['area_ratio'] = (df['area_after'] / df['area_before']) * 100
    df['hausdorff'] = gdf.apply(safe_hausdorff, axis=1)

    return df


def show_comparison(gdf, idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    gdf.iloc[[idx]].plot(ax=ax1, color='blue', alpha=0.5)
    gpd.GeoDataFrame(geometry=[gdf.iloc[idx].geometry_reg], crs=gdf.crs).plot(ax=ax2, color='green', alpha=0.5)
    ax1.set_title("Original"); ax2.set_title("Regularized")
    ax1.axis('equal'); ax2.axis('equal')
    st.pyplot(fig)

def show_map(gdf):
    gdf = gdf.to_crs(epsg=4326)
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=16)
    folium.GeoJson(gdf, name='Original', style_function=lambda x: {"color": "blue"}).add_to(m)
    folium.GeoJson(gpd.GeoDataFrame(geometry=gdf['geometry_reg'], crs=gdf.crs), name='Regularized', style_function=lambda x: {"color": "green"}).add_to(m)
    folium.LayerControl().add_to(m)
    folium_static(m)

def process_image(image, method="edge"):
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if method == "edge":
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges)
    elif method == "mask":
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return Image.fromarray(mask)
    return image
def count_vertices(geom):
    if geom.geom_type == 'Polygon':
        return len(list(geom.exterior.coords)) - 1
    elif geom.geom_type == 'MultiPolygon':
        return sum(len(list(p.exterior.coords)) - 1 for p in geom.geoms)
    else:
        print(f"Unsupported geometry type: {geom.geom_type}")
        return 0


# ----------------- Sidebar Navigation -----------------

page = st.sidebar.radio("Navigation", [
    "📁 Upload Data",
    "🏗️ Run Regularization",
    "📊 View Metrics",
    "🖼️ Image Processing",
    "🧪 Compare Methods",
    "🎭 Mask Output"
])

# ----------------- Pages -----------------

if page == "📁 Upload Data":
    st.title("📁 Upload GeoJSON or Sample")
    if st.button("Load Sample Data"):
        gdf = gpd.GeoDataFrame({"geometry": [
            Polygon([(0, 0), (1, 0.1), (1.1, 1), (0.1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2.2), (3.2, 3), (2.2, 3), (2, 2)])
        ]}, crs="EPSG:4326")
        st.session_state.gdf = gdf
        st.success("Sample data loaded.")
    file = st.file_uploader("Upload GeoJSON, SHP or ZIP", type=["geojson", "shp", "zip"])
    if file:
        ext = file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        gdf = gpd.read_file(f"zip://{tmp_path}" if ext == 'zip' else tmp_path)
        st.session_state.gdf = gdf
        st.success(f"{len(gdf)} features loaded.")
        st.dataframe(gdf.head())

elif page == "🏗️ Run Regularization":
    st.title("🏗️ Run Regularization")
    if 'gdf' in st.session_state:
        t = st.slider("Simplification Tolerance", 0.1, 5.0, 1.0, 0.1)
        a = st.slider("Orthogonality Threshold", 1, 15, 5, 1)
        method = st.selectbox("Method", ["Standard", "Adaptive", "Hybrid"])
        if st.button("Regularize"):
            gdf = regularize_gdf(st.session_state.gdf, t, a, method.lower())
            st.session_state.result = gdf
            st.session_state.metrics = calculate_metrics(gdf)
            st.success("Regularization complete.")
    else:
        st.warning("Please upload data first.")

elif page == "📊 View Metrics":
    st.title("📊 View Metrics and Comparison")
    if 'result' in st.session_state:
        df = st.session_state.metrics
        st.dataframe(df)
        st.metric("Avg Vertex Reduction", f"{df['vertex_reduction'].mean():.2f}%")
        st.metric("Avg Area Retention", f"{df['area_ratio'].mean():.2f}%")
        show_map(st.session_state.result)
        idx = st.slider("Compare Feature", 0, len(st.session_state.result)-1, 0)
        show_comparison(st.session_state.result, idx)
    else:
        st.warning("Run regularization first.")

elif page == "🖼️ Image Processing":
    st.title("🖼️ GeoAI Image Processing")
    img = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"], key="img_proc")
    if img:
        image = Image.open(img)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(process_image(image, method="edge"), caption="Edge Detection", use_column_width=True)

elif page == "🧪 Compare Methods":
    st.title("🧪 Compare Regularization Methods on Image")

    uploaded_img = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"], key="compare_demo")
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        st.subheader("Comparison of Regularization Outputs")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Standard")
            output_std = process_image(image, method="edge")
            st.image(output_std, caption="Standard Output", use_column_width=True)

        with col2:
            st.markdown("### Adaptive")
            output_adaptive = process_image(image, method="edge")  # Simulating adaptive, use color/marker if needed
            st.image(output_adaptive, caption="Adaptive Output", use_column_width=True)

        with col3:
            st.markdown("### Hybrid")
            output_hybrid = process_image(image, method="edge")  # Simulating hybrid, use different processing if needed
            st.image(output_hybrid, caption="Hybrid Output", use_column_width=True)

        st.info("Note: All methods use the same edge detector for now. To differentiate, add distinct filters for each.")


elif page == "🎭 Mask Output":
    st.title("🎭 Generate Mask from Image")
    mask_img = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"], key="mask_input")
    if mask_img:
        image = Image.open(mask_img)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_column_width=True)
        col2.image(process_image(image, method="mask"), caption="Masked Output", use_column_width=True)
        if st.button("Download Mask"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                mask = process_image(image, method="mask")
                mask.save(tmp.name)
                st.download_button("Download Mask", tmp.name, file_name="mask_output.png", mime="image/png")