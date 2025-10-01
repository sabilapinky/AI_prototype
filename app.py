# app.py — Graph digitizer (upload → calibrate → digitize → CSV)
# Requires:
#   pip install streamlit pillow opencv-python numpy pandas streamlit-drawable-canvas

import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

try:
    import cv2
except Exception as e:
    cv2 = None

st.set_page_config(page_title="Graph Digitizer", layout="wide")
st.title("📈 Graph Digitizer: Upload → Calibrate → Extract → CSV")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Workflow")
    st.markdown("**1. Upload** → **2. Calibrate** → **3. Digitize** → **4. Export**")

    scale_type = st.selectbox("Axis scale", ["linear", "semilogx", "semilogy", "loglog"])
    mode = st.radio("Digitize mode", ["Manual clicks", "Auto by color (HSV)"])

    st.markdown("---")
    st.caption("Tip: Use browser zoom 110–125% for easier clicking.")

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload a graph image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload a graph image to begin.")
    st.stop()

img = Image.open(uploaded).convert("RGBA")
W, H = img.size

# Canvas size (keep big but bounded)
canvas_h = min(800, int(H * (800 / max(H, 800))))
scale = canvas_h / H
canvas_w = int(W * scale)

# ---------------- Canvas ----------------
st.subheader("1) Canvas")
canvas = st_canvas(
    fill_color="rgba(255,0,0,0.0)",
    stroke_width=6,
    stroke_color="#000000",
    background_image=img,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="point",
    key="canvas",
)

# Gather clicks (circles placed on canvas)
clicks = []
if canvas.json_data is not None:
    objs = canvas.json_data.get("objects", [])
    # Canvas coordinates are in the resized space; convert back to original pixel space.
    for o in objs:
        if o.get("type") == "circle":
            # st_canvas places a circle with center at (left + radius, top + radius)
            radius = o.get("radius", 0)
            cx = (o["left"] + radius) / scale
            cy = (o["top"] + radius) / scale
            clicks.append((cx, cy))

st.info("Click order for calibration: x1, x2, y1, y2. After that, add curve points if using Manual mode.")

# ---------------- Calibration inputs ----------------
st.subheader("2) Calibration (Axis mapping)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**X-axis** — click **two** points on the x-axis (left→right), then enter their values.")
    x_val1 = st.text_input("Value at x1 (first x-axis click)", value="0")
    x_val2 = st.text_input("Value at x2 (second x-axis click)", value="1")
with c2:
    st.markdown("**Y-axis** — click **two** points on the y-axis (bottom→top), then enter their values.")
    y_val1 = st.text_input("Value at y1 (first y-axis click)", value="0")
    y_val2 = st.text_input("Value at y2 (second y-axis click)", value="1")

def _to_float(s):
    try:
        return float(s)
    except:
        return None

def build_mapping(pix1, pix2, v1, v2, is_log):
    """
    Build pixel -> value mapping for one axis.
    pix1, pix2 are pixel positions along that axis (x for X-axis, y for Y-axis).
    If is_log, map in log10 space.
    Returns (fwd, inv, desc) or (None, None, None) on failure.
    """
    if abs(pix2 - pix1) < 1e-9:
        return None, None, None
    if is_log:
        if v1 is None or v2 is None or v1 <= 0 or v2 <= 0:
            return None, None, None
        L1, L2 = np.log10(v1), np.log10(v2)
        m = (L2 - L1) / (pix2 - pix1)
        b = L1 - m * pix1
        fwd = lambda p: 10 ** (m * p + b)
        inv = lambda v: (np.log10(v) - b) / m if (v is not None and v > 0 and abs(m) > 1e-12) else np.nan
        desc = f"log10(v) = {m:.6g} * p + {b:.6g}  →  v = 10^(...)"
    else:
        m = (v2 - v1) / (pix2 - pix1)
        b = v1 - m * pix1
        fwd = lambda p: m * p + b
        inv = lambda v: (v - b) / m if abs(m) > 1e-12 else np.nan
        desc = f"v = {m:.6g} * p + {b:.6g}"
    return fwd, inv, desc

valid_nums = all(_to_float(v) is not None for v in [x_val1, x_val2, y_val1, y_val2])
have_4_clicks = len(clicks) >= 4

if not have_4_clicks or not valid_nums:
    st.warning("Add at least 4 clicks (x1, x2, y1, y2) and ensure all four axis values are numeric.")
    st.stop()

# Extract calibration pixels
x1p, x2p = clicks[0][0], clicks[1][0]     # use x-coordinates for x-axis calibration
y1p, y2p = clicks[2][1], clicks[3][1]     # use y-coordinates for y-axis calibration (top-origin)

# Determine log/linear for axes from scale_type
x_log = scale_type in ["loglog", "semilogx"]
y_log = scale_type in ["loglog", "semilogy"]

xfwd, xinv, xdesc = build_mapping(x1p, x2p, float(x_val1), float(x_val2), x_log)
yfwd, yinv, ydesc = build_mapping(y1p, y2p, float(y_val1), float(y_val2), y_log)

if xfwd is None or yfwd is None:
    st.error("Calibration failed — check that clicks are distinct and (for log) values are > 0.")
    st.stop()

st.markdown("**Mapping equations**")
st.code(f"X: {xdesc}\nY: {ydesc}")

# ---------------- Digitizing ----------------
st.subheader("3) Digitize")

def clicks_to_dataframe(all_clicks):
    """Convert post-calibration clicks (from index 4 onward) into data values."""
    pts = all_clicks[4:] if len(all_clicks) > 4 else []
    rows = []
    for (px, py) in pts:
        x_val = xfwd(px)
        y_val = yfwd(py)  # note: py already in original pixel space
        rows.append((x_val, y_val, px, py))
    df_local = pd.DataFrame(rows, columns=["x", "y", "x_pix", "y_pix"])
    return df_local

def auto_color_points(image_rgba, hsv_lo, hsv_hi, min_area=15, max_points=200):
    """
    Auto-pick curve-like pixels based on HSV color threshold.
    - image_rgba: PIL RGBA image
    - hsv_lo/hi: tuples like (H(0-179), S(0-255), V(0-255))
    Returns list of (x_pix, y_pix) in original image coordinates (not scaled).
    """
    if cv2 is None:
        return []

    bgr = cv2.cvtColor(np.array(image_rgba), cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)

    mask = cv2.inRange(hsv, lo, hi)

    # Clean up mask a bit
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find connected components as blobs along the curve
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))

    # If there are too many points, sort by x then take evenly spaced
    if len(pts) > max_points:
        pts_sorted = sorted(pts, key=lambda t: t[0])
        idx = np.linspace(0, len(pts_sorted) - 1, max_points).astype(int)
        pts = [pts_sorted[i] for i in idx]

    return pts

result_df = None

if mode == "Manual clicks":
    st.caption("Keep clicking on the curve in the canvas above. Those clicks (after the first four) will become data points.")
    df = clicks_to_dataframe(clicks)
    if df.empty:
        st.info("No curve points yet. Add some clicks after the first 4 calibration clicks.")
    else:
        result_df = df

else:  # Auto by color
    if cv2 is None:
        st.error("OpenCV not installed. Run: pip install opencv-python")
    else:
        st.caption("Pick HSV range for your curve color. Try narrow H and high S to isolate vivid lines.")
        colA, colB = st.columns(2)
        with colA:
            h_lo = st.slider("Hue low (0–179)", 0, 179, 0)
            s_lo = st.slider("Sat low (0–255)", 0, 255, 50)
            v_lo = st.slider("Val low (0–255)", 0, 255, 50)
        with colB:
            h_hi = st.slider("Hue high (0–179)", 0, 179, 179)
            s_hi = st.slider("Sat high (0–255)", 0, 255, 255)
            v_hi = st.slider("Val high (0–255)", 0, 255, 255)

        min_area = st.number_input("Min blob area (pixels)", value=15, step=1)
        max_pts = st.number_input("Max points", value=200, step=10)

        if st.button("🔍 Auto-detect points by color"):
            cand_pts = auto_color_points(img, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi), int(min_area), int(max_pts))
            if len(cand_pts) == 0:
                st.warning("No points detected with this HSV range. Adjust sliders and try again.")
            else:
                # Map to values
                rows = []
                for (px, py) in cand_pts:
                    rows.append((xfwd(px), yfwd(py), px, py))
                df_auto = pd.DataFrame(rows, columns=["x", "y", "x_pix", "y_pix"])
                # Sort by x for nicer CSV
                df_auto.sort_values("x", inplace=True, ignore_index=True)
                result_df = df_auto

# ---------------- Results & Export ----------------
st.subheader("4) Results & Export")
if result_df is not None and not result_df.empty:
    st.dataframe(result_df, use_container_width=True, height=300)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="digitized.csv", mime="text/csv")

    recipe = {
        "image_size": [W, H],
        "scale_type": scale_type,
        "calibration": {
            "x": {"pix": [x1p, x2p], "val": [float(x_val1), float(x_val2)], "equation": xdesc},
            "y": {"pix": [y1p, y2p], "val": [float(y_val1), float(y_val2)], "equation": ydesc},
        },
        "mode": mode,
    }
    st.download_button("⬇️ Download recipe.json",
                       data=json.dumps(recipe, indent=2).encode("utf-8"),
                       file_name="recipe.json",
                       mime="application/json")
else:
    st.info("No data yet. Add manual curve clicks or run auto-detect.")
