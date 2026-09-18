"""
ParkVision AI — Streamlit dashboard
Upload a parking lot image and see real YOLO-detected slot-level occupancy,
live metrics, and a recommendation.
"""

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")

MODEL_PATH = "model/parkvision_yolo_best.pt"

# Must match CLASS_NAMES printed by the notebook's Step 2, in the same order (index = class id).
CLASS_NAMES = ["spaces", "space-empty", "space-occupied"]  # "spaces" is a parent/lot box, not a slot
EMPTY_NAMES = [n for n in CLASS_NAMES if "empty" in n.lower() or "free" in n.lower() or "vacant" in n.lower()]
OCCUPIED_NAMES = [n for n in CLASS_NAMES if "occup" in n.lower() or "busy" in n.lower() or "full" in n.lower()]
SLOT_NAMES = EMPTY_NAMES + OCCUPIED_NAMES  # only these count as individual parking slots

CONF_THRESHOLD = 0.25


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def parking_insights(detected_class_names):
    total = len(detected_class_names)
    occupied = sum(1 for c in detected_class_names if c in OCCUPIED_NAMES)
    available = sum(1 for c in detected_class_names if c in EMPTY_NAMES)
    occupancy_pct = (occupied / total * 100) if total > 0 else 0

    if occupancy_pct < 40:
        congestion = "Low"
    elif occupancy_pct <= 75:
        congestion = "Moderate"
    else:
        congestion = "High"

    recommendation = (
        "Parking nearly full — try another location."
        if occupancy_pct >= 90
        else "Slots available — proceed to park."
    )

    return {
        "total_slots": total,
        "occupied_slots": occupied,
        "available_slots": available,
        "occupancy_pct": round(occupancy_pct, 1),
        "congestion_level": congestion,
        "recommendation": recommendation,
    }


def annotate_image(image: Image.Image, boxes, class_names) -> Image.Image:
    """Draws color-coded bounding boxes: green for empty, red for occupied."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = class_names[int(box.cls[0])]
        conf = float(box.conf[0])
        color = (30, 200, 30) if cls_name in EMPTY_NAMES else (220, 30, 30)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 12)), f"{cls_name} {conf:.2f}", fill=color)

    return img


def main():
    st.title("🅿️ ParkVision AI")
    st.caption("Intelligent Urban Parking Analytics & Space Optimisation — UrbanFlow AI")

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Detection confidence threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05)
        infer_size = st.select_slider(
            "Inference resolution", options=[416, 640, 960, 1280], value=960
        )
        st.markdown(
            "Lower the confidence threshold to catch more (possibly noisier) detections. "
            "Raise the inference resolution for images with many small/distant slots "
            "(wide aerial shots, large lots) — it's slower but catches detail the model "
            "would otherwise lose when the image gets downscaled."
        )

    uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a parking lot image to get started.")
        return

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model at {MODEL_PATH}. Make sure it's committed to the repo. ({e})")
        return

    image = Image.open(uploaded_file)

    with st.spinner("Detecting parking slots..."):
        results = model.predict(source=image, conf=conf, imgsz=infer_size, verbose=False)
        all_boxes = results[0].boxes
        # Drop the "spaces" parent/lot box (or any other non-slot class) — only keep individual slots
        boxes = [b for b in all_boxes if CLASS_NAMES[int(b.cls[0])] in SLOT_NAMES]

    detected_names = [CLASS_NAMES[int(b.cls[0])] for b in boxes]
    annotated = annotate_image(image, boxes, CLASS_NAMES)
    insights = parking_insights(detected_names)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected (green = empty, red = occupied)")
        st.image(annotated, use_container_width=True)

    st.subheader("Live Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Slots", insights["total_slots"])
    m2.metric("Occupied", insights["occupied_slots"])
    m3.metric("Available", insights["available_slots"])
    m4.metric("Occupancy %", f"{insights['occupancy_pct']}%")

    congestion_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}[insights["congestion_level"]]
    st.markdown(f"### Congestion Level: {congestion_color} **{insights['congestion_level']}**")
    st.markdown(f"### Recommendation: **{insights['recommendation']}**")

    if insights["total_slots"] == 0:
        st.warning(
            "No slots detected. Try lowering the confidence threshold in the sidebar, "
            "or upload a clearer parking-lot image similar to the training data."
        )


if __name__ == "__main__":
    main()
