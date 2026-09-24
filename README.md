# IADAI2011000465-Nihith Ram Bikkina

# ParkVision AI: Intelligent Urban Parking Analytics & Space Optimisation Platform

# Candidate Name - Nihith Ram Bikkina

# Candidate Registration Number - 1000465

# CRS Name: Artificial Intelligence

# Course Name - Machine Learning and Deep Learning

# School name - Birla Open Minds International School, Kollur

# Summative Assessment

# ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

## 1. Overview

ParkVision AI is a computer vision system that analyzes photos of parking lots and automatically identifies which spaces are occupied and which are empty. It calculates real-time occupancy metrics and generates simple recommendations for drivers, all through a web app built with Streamlit.

**Problem Statement:** Urban areas face serious parking challenges — drivers routinely spend several minutes searching for an empty spot, increasing traffic congestion, fuel consumption, and driver frustration. This is a slot-level problem, not a simple whole-image classification: each individual parking space in a lot must be located and classified, under real-world conditions that include shadows, partial occlusion (cars parked close together), and varying weather.

**System Definition:**
- **Input:** a photo of a parking lot (uploaded by the user)
- **Output:** slot-wise status for every visible space (Occupied / Empty), plus total/occupied/available counts, occupancy percentage, a congestion level, and a recommendation

**Real-World Relevance:** by giving drivers (or a city dashboard) immediate, slot-level visibility into parking availability, this system directly reduces the time spent circling for a spot — cutting unnecessary traffic and emissions in dense urban areas.

## 2. Research Findings that Influenced the Project

Before building the system, I reviewed existing research on parking occupancy detection to understand common approaches and their trade-offs:

- Deep learning-based smart parking systems generally fall into two categories: **classification-based** approaches (classifying individually cropped images of each parking slot as occupied/empty) and **detection-based** approaches (using object detectors like YOLO to locate and classify all slots directly in a full scene image). Classification approaches are simpler but require known slot locations in advance. Detection-based approaches are more flexible — they work on new camera angles without manual slot marking — but need higher-resolution input to reliably distinguish many small, densely packed objects.
- The dataset available to me (a Roboflow export of PKLot) already provided bounding-box annotations in COCO format, which made a detection-based (YOLO) approach the natural fit — it detects and classifies every slot in a single pass, with no need to separately mark slot coordinates in advance.
- Vision-based parking slot detection research highlights that detection accuracy is strongly affected by image resolution relative to object (slot) size — a finding I confirmed directly during this project (see Section 7, Testing & Limitations).
- The PKLot dataset itself was designed specifically to support robust parking classification research across varying lighting and weather conditions (sunny, cloudy, rainy), which is why it remains a standard benchmark for this type of task.

These findings directly shaped my decision to use a YOLOv8 object detector rather than a per-slot classifier, and later informed how I diagnosed and fixed reduced detection coverage on wide, high-altitude test images.

## 3. Academic References and Key Sources

- [PKLot: A Robust Dataset for Parking Lot Classification](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
- [Vision-Based Parking Slot Detection using Deep Learning](https://www.mdpi.com/1424-8220/23/15/6869)
- [YOLO Object Detection Documentation (Ultralytics Official)](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 4. Data Preparation

### Dataset
Source: [PKLot dataset](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf) via a Roboflow export — parking lot images captured under varying weather conditions (sunny, cloudy, rainy), pre-split into `train/`, `valid/`, and `test/` folders, each with a `_annotations.coco.json` file.

### Preparation Process

- Inspected the COCO annotation files to identify the actual classes present: `spaces` (a parent/lot-level box covering the whole lot, not an individual slot), `space-empty`, and `space-occupied`.
- Excluded `spaces` from slot-counting logic since it isn't an individual parking space.
- Converted COCO-format bounding-box annotations into YOLO's normalized label format (`class x_center y_center width height`) for every image in each split, writing one `.txt` label file per image.
- Built a `data.yaml` config pointing to the train/valid/test folders and the detected class names.
- Used the dataset's native train/valid/test split rather than re-splitting manually.

### Data Cleaning
- Verified each image had a corresponding, correctly formatted annotation entry before conversion.
- Confirmed no zero-width/zero-height boxes were produced during the COCO-to-YOLO coordinate conversion.

### Dataset Size
- Training set: 8,691 images, 497,856 bounding-box annotations
- Test set: 1,242 images, 70,684 slot instances

## 5. Models Used

### YOLOv8n (Detection Approach)
- **Architecture:** YOLOv8n (Ultralytics), the smallest/fastest variant in the YOLOv8 family, trained from pretrained COCO weights.
- **Training parameters:** 30 epochs (early stopping patience 10), image size 640×640, batch size 16.
- **Classes:** `space-empty`, `space-occupied` (`spaces` is present in the data but excluded from occupancy counting)
- **Technique:** Full-image object detection — the model directly outputs bounding boxes and class labels for every parking slot visible in an uploaded photo, without requiring pre-marked slot positions.
- An initial faster run (15 epochs, image size 416×416) was used first to validate the full pipeline end-to-end before committing to the longer, higher-resolution training run used for the final model.

## 6. Metrics and Results

### YOLOv8n Detector — Test Set Results
Evaluated on the held-out test set (1,242 images, 70,684 annotated slots):

| Metric | space-empty | space-occupied |
|---|---|---|
| Precision | 0.996 | 0.997 |
| Recall | 0.991 | 0.997 |

**mAP50:** 0.994 | **mAP50-95:** 0.918

Confusion matrix and training curves: see `confusion_matrix.png` and `training_curves.png`.

The model performs strongly and near-identically on both classes, indicating the empty-vs-occupied distinction is visually clear-cut across the weather/lighting conditions and camera angles present in PKLot.

## 7. Testing & Limitations

The system was tested on unseen images from the PKLot test split, as well as external test photos not drawn from the training distribution.

**Key finding — resolution-dependent detection coverage:** When I tested the deployed app on a wide, high-altitude drone photo of a large parking lot (a much wider framing than PKLot's typical camera shots), detection coverage dropped noticeably for rows of cars further from the camera — those slots appeared as very few pixels once the image was downscaled to the model's inference resolution, making them too small to detect confidently. Rows closer to the camera, with larger apparent slot size, were detected reliably.

This matches the research finding in Section 2 about resolution's effect on small-object detection. The fix implemented in the app is an adjustable inference-resolution setting (416–1280px) in the sidebar — raising it for wide/high-altitude images preserves more detail for distant slots, at the cost of slower inference. This is a dataset/scale limitation rather than a flaw in the model architecture: PKLot's training images don't include this style of very-wide aerial framing, so the model generalizes less well to it by default.

## 8. System Logic

- **Occupancy % =** (occupied slots ÷ total slots) × 100
- **Congestion levels:**
  - Low: occupancy < 40%
  - Moderate: occupancy 40–75%
  - High: occupancy > 75%
- **Recommendations:**
  - ≥90% occupied: "Parking nearly full — try another location."
  - <90% occupied: "Slots available — proceed to park."

## 9. Web App

Built with [Streamlit](https://docs.streamlit.io/). Users upload a photo of a parking lot and the app:
1. Runs the trained YOLOv8n model to detect and classify every visible slot.
2. Draws color-coded bounding boxes (green = empty, red = occupied) directly on the image.
3. Displays total/occupied/available slot counts, occupancy percentage, congestion level, and a recommendation — all updating live as the confidence threshold and inference resolution are adjusted in the sidebar.

Live Streamlit app link: https://machine-learning-sa-zdze2bbmdrbbzmvoxkp387.streamlit.app/  

## 10. Screenshots

<img width="1201" height="850" alt="image" src="https://github.com/user-attachments/assets/795c8e83-077b-4600-8f2b-439de8610458" />
<img width="1200" height="866" alt="image" src="https://github.com/user-attachments/assets/67d80761-0165-4688-81da-f07a7bcbbf6d" />
<img width="1201" height="861" alt="image" src="https://github.com/user-attachments/assets/7059ab01-3e50-4879-bea0-5c4681931c74" />


## 11. Deployment & Repository

The application uses a YOLOv8 model (Ultralytics) loaded directly from a committed `.pt` weights file — no external API key required.

- Trained the model in Google Colab (`ParkVision_AI_Colab.ipynb`) and exported the weights (`best.pt`) to Google Drive.
- Placed the downloaded weights file in the project at `model/parkvision_yolo_best.pt`.
- Added a `.gitattributes` file marking `*.pt` as binary, to prevent Git from corrupting the weights file via line-ending conversion.
- Added a `packages.txt` file listing `libgl1` and `libglib2.0-0` — required system libraries for OpenCV (a dependency of Ultralytics) on Streamlit Cloud's container.
- Added a `runtime.txt` pinning Python 3.11 for a stable build environment.
- Uploaded all project files to GitHub — code (.ipynb, .py), the trained model, a representative sample of the dataset, and this README — then deployed via Streamlit Cloud (streamlit.io/cloud) by selecting the repository, branch, and `app.py` as the main file.
- Repository access granted to `ai.assignments@wacpinternational.org` as required for submission.

**Repository structure:**
```
├── ParkVision_AI_Colab.ipynb   # data prep, training, evaluation
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── packages.txt
├── runtime.txt
├── .gitattributes
├── model/
│   └── parkvision_yolo_best.pt
├── data_sample/                # representative sample of the dataset
├── training_curves.png
├── confusion_matrix.png
└── README.md
```
