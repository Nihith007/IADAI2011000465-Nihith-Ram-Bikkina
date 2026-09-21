# IADAI201(YourStudentID)-YourName

# ParkVision AI: Intelligent Urban Parking Analytics & Space Optimisation Platform

# Candidate Name - [your name]

# Candidate Registration Number - [your registration number]

# CRS Name: Artificial Intelligence

# Course Name - Machine Learning and Deep Learning

# School name - [your school name]

# Summative Assessment

# Project Overview

This project is a computer vision system that analyzes parking lot images and determines, slot by slot, whether each space is occupied or empty. Built using YOLOv8 (Ultralytics) for object detection and deployed as a Streamlit web app, the platform draws color-coded bounding boxes over detected slots, calculates real-time availability metrics, classifies congestion level, and gives users a simple recommendation on whether to proceed to the lot or look elsewhere.

# Problem Statement

Urban areas face significant parking challenges — drivers often spend several minutes searching for an empty spot, which increases traffic congestion, fuel consumption, and driver frustration. Existing systems rarely give slot-level, real-time visibility into parking availability. This project addresses that gap using a computer vision model that identifies each individual parking slot in an image and classifies its occupancy status, then converts that into actionable insights.

# Objectives

* Develop a slot-level parking occupancy detection system using object detection
* Generate real-time availability metrics: total, occupied, and available slots
* Provide visual smart overlays with color-coded bounding boxes (green = empty, red = occupied)
* Calculate occupancy percentage and classify congestion as low, moderate, or high
* Generate simple, actionable recommendations based on availability
* Deploy the system as an interactive Streamlit dashboard, accessible publicly via Streamlit Cloud

# Research Summary

The development process involved studying vision-based parking slot detection approaches, comparing crop-then-classify pipelines (e.g. MobileNet/EfficientNet on per-slot images) against direct object detection (YOLO) on full parking-lot images. Research into the PKLot dataset's structure and the tradeoffs between classification and detection approaches for slot-level tasks informed the final design. Because the available dataset (a Roboflow export of PKLot) already provided bounding-box annotations in COCO format, a YOLO-based object detection approach was chosen over classification, since it detects and classifies every slot in a single pass without needing separately known slot coordinates. Additional research into image resolution's effect on small-object detection informed decisions during training and inference tuning — downscaling wide, high-altitude parking images to a small fixed resolution was found to hurt detection of distant/small slots, which shaped both the training image size and an adjustable inference-resolution control in the final app.

# Data Preparation

* Dataset: PKLot (Roboflow export, COCO-annotated), pre-split into train/valid/test folders
* Training set: 8,691 images, 497,856 bounding-box annotations
* Test set: 1,242 images, 70,684 slot instances
* Classes: `space-empty`, `space-occupied` (plus a `spaces` parent/lot-level box, excluded from slot counting)
* COCO JSON annotations converted to YOLO `.txt` label format (normalized bounding boxes)
* Native train/valid/test split as provided by the dataset export

# Model Configuration

* Base model: YOLOv8n (Ultralytics), pretrained on COCO, fine-tuned on the PKLot slot-detection data
* Training run: 30 epochs, image size 640px, batch size 16, early stopping patience 10
* An initial faster run (15 epochs, 416px) was used to validate the pipeline end-to-end before committing to the longer, higher-resolution training run for the final model
* Inference resolution is configurable in the deployed app (416–1280px) — raising it improves detection of small/distant slots in wide aerial shots at the cost of slower inference

# Sample Outputs and Validation

* mAP50: 0.994 | mAP50-95: 0.918
* Precision: `space-empty` 0.996, `space-occupied` 0.997
* Recall: `space-empty` 0.991, `space-occupied` 0.997
* Evaluated on the held-out test set (1,242 images, 70,684 annotated slots)
* The model performs strongly and near-identically on both classes. Testing on a wide, high-altitude drone photo (outside the typical PKLot camera framing) showed reduced detection coverage on distant/small slots at default settings — resolved by raising the inference resolution, confirming the research finding above about downscaling and small-object detection.

# Web Application Features

* Image upload for any parking lot photo
* Real YOLO-detected bounding boxes overlaid on the image — green for empty, red for occupied
* Live metrics: total slots, occupied slots, available slots, occupancy percentage
* Congestion level classification: Low (<40%), Moderate (40–75%), High (>75%)
* Automatic recommendation: proceed to park, or try another location
* Adjustable confidence threshold and inference resolution in the sidebar
* Clean, responsive Streamlit interface

# Deployment

The application was built with Streamlit and uses a YOLOv8 model (Ultralytics) loaded directly from a committed `.pt` weights file — no external API key required. The complete project was hosted on GitHub and deployed on Streamlit Community Cloud.

* Train the model in Google Colab (see `ParkVision_AI_Colab.ipynb`) and export the trained weights (`best.pt`) to Google Drive.
* Download the weights file and place it in the project at `model/parkvision_yolo_best.pt`.
* Add a `.gitattributes` file marking `*.pt` as binary, to prevent Git from corrupting the weights file via line-ending conversion.
* Add a `packages.txt` file listing `libgl1` and `libglib2.0-0` — required system libraries for OpenCV (a dependency of Ultralytics) to run on Streamlit Cloud's container.
* Add a `runtime.txt` pinning Python 3.11 for a stable, well-supported build environment.
* Upload all project files to a GitHub repository, including the code, trained model, and requirements file.
* Log in to Streamlit Cloud (streamlit.io/cloud) with GitHub, select the repository, branch, and `app.py` as the main file, and deploy.
* Test the deployed app with a fresh parking-lot image to confirm detections render correctly.

**Live app link:** [your Streamlit Cloud link here]

# Screenshots

[Insert screenshots of the running app here — upload screen, annotated detection output, live metrics panel, and the congestion/recommendation display]

# Example Test Images

Upload a clear, moderately-zoomed parking lot photo similar to PKLot's camera framing.

[Insert screenshot of result]

Upload a wide, high-altitude aerial/drone shot of a large lot.

[Insert screenshot of result, noting the inference-resolution setting used]

Upload an image with heavy shadows or partial occlusion (cars parked close together).

[Insert screenshot of result]

Upload a rainy or overcast-weather parking lot image.

[Insert screenshot of result]

# Key Research Findings That Shaped This Project

* Object detection (YOLO) outperforms crop-then-classify pipelines for slot-level parking occupancy when bounding-box-labeled data is available, since it locates and classifies slots in one pass rather than depending on pre-known slot coordinates.
* PKLot's coverage of multiple weather conditions (sunny, cloudy, rainy) matters for generalization — a model trained only on clear-weather images tends to misclassify shadowed or wet-surface slots.
* Small/distant objects lose detail when images are downscaled to a fixed training resolution; training and inference at higher resolution (640px+) improves detection of slots far from the camera.
* [Add your own additional finding here based on your own testing.]

# References

* [PKLot: A Robust Dataset for Parking Lot Classification](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
* [Vision-Based Parking Slot Detection using Deep Learning](https://www.mdpi.com/1424-8220/23/15/6869)
* [YOLO Object Detection Documentation (Ultralytics Official)](https://docs.ultralytics.com/)
* [Streamlit Documentation](https://docs.streamlit.io)
* [Add 1-2 more references from the assignment brief's suggested reading list.]

# Repository Structure

```
├── ParkVision_AI_Colab.ipynb   # data prep, training, evaluation (run in Google Colab)
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── packages.txt                # system dependencies for Streamlit Cloud (OpenCV support)
├── runtime.txt                 # pins Python version for Streamlit Cloud
├── .gitattributes              # ensures the model file is treated as binary
├── model/
│   └── parkvision_yolo_best.pt # trained YOLOv8 weights
├── data_sample/                # small sample of the dataset (not the full PKLot set)
├── training_curves.png
├── confusion_matrix.png
└── README.md
```

# How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
