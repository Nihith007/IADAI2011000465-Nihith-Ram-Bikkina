# MAchine-learning-SA
# ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation

**Student Name:** _[your name]_
**Registration Number:** _[your reg. number]_
**CRS:** Artificial Intelligence
**Course:** Machine Learning and Deep Learning
**School:** _[your school name]_
**Live App:** _[Streamlit Cloud link once deployed]_

> **Submission checklist (per assignment brief — not part of the README content itself, just a reminder):**
> - [ ] Repo named `IADAI201(YourStudentID)-YourName`
> - [ ] Repo access granted to `ai.assignments@wacpinternational.org`
> - [ ] Code files (.ipynb, .py) uploaded
> - [ ] Dataset (or a representative sample) uploaded to the repo
> - [ ] Functional Streamlit Cloud link works when tested fresh
> - [ ] Submission PDF includes: GitHub link, full name, registration number, CRS name, course name, school name

## Problem Statement

UrbanFlow AI Pvt. Ltd. needs a system that analyzes parking lot images and determines, slot by slot,
whether each space is occupied or empty — surfacing real-time availability, congestion level, and a
recommendation, through a Streamlit dashboard.

## Approach

- **Input:** parking lot image
- **Output:** slot-wise bounding-box detections (Occupied / Empty), counts, occupancy %, congestion
  level, recommendation
- **Method:** YOLOv8 (Ultralytics) object detector, fine-tuned on a Roboflow export of the PKLot
  dataset that already provides bounding-box labels in COCO JSON format. The Streamlit app runs the
  trained model directly on each uploaded image and draws real detected bounding boxes — no grid
  guessing or manual slot coordinates needed.

## Data Preparation

- **Dataset:** PKLot (Roboflow export, COCO-annotated), split into `train/`, `valid/`, `test/`
- Total training images: 8,691; total training bounding-box annotations: 497,856
- Classes: `spaces` (parent/lot-level box, excluded from slot counting), `space-empty`, `space-occupied`
- COCO annotations converted to YOLO `.txt` label format (one file per image, normalized boxes)
- Native train/valid/test split as provided by the Roboflow export

## Model & Training

- Base model: YOLOv8n (`yolov8n.pt`, pretrained on COCO), fine-tuned end-to-end
- Epochs: 15 (early stopping patience 5), image size 416, batch size 32
- Optimizer/loss: Ultralytics default (auto optimizer selection, combined box + classification + DFL loss)

## Results

- **mAP50:** 0.994 | **mAP50-95:** 0.918
- **Precision:** space-empty 0.996, space-occupied 0.997
- **Recall:** space-empty 0.991, space-occupied 0.997
- Evaluated on the held-out test set: 1,242 images, 70,684 slot instances
- Confusion matrix: see `confusion_matrix.png`
- Training curves: see `training_curves.png`
- The model performs strongly on both classes with near-identical precision/recall, suggesting the
  empty vs. occupied distinction is visually clear-cut in this dataset across the weather/lighting
  conditions and camera angles present in PKLot. _[Add your own note here on any failure cases you
  noticed when testing the app — e.g. heavy shadows, partially visible slots at frame edges.]_

## System Logic

Occupancy percentage is computed from slot predictions and mapped to:
- **Low congestion:** < 40%
- **Moderate congestion:** 40–75%
- **High congestion:** > 75%

Recommendation: "Slots available — proceed to park" unless occupancy ≥ 90%, in which case
"Parking nearly full — try another location."

## Repository Structure

```
├── ParkVision_AI_Colab.ipynb   # data prep, training, evaluation (run in Google Colab)
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── model/
│   └── parkvision_yolo_best.pt # trained YOLOv8 weights (exported from the notebook)
├── data_sample/                # small sample of the dataset (not the full PKLot set)
├── training_curves.png
├── confusion_matrix.png
└── README.md
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud: _[link]_

## Screenshots

_[Insert screenshots of the running app here: upload screen, annotated output, metrics panel.]_

## Key Research Findings That Shaped This Project

_[Write 3-5 sentences here — this is explicitly required by the assignment brief as its own section,
separate from the reference list below. Example points to adapt in your own words:]_
- Object detection (YOLO) outperforms crop-then-classify pipelines for slot-level parking occupancy
  when bounding-box-labeled data is available, since it locates and classifies slots in one pass
  rather than depending on pre-known slot coordinates.
- The PKLot dataset's coverage of multiple weather conditions (sunny, cloudy, rainy) is important for
  generalization — a model trained only on clear-weather images tends to misclassify shadowed or
  wet-surface slots.
- Small/distant objects lose detail when images are downscaled to a fixed training resolution;
  training and inference at higher resolution (640px+) improves detection of slots far from the camera.
- _[Add your own finding — e.g. what you observed about your specific model's failure cases, or what
  a specific reference paper below informed about your approach.]_

## References

- [PKLot: A Robust Dataset for Parking Lot Classification](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
- [Vision-Based Parking Slot Detection using Deep Learning](https://www.mdpi.com/1424-8220/23/15/6869)
- _[Add 1-2 more references from the assignment brief's suggested reading list — "Deep Learning Based
  Smart Parking Occupancy Detection using Computer Vision" and "Real-Time Parking Occupancy Detection
  using CNN and Computer Vision" are both listed there; search for the exact papers to link correctly.]_
- [YOLO Object Detection Documentation (Ultralytics Official)](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io)
