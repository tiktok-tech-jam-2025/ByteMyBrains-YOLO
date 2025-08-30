# YOLO For Non-text Sensitive Information Detection

This repository contains code and resources for **fine-tuning YOLO** (Ultralytics) to automatically detect **sensitive information** in images, such as:

- Human faces  
- License plates  
- ID cards / documents  
- Phone numbers  
- Email addresses  

The goal is to provide a practical pipeline for detecting and masking sensitive data in photos or scanned documents.

---

## 📂 Repository Structure

```

.
├── yolo12n.pt            # Middle checkpoint
├── yolo12n.zip           # Middle checkpoint as CoreML
├── openimage.pt          # Final checkpoint
├── yolo12n_finetuned.zip # Final checkpoint as CoreML
├── yolo_test.py          # Script for checking YOLO performance
├── yolo_finetune.ipynb   # Main notebook for fine-tuning YOLO
├── README.md             # Project documentation
└── yolo12n.pt

````

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/tiktok-tech-jam-2025/ByteMyBrains-YOLO.git
cd ByteMyBrains-YOLO
````

### 2. Install dependencies

You’ll need Python 3.8+ and [Ultralytics YOLO](https://docs.ultralytics.com).

```bash
pip install ultralytics fiftyone pycocotools
```

Other common packages:

```bash
pip install torch torchvision opencv-python
```

### 3. Dataset Preparation

* This project supports **YOLO** and **COCO-format datasets**.
* You can use [`fiftyone`](https://voxel51.com/docs/fiftyone/) to download balanced subsets from OpenImages or COCO.
* Example data.yaml:

```yaml
path: /content/full_data/yolo_dataset
train: images/train
val: images/val

nc: 5
names: ["face", "plate", "id_card", "phone", "document"]
```

Place your dataset under the right path in the notebook.

---

## 📓 Fine-tuning

Run the Jupyter notebook:

```bash
jupyter notebook yolo_finetune.ipynb
```

Inside the notebook:

* Loads a pretrained YOLOv8/v12 model (`yolov8n.pt`, `yolov12n.pt`, etc.)
* Trains on your sensitive data detection dataset
* Logs training runs in `runs/detect/`

---

## 📊 Evaluation

![alt text](<assets/confusion_matrix .png>)
![alt text](<assets/results.png>)


## 🛡️ Applications

* Automatic **blurring or masking** of sensitive data in images
* Privacy-compliant data preparation
* Document redaction systems

---

## 📜 License

This project is provided under the **MIT License**.
Please ensure compliance with data privacy regulations when using this code.

---

## ✨ Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [FiftyOne](https://github.com/voxel51/fiftyone) for dataset curation
* [COCO](https://cocodataset.org) and [OpenImages](https://storage.googleapis.com/openimages/web/index.html) datasets


