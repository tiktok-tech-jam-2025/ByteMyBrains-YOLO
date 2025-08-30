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
└── README.md             # Project documentation

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
pip install torch torchvision torchaudio
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

For this application, the dataset used for fine-tuning the model is the Open Images V7 dataset, choosing the classes     "Human face",
"Mobile phone",
"Laptop",
"Computer keyboard",
"Computer monitor",
"Book",
"Envelope",
"Camera",
"Tablet computer",
"Briefcase". These are additional classes added to the original list of classes of the base YOLOv12n checkpoint model.

---

# 📊 YOLOv12n Evaluation Results

![alt text](<assets/confusion_matrix .png>)
![alt text](<assets/results.png>)

**Environment**  
Ultralytics 8.3.189 🚀
Python-3.12.11
torch-2.8.0+cu128
CUDA:0 (Tesla T4, 15095MiB)

**Model Summary**  
YOLOv12n summary (fused):
159 layers, 2,558,678 parameters, 0 gradients, 6.3 GFLOPs

---

## 📈 Validation Metrics

| Class               | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---------------------|--------|-----------|-----------|--------|--------|-----------|
| **all**             | 641    | 1627      | 0.553     | 0.542  | 0.541  | 0.422     |
| Human face          | 192    | 343       | 0.470     | 0.338  | 0.355  | 0.229     |
| Mobile phone        | 105    | 147       | 0.748     | 0.668  | 0.734  | 0.638     |
| Laptop              | 52     | 70        | 0.706     | 0.414  | 0.511  | 0.382     |
| Computer keyboard   | 63     | 67        | 0.542     | 0.582  | 0.582  | 0.406     |
| Computer monitor    | 73     | 100       | 0.724     | 0.340  | 0.524  | 0.372     |
| Book                | 95     | 720       | 0.417     | 0.272  | 0.235  | 0.136     |
| Envelope            | 7      | 7         | 0.533     | 0.714  | 0.721  | 0.638     |
| Camera              | 142    | 153       | 0.765     | 0.618  | 0.682  | 0.476     |
| Tablet computer     | 11     | 13        | 0.220     | 0.615  | 0.297  | 0.280     |
| Briefcase           | 5      | 7         | 0.405     | 0.857  | 0.769  | 0.666     |

---


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


