# 🍃 Leaf Disease Classification using Triton Inference Server and Grad-CAM

![Triton Workflow](Triton_Workflow.png)

This project demonstrates an end-to-end workflow for segmenting and classifying plant leaf diseases using a YOLOv8 segmentation model, ResNet-50 classifier, and NVIDIA Triton Inference Server. The entire pipeline is production-ready and simulates real-world conditions where leaves need to be segmented before classification due to non-uniform backgrounds.

---

## 🔍 Motivation

In real-world settings, images are rarely perfectly cropped or clean. Instead of training directly on well-cropped images, we simulate deployment-like conditions by segmenting leaves from complex backgrounds first. This improves generalization and model focus, as supported by Grad-CAM visualizations.

- 🔸 `Original_Grid.jpg` shows original vs segmented images.
- 🔸 `GradCam_Grid.jpg` clearly shows how background-free images help the model focus better on leaf regions.

---

## 🧠 Architecture Overview

The project is structured as follows:

1. **YOLOv8 Segmentation**
   - Segments leaf regions from raw images.
2. **Post-processing Module**
   - Extracts crops from segmentation masks and resizes them.
3. **ResNet-50 Classifier**
   - Classifies segmented crops into: **Black Rot**, **Healthy**, and **Scab**.
4. **GradCAM**
   - Validates model attention using heatmaps.

---

---

## 📦 Dependencies

**Python setup** (for preprocessing, data sorting, visualization):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Triton inference setup**:

- Docker image: `nvcr.io/nvidia/tritonserver:23.10-py3`
- Classification training: `nvcr.io/nvidia/pytorch:23.10-py3`

---

## 🧪 Dataset

**Source**: [PlantVillage Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

> Arun Pandian & G. Geetharamani, Mendeley Data, 2019  
> DOI: [10.17632/tywbtsjrjv.1](https://doi.org/10.17632/tywbtsjrjv.1)

- Annotated via [Roboflow](https://roboflow.com/) for YOLOv8 segmentation.

---

## 🧩 YOLOv8 Segmentation (Ultralytics)

Install Ultralytics:

```bash
pip install ultralytics
```

Train segmentation model:

```bash
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=40 imgsz=640
```

Export to ONNX-TensorRT:

```bash
yolo export model=best.pt format=onnx_trt
```

---

## 🧪 TensorRT Conversion (inside Docker)

```bash
trtexec \
  --onnx=yolo_best.onnx \
  --minShapes=input:1x3x640x640 \
  --optShapes=input:15x3x640x640 \
  --maxShapes=input:30x3x640x640 \
  --fp16 \
  --workspace=4096 \
  --saveEngine=model.plan
```

> ⚠️ **Tip:** Use [Netron](https://netron.app/) to verify input layer names before conversion.

---

## 🧪 Classification with ResNet-50

Used [`NVIDIA DeepLearningExamples`](https://github.com/NVIDIA/DeepLearningExamples)  
Path: `PyTorch/Classification/ConvNets`

### 📁 Data Format

```
train/
 ├── 0/  (Black Rot)
 ├── 1/  (Healthy)
 └── 2/  (Scab)

val/
 ├── 0/
 ├── 1/
 └── 2/
```

### 🏋️ Training

```bash
python3 main.py /home/genai/Dataset \
  --arch resnet50 \
  --epochs 50 \
  --batch-size 4 \
  --lr 0.01 \
  --training-only \
  --num_classes 3 \
  --data-backend pytorch \
  --workspace checkpoints \
  --raport-file experiment_raport.json \
  --print-freq 1
```

### 🧪 Evaluation & Export

```bash
python3 classify.py --arch resnet50 \
  --pretrained-from-file resnet50_inference_weights.pth \
  --image Blackrot.png --num_classes 3
```

```bash
python3 model2onnx.py --arch resnet50 \
  --device cuda --image-size 224 \
  --batch-size 1 \
  --output resnet50_dynamic.onnx \
  --pretrained-from-file resnet50_inference_weights.pth \
  --num-classes 3
```

> ⚠️ **Note:** Batch size is set to dynamic inside the script.

---

## 🧠 Embedding Visualization (FiftyOne)

Used `fiftyone_embeddings.py` to visualize class separability.  
Output image: `FiftyOne_Embeddings_Visualization.png` confirms strong class clustering.

---

## 🚀 Triton Inference Server

Run Triton server:

```bash
tritonserver --model-repository=./model_repository/
```

Required dependencies (install inside Docker if needed):

```bash
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
pip install opencv-python
```

---

## 🎯 Grad-CAM Visualization

Used `grad_cam.py` and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) to verify focus regions.

- **Input**: `Healthy_Original.jpg`
- **Segmented**: `Healthy_Segmented.png`
- **Grad-CAM Grid**: `GradCam_Grid.jpg`

---

## 🎬 Demo

See `Demo.gif` for an end-to-end pipeline walkthrough.



