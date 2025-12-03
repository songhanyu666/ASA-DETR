# ASA-DETR Quick Start Guide

This guide will help you quickly get started with ASA-DETR for remote sensing landslide detection.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [FAQ](#faq)

## 🔧 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (≥8GB VRAM recommended)
- **CPU**: Intel Core i7 or higher
- **RAM**: ≥16GB
- **Storage**: ≥50GB available space

### Software Requirements
- **OS**: Linux / Windows / macOS
- **Python**: 3.8 - 3.10
- **CUDA**: 11.3 or higher (for GPU)
- **cuDNN**: 8.2 or higher

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/songhanyu666/ASA-DETR.git
cd ASA-DETR
```

### 2. Create Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n asa-detr python=3.9
conda activate asa-detr
```

**Using venv:**
```bash
python -m venv asa-detr-env
source asa-detr-env/bin/activate  # Linux/macOS
# or
asa-detr-env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python test_env.py
```

If you see the following output, installation is successful:
```
✓ Python version: 3.9.x
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ CUDA version: 11.8
✓ GPU device: NVIDIA GeForce RTX 4070
✓ All dependencies installed successfully!
```

## 📊 Data Preparation

### 1. Download RSLD-2K Dataset

**Baidu Netdisk**: https://pan.baidu.com/s/1iYlw3FvCyWV81jxMnjZBOQ?pwd=ap6u
**Extraction Code**: ap6u
**File Size**: Approximately 2.5GB

### 2. Dataset Directory Structure

After extraction, place the dataset in the following location:

```
ASA-DETR/
└── datasets/
    └── RSLD-2K/
        ├── images/
        │   ├── train/      # 1609 training images
        │   ├── val/        # 460 validation images
        │   └── test/       # 230 test images
        ├── labels/
        │   ├── train/      # Training labels (YOLO format)
        │   ├── val/        # Validation labels
        │   └── test/       # Test labels
        └── data.yaml       # Dataset configuration
```

### 3. Verify Dataset

```bash
python utils/check_dataset.py --data datasets/RSLD-2K/data.yaml
```

## 🚀 Training

### 1. Train from Scratch

```bash
python train.py \
    --cfg configs/asa-detr.yaml \
    --data datasets/RSLD-2K/data.yaml \
    --epochs 150 \
    --batch-size 8 \
    --device 0
```

### 2. Train with Pretrained Weights

```bash
python train.py \
    --cfg configs/asa-detr.yaml \
    --data datasets/RSLD-2K/data.yaml \
    --weights weights/asa-detr-pretrained.pt \
    --epochs 150 \
    --batch-size 8 \
    --device 0
```

### 3. Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py \
    --cfg configs/asa-detr.yaml \
    --data datasets/RSLD-2K/data.yaml \
    --epochs 150 \
    --batch-size 16 \
    --device 0,1
```

### 4. Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cfg` | Model configuration file path | configs/asa-detr.yaml |
| `--data` | Dataset configuration file path | - |
| `--weights` | Pretrained weights path | None |
| `--epochs` | Number of training epochs | 150 |
| `--batch-size` | Batch size | 8 |
| `--img-size` | Input image size | 640 |
| `--device` | GPU device to use | 0 |
| `--workers` | Number of data loading workers | 8 |
| `--project` | Project name for saving results | runs/train |
| `--name` | Experiment name | exp |

### 5. Monitor Training

During training, the following will be automatically saved:
- **Weight files**: `runs/train/exp/weights/`
  - `best.pt` - Best model
  - `last.pt` - Last epoch model
- **Training logs**: `runs/train/exp/results.txt`
- **Visualizations**: `runs/train/exp/`
  - `results.png` - Training curves
  - `confusion_matrix.png` - Confusion matrix

View training progress with TensorBoard:
```bash
tensorboard --logdir runs/train
```

## ✅ Validation

### 1. Validate Trained Model

```bash
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data datasets/RSLD-2K/data.yaml \
    --batch-size 8 \
    --device 0
```

### 2. Validate Pretrained Model

```bash
python val.py \
    --weights weights/asa-detr.pt \
    --data datasets/RSLD-2K/data.yaml \
    --batch-size 8 \
    --device 0
```

### 3. Validation Results

After validation, you will see output like:
```
Class     Images  Instances      P      R  mAP50  mAP50-95
all         230        653  0.753  0.664  0.732     0.525
landslide   230        653  0.753  0.664  0.732     0.525
```

## 🔍 Inference

### 1. Single Image Inference

```bash
python detect.py \
    --weights weights/asa-detr.pt \
    --source path/to/image.jpg \
    --conf-thres 0.25 \
    --device 0
```

### 2. Batch Image Inference

```bash
python detect.py \
    --weights weights/asa-detr.pt \
    --source path/to/images/ \
    --conf-thres 0.25 \
    --device 0
```

### 3. Video Inference

```bash
python detect.py \
    --weights weights/asa-detr.pt \
    --source path/to/video.mp4 \
    --conf-thres 0.25 \
    --device 0
```

### 4. Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Model weights path | weights/asa-detr.pt |
| `--source` | Input source (image/folder/video) | - |
| `--conf-thres` | Confidence threshold | 0.25 |
| `--iou-thres` | NMS IoU threshold | 0.45 |
| `--img-size` | Inference image size | 640 |
| `--device` | GPU device to use | 0 |
| `--save-txt` | Save results as txt | False |
| `--save-conf` | Save confidence scores | False |
| `--project` | Project name for saving results | runs/detect |
| `--name` | Experiment name | exp |

### 5. Inference Results

Results are saved in `runs/detect/exp/`:
- Detection result images (with bounding boxes)
- Detection result text files (if using `--save-txt`)

## ❓ FAQ

### Q1: CUDA out of memory error

**Solutions:**
- Reduce `batch-size`
- Reduce `img-size`
- Use mixed precision training: `--amp`

### Q2: Slow training speed

**Solutions:**
- Increase `workers` count
- Use faster data augmentation
- Enable mixed precision training
- Use multi-GPU training

### Q3: Poor model accuracy

**Solutions:**
- Increase training epochs
- Adjust learning rate
- Use data augmentation
- Use pretrained weights

### Q4: Dataset format error

**Solutions:**
- Check if labels are in YOLO format
- Run `python utils/check_dataset.py` to verify dataset
- Refer to [Dataset Documentation](DATASET_EN.md)

### Q5: Inaccurate inference results

**Solutions:**
- Adjust `conf-thres` confidence threshold
- Adjust `iou-thres` NMS threshold
- Use larger `img-size`
- Use best model weights `best.pt`

## 📚 More Resources

- [Full Documentation](../README_EN.md)
- [Dataset Documentation](DATASET_EN.md)
- [Project Structure](PROJECT_STRUCTURE_EN.md)
- [API Documentation](API_EN.md)
- [FAQ](FAQ_EN.md)

## 🤝 Get Help

If you encounter issues:
1. Check [FAQ](FAQ_EN.md)
2. Search [GitHub Issues](https://github.com/songhanyu666/ASA-DETR/issues)
3. Submit a new Issue
4. Email: your.email@example.com

## 📝 Next Steps

- [ ] Try training on your own dataset
- [ ] Tune hyperparameters for better performance
- [ ] Export model to ONNX format
- [ ] Deploy model to production

Enjoy using ASA-DETR! 🎉