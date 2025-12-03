# ASA-DETR: Adaptive Sparse Attention Enhanced RT-DETR for Remote Sensing Landslide Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Introduction

ASA-DETR is an advanced object detection algorithm specifically designed for remote sensing landslide detection, based on improvements to RT-DETR.

### Core Innovations

1. **LASAB (Lightweight Adaptive Sparse Attention Backbone)** - 30.8% parameter reduction
2. **CSPMFOK (Cross-Stage Partial Multi-scale Frequency-aware Omni-Kernel)** - Spatial-frequency joint learning
3. **HMSAF (Hierarchical Multi-Scale Attention Fusion)** - Active adaptive feature fusion

### Performance Metrics

**RSLD-2K Dataset:**
- mAP@0.5: 73.2% (↑3.7%)
- mAP@0.5:0.95: 52.5% (↑2.2%)
- Recall: 66.4% (↑4.5%)
- Parameters: 18.3M (↓7.6%)

**DOTAv2 Dataset (Cross-domain Generalization):**
- mAP@0.5: 55.1%
- mAP@0.5:0.95: 35.9%

## 🚀 Quick Start

### Environment Setup

```bash
# Create virtual environment
conda create -n asa-detr python=3.9
conda activate asa-detr

# Install dependencies
pip install -r requirements.txt
```

### Inference

```bash
# Single image inference
python detect.py --weights weights/asa-detr.pt --source path/to/image.jpg

# Batch inference
python detect.py --weights weights/asa-detr.pt --source path/to/images/
```

### Training

```bash
python train.py --cfg configs/asa-detr.yaml --data datasets/RSLD-2K/data.yaml --epochs 150 --batch-size 8
```

### Evaluation

```bash
python val.py --weights weights/asa-detr.pt --data datasets/RSLD-2K/data.yaml
```

## � Dataset

### RSLD-2K Dataset

- **Images**: 2,299
- **Annotations**: 6,545 landslide targets
- **Sources**: Google Earth, Planet Labs, Sentinel-2
- **Coverage**: Southwest China mountains, Himalayan region, Honshu Island Japan, etc.

Dataset structure:
```
RSLD-2K/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**Download Dataset:**
- **Baidu Netdisk**: https://pan.baidu.com/s/1iYlw3FvCyWV81jxMnjZBOQ?pwd=ap6u (Code: ap6u)
- See [Dataset Documentation](docs/DATASET.md) for more details

## 📈 Experimental Results

### Comparison with SOTA Methods

| Model | mAP@0.5 | mAP@0.5:0.95 | Params | FLOPs |
|-------|---------|--------------|--------|-------|
| Faster-RCNN | 65.3% | 46.2% | 41.39M | 208G |
| YOLOv11m | 70.1% | 50.7% | 20.04M | 67.7G |
| RT-DETR-L | 71.6% | 51.8% | 33.0M | 103.5G |
| **ASA-DETR** | **73.2%** | **52.5%** | 18.3M | 72.4G |

### Ablation Study

| Model | LASAB | CSPMFOK | HMSAF | mAP@0.5 |
|-------|-------|---------|-------|---------|
| Baseline | ✗ | ✗ | ✗ | 69.5% |
| +LASAB | ✓ | ✗ | ✗ | 70.6% |
| +CSPMFOK | ✗ | ✓ | ✗ | 71.1% |
| +HMSAF | ✗ | ✗ | ✓ | 70.8% |
| **ASA-DETR** | ✓ | ✓ | ✓ | **73.2%** |

## � Project Structure

```
ASA-DETR/
├── configs/              # Configuration files
├── models/              # Model definitions
│   ├── backbone/        # LASAB backbone
│   ├── neck/           # SOEFPN feature pyramid
│   └── head/           # RT-DETR detection head
├── utils/              # Utility functions
├── datasets/           # Datasets
├── weights/            # Model weights
├── docs/               # Documentation
├── train.py           # Training script
├── val.py             # Validation script
├── detect.py          # Inference script
└── requirements.txt   # Dependencies
```

## 📝 Citation

If this project helps your research, please cite:

```bibtex
@article{asa-detr2025,
  title={ASA-DETR: Adaptive Sparse Attention Enhanced RT-DETR for Remote Sensing Landslide Detection},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## � Acknowledgments

This project is based on the following excellent open-source projects:
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions:
- Issues: [GitHub Issues](https://github.com/songhanyu666/ASA-DETR/issues)
- Email: songhanyu2025@163.com

---

⭐ If this project helps you, please give us a Star!