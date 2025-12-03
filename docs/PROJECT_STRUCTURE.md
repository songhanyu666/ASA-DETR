# ASA-DETR Project Structure

## 📁 Complete Directory Structure

```
ASA-DETR/
├── configs/                    # Configuration files
│   └── asa-detr.yaml          # ASA-DETR model configuration
│
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── asa_detr.py            # ASA-DETR main model
│   ├── backbone/              # Backbone network
│   │   ├── __init__.py
│   │   └── lasab.py           # LASAB lightweight adaptive sparse attention backbone
│   ├── neck/                  # Feature pyramid network
│   │   ├── __init__.py
│   │   └── soefpn.py          # SOEFPN small object enhanced feature pyramid
│   └── head/                  # Detection head (to be implemented)
│       ├── __init__.py
│       └── rtdetr_decoder.py  # RT-DETR decoder
│
├── utils/                      # Utility functions (to be created)
│   ├── __init__.py
│   ├── dataset.py             # Dataset loading
│   ├── loss.py                # Loss functions
│   ├── metrics.py             # Evaluation metrics
│   ├── general.py             # General utilities
│   └── visualize.py           # Visualization tools
│
├── datasets/                   # Datasets directory
│   └── RSLD-2K/               # RSLD-2K dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── weights/                    # Model weights directory
│   ├── .gitkeep
│   └── README.md              # Weights documentation
│
├── docs/                       # Documentation directory
│   ├── DATASET.md             # Dataset documentation (Chinese)
│   ├── DATASET_EN.md          # Dataset documentation (English)
│   ├── PROJECT_STRUCTURE.md   # Project structure (Chinese)
│   ├── PROJECT_STRUCTURE_EN.md # Project structure (English)
│   └── images/                # Documentation images
│
├── runs/                       # Results directory
│   ├── train/                 # Training results
│   ├── val/                   # Validation results
│   └── detect/                # Detection results
│
├── train.py                    # Training script
├── val.py                      # Validation script
├── detect.py                   # Inference script
├── export.py                   # Model export script (to be created)
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── LICENSE                     # Open source license
├── README.md                   # Project documentation (Chinese)
└── README_EN.md                # Project documentation (English)
```

## 📝 Core Files Description

### 1. Configuration Files

#### `configs/asa-detr.yaml`
Contains all configuration information including model architecture, training parameters, and data augmentation.

### 2. Model Files

#### `models/asa_detr.py`
ASA-DETR main model, integrating LASAB, SOEFPN, and RT-DETR Decoder.

#### `models/backbone/lasab.py`
Lightweight Adaptive Sparse Attention Backbone, including:
- **PCDSA**: Partial Channel Dynamic Sparse Attention
- **CADG**: Content-Adaptive Dynamic Gating
- **DSAB**: Dynamic Sparse Attention Block
- **AAE-CSP**: Adaptive Attention Enhanced Cross-Stage Partial Fusion

#### `models/neck/soefpn.py`
Small Object Enhanced Feature Pyramid Network, including:
- **CSPMFOK**: Cross-Stage Partial Multi-scale Frequency-aware Omni-Kernel
- **HMSAF**: Hierarchical Multi-Scale Attention Fusion
- **SPDConv**: Space-to-Depth Convolution

### 3. Training and Inference Scripts

#### `train.py`
Model training script, supporting:
- Training from scratch
- Loading pretrained weights
- Distributed training
- Mixed precision training

#### `val.py`
Model validation script, computing:
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

#### `detect.py`
Model inference script, supporting:
- Single image inference
- Batch image inference
- Video inference

## 🔧 Features to be Implemented

### High Priority
- [ ] RT-DETR Decoder implementation
- [ ] Loss function implementation
- [ ] Dataset loader implementation
- [ ] Evaluation metrics implementation

### Medium Priority
- [ ] Visualization tools (heatmaps, feature maps, etc.)
- [ ] Model export (ONNX, TensorRT)
- [ ] Training logs and TensorBoard
- [ ] Data augmentation strategies

### Low Priority
- [ ] Model pruning and quantization
- [ ] Distributed training support
- [ ] Automatic hyperparameter search
- [ ] Web demo interface

## 📊 Code Organization Principles

1. **Modular Design**: Each component is independently implemented for easy testing and reuse
2. **Configuration-Driven**: All hyperparameters managed through configuration files
3. **Well-Documented**: Each module has detailed docstrings
4. **Code Standards**: Follows PEP 8 coding standards
5. **Version Control**: Uses Git for version management

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda create -n asa-detr python=3.9
conda activate asa-detr
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Download RSLD-2K dataset
# Extract to datasets/RSLD-2K/
```

### 3. Train Model
```bash
python train.py --cfg configs/asa-detr.yaml --data datasets/RSLD-2K/data.yaml
```

### 4. Test Model
```bash
python val.py --weights weights/asa-detr.pt --data datasets/RSLD-2K/data.yaml
```

### 5. Inference
```bash
python detect.py --weights weights/asa-detr.pt --source path/to/image.jpg
```

## 📖 Related Documentation

- [Dataset Documentation](DATASET_EN.md)
- [Training Guide](TRAINING_EN.md)
- [API Documentation](API_EN.md)
- [FAQ](FAQ_EN.md)

## 🤝 Contribution Guidelines

Welcome to submit Issues and Pull Requests!

1. Fork this project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Contact

For questions, please contact:
- Issues: [GitHub Issues](https://github.com/songhanyu666/ASA-DETR/issues)
- Email: your.email@example.com