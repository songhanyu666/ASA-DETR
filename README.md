# ASA-DETR

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
