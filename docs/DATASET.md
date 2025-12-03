# RSLD-2K Dataset Documentation

## Dataset Overview

RSLD-2K (Remote Sensing Landslide Detection 2K) is a large-scale dataset specifically constructed for remote sensing landslide detection tasks.

### Basic Information

- **Number of Images**: 2,299
- **Number of Annotations**: 6,545 landslide targets
- **Average Instances per Image**: 2.85
- **Data Sources**: Google Earth, Planet Labs, Sentinel-2
- **Coverage Areas**: Southwest China mountains, Himalayan region, Honshu Island Japan, and other landslide-prone areas

### Dataset Split

| Subset | Images | Annotations | Ratio |
|--------|--------|-------------|-------|
| Training | 1,609 | 4,581 | 70% |
| Validation | 460 | 1,311 | 20% |
| Testing | 230 | 653 | 10% |

## Dataset Features

### 1. Diversity

- **Geographic Diversity**: Covers multiple geographic regions and geological backgrounds
- **Seasonal Diversity**: Includes landslide samples under different seasons and climate conditions
- **Scale Diversity**: Contains large, medium, and small-scale landslide targets

### 2. Annotation Quality

- Annotated by 3 professionals with geological hazard identification experience
- Cross-validated to ensure annotation quality
- Rectangular bounding boxes annotate the main body of landslides

### 3. Annotation Guidelines

1. Bounding boxes must completely frame the main body of landslides (including landslide walls, tongues, and accumulation areas)
2. For landslides partially obscured by vegetation or clouds, boundaries are inferred based on terrain features
3. Exclude suspected landslides and uncertain targets, only annotate clearly identifiable landslide bodies

## Dataset Structure

```
RSLD-2K/
├── images/
│   ├── train/          # Training images
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── val/            # Validation images
│   │   ├── 000001.jpg
│   │   └── ...
│   └── test/           # Testing images
│       ├── 000001.jpg
│       └── ...
├── labels/
│   ├── train/          # Training annotations (YOLO format)
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
│   ├── val/            # Validation annotations
│   │   ├── 000001.txt
│   │   └── ...
│   └── test/           # Testing annotations
│       ├── 000001.txt
│       └── ...
└── data.yaml           # Dataset configuration file
```

## Annotation Format

### YOLO Format

Each txt file corresponds to one image, with each line representing one target:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Class ID (landslide is 0)
- `x_center, y_center`: Bounding box center coordinates (normalized to 0-1)
- `width, height`: Bounding box width and height (normalized to 0-1)

Example:
```
0 0.5123 0.4567 0.2345 0.1890
0 0.7234 0.6789 0.1567 0.1234
```

### COCO Format

Also supports COCO format annotations with more detailed information:

```json
{
    "images": [...],
    "annotations": [...],
    "categories": [
        {"id": 1, "name": "landslide"}
    ]
}
```

## Dataset Download

### Baidu Netdisk

**Link**: https://pan.baidu.com/s/1iYlw3FvCyWV81jxMnjZBOQ?pwd=ap6u
**Extraction Code**: ap6u

**File**: RSLD-2K.zip (Approximately 2.5GB)

After downloading, extract the zip file to the `datasets/` directory:
```bash
unzip RSLD-2K.zip -d datasets/
```

## Dataset Usage

### 1. Download Dataset

```bash
# Download and extract dataset
wget [download_link] -O RSLD-2K.zip
unzip RSLD-2K.zip -d datasets/
```

### 2. Configure data.yaml

```yaml
# datasets/RSLD-2K/data.yaml
path: datasets/RSLD-2K
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['landslide']  # Class names
```

### 3. Data Loading Example

```python
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class LandslideDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        
        # Read annotations
        label_path = os.path.join(self.label_dir, 
                                  self.img_files[idx].replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], 
                            dtype=np.float32)
        
        return img, labels

# Usage example
dataset = LandslideDataset('datasets/RSLD-2K/images/train',
                          'datasets/RSLD-2K/labels/train')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## Dataset Statistics

### Landslide Size Distribution

| Size Category | Count | Percentage |
|--------------|-------|------------|
| Small (<32²) | 1,856 | 28.4% |
| Medium (32²-96²) | 3,245 | 49.6% |
| Large (>96²) | 1,444 | 22.0% |

### Geographic Distribution

| Region | Images | Percentage |
|--------|--------|------------|
| Southwest China | 1,124 | 48.9% |
| Himalayan Region | 687 | 29.9% |
| Honshu Island Japan | 488 | 21.2% |

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{rsld2k2025,
  title={RSLD-2K: A Large-Scale Remote Sensing Landslide Detection Dataset},
  author={Your Name},
  year={2025}
}
```

## License

This dataset is for academic research only and may not be used for commercial purposes.

## Contact

For dataset-related questions, please contact:
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/songhanyu666/ASA-DETR/issues)