# RSLD-2K数据集说明

## 数据集概述

RSLD-2K（Remote Sensing Landslide Detection 2K）是专为遥感滑坡检测任务构建的大规模数据集。

### 基本信息

- **图像数量**: 2,299张
- **标注数量**: 6,545个滑坡目标
- **平均每张图像**: 2.85个滑坡实例
- **数据来源**: Google Earth、Planet Labs、Sentinel-2
- **覆盖区域**: 中国西南山区、喜马拉雅地区、日本本州岛等滑坡多发地带

### 数据集划分

| 子集 | 图像数量 | 标注数量 | 比例 |
|------|---------|---------|------|
| 训练集 | 1,609 | 4,581 | 70% |
| 验证集 | 460 | 1,311 | 20% |
| 测试集 | 230 | 653 | 10% |

## 数据集特点

### 1. 多样性

- **地理多样性**: 覆盖多个地理区域和地质构造背景
- **季节多样性**: 包含不同季节和气候条件下的滑坡样本
- **尺度多样性**: 包含大、中、小不同尺度的滑坡目标

### 2. 标注质量

- 由3名具有地质灾害识别经验的专业人员完成标注
- 经过交叉验证确保标注质量
- 采用矩形包围盒标注滑坡体的主体范围

### 3. 标注准则

1. 矩形包围盒需完整框定滑坡体的主体范围（包括滑坡壁、滑坡舌和堆积区）
2. 对于部分被植被或云层遮挡的滑坡，根据地形特征推断其边界
3. 排除疑似滑坡和不确定目标，仅标注清晰可辨的滑坡体

## 数据集结构

```
RSLD-2K/
├── images/
│   ├── train/          # 训练集图像
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── val/            # 验证集图像
│   │   ├── 000001.jpg
│   │   └── ...
│   └── test/           # 测试集图像
│       ├── 000001.jpg
│       └── ...
├── labels/
│   ├── train/          # 训练集标注（YOLO格式）
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
│   ├── val/            # 验证集标注
│   │   ├── 000001.txt
│   │   └── ...
│   └── test/           # 测试集标注
│       ├── 000001.txt
│       └── ...
└── data.yaml           # 数据集配置文件
```

## 标注格式

### YOLO格式

每个txt文件对应一张图像，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

其中：
- `class_id`: 类别ID（滑坡为0）
- `x_center, y_center`: 边界框中心点坐标（归一化到0-1）
- `width, height`: 边界框宽度和高度（归一化到0-1）

示例：
```
0 0.5123 0.4567 0.2345 0.1890
0 0.7234 0.6789 0.1567 0.1234
```

### COCO格式

也支持COCO格式的标注，包含更详细的信息：

```json
{
    "images": [...],
    "annotations": [...],
    "categories": [
        {"id": 1, "name": "landslide"}
    ]
}
```

## 数据集下载

### 方式1: 百度网盘

链接: [待补充]
提取码: [待补充]

### 方式2: Google Drive

链接: [待补充]

### 方式3: 学术网盘

链接: [待补充]

## 数据集使用

### 1. 下载数据集

```bash
# 下载并解压数据集
wget [下载链接] -O RSLD-2K.zip
unzip RSLD-2K.zip -d datasets/
```

### 2. 配置data.yaml

```yaml
# datasets/RSLD-2K/data.yaml
path: datasets/RSLD-2K
train: images/train
val: images/val
test: images/test

nc: 1  # 类别数
names: ['landslide']  # 类别名称
```

### 3. 数据加载示例

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
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        
        # 读取标注
        label_path = os.path.join(self.label_dir, 
                                  self.img_files[idx].replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], 
                            dtype=np.float32)
        
        return img, labels

# 使用示例
dataset = LandslideDataset('datasets/RSLD-2K/images/train',
                          'datasets/RSLD-2K/labels/train')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## 数据统计

### 滑坡尺寸分布

| 尺寸类别 | 数量 | 占比 |
|---------|------|------|
| 小目标 (<32²) | 1,856 | 28.4% |
| 中目标 (32²-96²) | 3,245 | 49.6% |
| 大目标 (>96²) | 1,444 | 22.0% |

### 地理区域分布

| 区域 | 图像数量 | 占比 |
|------|---------|------|
| 中国西南山区 | 1,124 | 48.9% |
| 喜马拉雅地区 | 687 | 29.9% |
| 日本本州岛 | 488 | 21.2% |

## 引用

如果使用本数据集，请引用：

```bibtex
@dataset{rsld2k2025,
  title={RSLD-2K: A Large-Scale Remote Sensing Landslide Detection Dataset},
  author={Your Name},
  year={2025}
}
```

## 许可证

本数据集仅供学术研究使用，不得用于商业目的。

## 联系方式

如有数据集相关问题，请联系：
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/ASA-DETR/issues)