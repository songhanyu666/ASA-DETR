# ASA-DETR: 自适应稀疏注意力增强型RT-DETR遥感滑坡检测算法

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 简介

ASA-DETR是一种专为遥感滑坡检测设计的先进目标检测算法，基于RT-DETR改进。

### 核心创新

1. **LASAB（轻量级自适应稀疏注意力主干网络）** - 参数量减少30.8%
2. **CSPMFOK（跨阶段部分连接多尺度频率感知全向卷积模块）** - 空域-频域联合学习
3. **HMSAF（层次化多尺度注意力融合模块）** - 主动式自适应特征融合

### 性能指标

**RSLD-2K数据集：**
- mAP@0.5: 73.2% (↑3.7%)
- mAP@0.5:0.95: 52.5% (↑2.2%)
- Recall: 66.4% (↑4.5%)
- 参数量: 18.3M (↓7.6%)

**DOTAv2数据集（跨域泛化）：**
- mAP@0.5: 55.1%
- mAP@0.5:0.95: 35.9%

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n asa-detr python=3.9
conda activate asa-detr

# 安装依赖
pip install -r requirements.txt
```

### 推理测试

```bash
# 单张图像推理
python detect.py --weights weights/asa-detr.pt --source path/to/image.jpg

# 批量推理
python detect.py --weights weights/asa-detr.pt --source path/to/images/
```

### 训练模型

```bash
python train.py --cfg configs/asa-detr.yaml --data datasets/RSLD-2K/data.yaml --epochs 150 --batch-size 8
```

### 评估模型

```bash
python val.py --weights weights/asa-detr.pt --data datasets/RSLD-2K/data.yaml
```

## 📊 数据集

### RSLD-2K数据集

- **图像数量**：2,299张
- **标注数量**：6,545个滑坡目标
- **数据来源**：Google Earth、Planet Labs、Sentinel-2
- **覆盖区域**：中国西南山区、喜马拉雅地区、日本本州岛等

数据集结构：
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

## 📈 实验结果

### 与SOTA方法对比

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Params | FLOPs |
|------|---------|--------------|--------|-------|
| Faster-RCNN | 65.3% | 46.2% | 41.39M | 208G |
| YOLOv11m | 70.1% | 50.7% | 20.04M | 67.7G |
| RT-DETR-L | 71.6% | 51.8% | 33.0M | 103.5G |
| **ASA-DETR** | **73.2%** | **52.5%** | 18.3M | 72.4G |

### 消融实验

| 模型 | LASAB | CSPMFOK | HMSAF | mAP@0.5 |
|------|-------|---------|-------|---------|
| Baseline | ✗ | ✗ | ✗ | 69.5% |
| +LASAB | ✓ | ✗ | ✗ | 70.6% |
| +CSPMFOK | ✗ | ✓ | ✗ | 71.1% |
| +HMSAF | ✗ | ✗ | ✓ | 70.8% |
| **ASA-DETR** | ✓ | ✓ | ✓ | **73.2%** |

## 📁 项目结构

```
ASA-DETR/
├── configs/              # 配置文件
├── models/              # 模型定义
│   ├── backbone/        # LASAB主干网络
│   ├── neck/           # SOEFPN特征金字塔
│   └── head/           # RT-DETR检测头
├── utils/              # 工具函数
├── datasets/           # 数据集
├── weights/            # 模型权重
├── docs/               # 文档和图片
├── train.py           # 训练脚本
├── val.py             # 验证脚本
├── detect.py          # 推理脚本
└── requirements.txt   # 依赖列表
```

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{asa-detr2025,
  title={ASA-DETR: Adaptive Sparse Attention Enhanced RT-DETR for Remote Sensing Landslide Detection},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## 🙏 致谢

本项目基于以下优秀开源项目：
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：
- Issues: [GitHub Issues](https://github.com/yourusername/ASA-DETR/issues)
- Email: your.email@example.com

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！