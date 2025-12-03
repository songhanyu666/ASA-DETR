# ASA-DETR 项目结构说明

## 📁 完整目录结构

```
ASA-DETR/
├── configs/                    # 配置文件目录
│   └── asa-detr.yaml          # ASA-DETR模型配置
│
├── models/                     # 模型定义目录
│   ├── __init__.py
│   ├── asa_detr.py            # ASA-DETR主模型
│   ├── backbone/              # 主干网络
│   │   ├── __init__.py
│   │   └── lasab.py           # LASAB轻量级自适应稀疏注意力主干
│   ├── neck/                  # 特征金字塔网络
│   │   ├── __init__.py
│   │   └── soefpn.py          # SOEFPN小目标增强特征金字塔
│   └── head/                  # 检测头（待实现）
│       ├── __init__.py
│       └── rtdetr_decoder.py  # RT-DETR解码器
│
├── utils/                      # 工具函数目录（待创建）
│   ├── __init__.py
│   ├── dataset.py             # 数据集加载
│   ├── loss.py                # 损失函数
│   ├── metrics.py             # 评估指标
│   ├── general.py             # 通用工具
│   └── visualize.py           # 可视化工具
│
├── datasets/                   # 数据集目录
│   └── RSLD-2K/               # RSLD-2K数据集
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
├── weights/                    # 模型权重目录
│   ├── .gitkeep
│   └── README.md              # 权重说明
│
├── docs/                       # 文档目录
│   ├── DATASET.md             # 数据集说明
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明
│   ├── TRAINING.md            # 训练指南（待创建）
│   └── images/                # 文档图片
│
├── runs/                       # 运行结果目录
│   ├── train/                 # 训练结果
│   ├── val/                   # 验证结果
│   └── detect/                # 检测结果
│
├── train.py                    # 训练脚本
├── val.py                      # 验证脚本
├── detect.py                   # 推理脚本
├── export.py                   # 模型导出脚本（待创建）
│
├── requirements.txt            # Python依赖
├── .gitignore                 # Git忽略文件
├── LICENSE                     # 开源许可证
└── README.md                   # 项目说明文档
```

## 📝 核心文件说明

### 1. 配置文件

#### `configs/asa-detr.yaml`
包含模型架构、训练参数、数据增强等所有配置信息。

### 2. 模型文件

#### `models/asa_detr.py`
ASA-DETR主模型，整合LASAB、SOEFPN和RT-DETR Decoder。

#### `models/backbone/lasab.py`
轻量级自适应稀疏注意力主干网络，包含：
- **PCDSA**: 部分通道动态稀疏注意力
- **CADG**: 内容自适应动态门控
- **DSAB**: 动态稀疏注意力块
- **AAE-CSP**: 自适应注意力增强跨阶段部分融合

#### `models/neck/soefpn.py`
小目标增强特征金字塔网络，包含：
- **CSPMFOK**: 跨阶段部分连接多尺度频率感知全向卷积
- **HMSAF**: 层次化多尺度注意力融合
- **SPDConv**: 空间到深度卷积

### 3. 训练和推理脚本

#### `train.py`
模型训练脚本，支持：
- 从头训练
- 预训练权重加载
- 分布式训练
- 混合精度训练

#### `val.py`
模型验证脚本，计算：
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

#### `detect.py`
模型推理脚本，支持：
- 单张图像推理
- 批量图像推理
- 视频推理

## 🔧 待实现功能

### 高优先级
- [ ] RT-DETR Decoder实现
- [ ] 损失函数实现
- [ ] 数据集加载器实现
- [ ] 评估指标实现

### 中优先级
- [ ] 可视化工具（热力图、特征图等）
- [ ] 模型导出（ONNX、TensorRT）
- [ ] 训练日志和TensorBoard
- [ ] 数据增强策略

### 低优先级
- [ ] 模型剪枝和量化
- [ ] 分布式训练支持
- [ ] 自动超参数搜索
- [ ] Web演示界面

## 📊 代码组织原则

1. **模块化设计**: 每个组件独立实现，便于测试和复用
2. **配置驱动**: 所有超参数通过配置文件管理
3. **文档完善**: 每个模块都有详细的docstring
4. **代码规范**: 遵循PEP 8编码规范
5. **版本控制**: 使用Git进行版本管理

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n asa-detr python=3.9
conda activate asa-detr
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 下载RSLD-2K数据集
# 解压到datasets/RSLD-2K/
```

### 3. 训练模型
```bash
python train.py --cfg configs/asa-detr.yaml --data datasets/RSLD-2K/data.yaml
```

### 4. 测试模型
```bash
python val.py --weights weights/asa-detr.pt --data datasets/RSLD-2K/data.yaml
```

### 5. 推理
```bash
python detect.py --weights weights/asa-detr.pt --source path/to/image.jpg
```

## 📖 相关文档

- [数据集说明](DATASET.md)
- [训练指南](TRAINING.md)
- [API文档](API.md)
- [常见问题](FAQ.md)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📧 联系方式

如有问题，请通过以下方式联系：
- Issues: [GitHub Issues](https://github.com/yourusername/ASA-DETR/issues)
- Email: your.email@example.com