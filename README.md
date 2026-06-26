# XH-202625 光学遥感卫星陆上目标检测识别竞赛方案

## 概述

本项目针对"挑战杯"揭榜挂帅竞赛题目XH-202625，解决光学遥感卫星图像中的不均衡小样本目标检测问题。

### 核心挑战

1. **长尾分布极端**: 航母(17框) vs FA-18(2147框)，比例126:1
2. **小样本类别**: HM(17), LQS(30), KC-10(262), TU-160(361), FSC(402)
3. **大尺幅推理**: 10,000×10,000像素图像，要求≤20秒
4. **严格指标**: 召回率≥85%，虚警率≤20%

### 技术方案

```
┌─────────────────────────────────────────────────────────┐
│                 三阶段训练策略                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 1: 标准训练 + 强数据增强 (150 epochs)              │
│    - YOLOv11-Large预训练模型                             │
│    - Mosaic, MixUp, Copy-Paste                          │
│    - 学习特征表示，不引入类别偏见                          │
│                                                         │
│  Stage 2: 类别平衡微调 (50 epochs)                       │
│    - 冻结backbone，仅训练检测头                          │
│    - 平衡采样策略                                        │
│    - Focal Loss + 类别权重                               │
│                                                         │
│  Stage 3: 尾类专项微调TFA (30 epochs)                    │
│    - 仅训练最后一个分类层                                │
│    - 仅使用尾类图像                                      │
│    - 5倍过采样                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
XH-202625/
├── configs/
│   └── competition_config.yaml     # 竞赛配置
├── src/
│   ├── data/
│   │   ├── analyze_dataset.py      # 数据集分析
│   │   ├── split_dataset.py        # 数据集划分
│   │   └── oversample_tail.py      # 尾类过采样
│   ├── models/
│   │   └── (使用ultralytics API)
│   ├── utils/
│   │   ├── class_balanced_sampler.py   # 类别平衡采样
│   │   └── augmentation.py              # 数据增强
│   ├── inference/
│   │   └── sahi_inference.py       # SAHI切片推理
│   └── train.py                    # 训练主脚本
├── scripts/
│   └── train_v1.sh                 # 训练脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 分析数据集分布
python src/data/analyze_dataset.py

# 划分训练集/验证集 (按源图像ID分组，防止泄露)
python src/data/split_dataset.py

# 创建尾类过采样数据集 (可选)
python src/data/oversample_tail.py
```

### 3. 训练

```bash
# 完整三阶段训练
python src/train.py --stage all

# 单独运行某个阶段
python src/train.py --stage 1
python src/train.py --stage 2 --weights runs/stage1/weights/best.pt
python src/train.py --stage 3 --weights runs/stage2/weights/best.pt
```

### 4. 推理

```bash
# SAHI切片推理 (适用于大尺幅图像)
python src/inference/sahi_inference.py \
    --model runs/stage3/weights/best.pt \
    --images /path/to/test/images \
    --output predictions.json \
    --conf 0.25 \
    --iou 0.45 \
    --slice-size 640 \
    --overlap 0.2
```

## 核心技术

### 1. 解耦训练 (Decoupling)

遵循CVPR 2020的原则：
- Stage 1: **Instance-Balanced Sampling** - 学习无偏见的特征
- Stage 2-3: **Class-Balanced Sampling** - 修复分类器

### 2. 类别平衡采样策略

```python
# 平方根逆频率采样 (比直接逆频率更稳定)
weight_i = 1 / sqrt(freq_i)

# 类别平衡重复采样
repeat_factor_i = median_freq / freq_i
```

### 3. 动态特征幻觉 (Feature Hallucination)

对极端少样本类(航母17框、两栖舰30框)：
- 提取现有样本的特征向量
- 学习特征的均值和方差分布
- 合成新特征向量用于训练

### 4. SAHI切片推理

处理10,000×10,000大图：
- 切片尺寸: 640×640
- 重叠率: 20%
- 后处理: 跨切片NMS
- 预估时间: ~15秒/图 (RTX3090)

## 类别分布

```
Head (>1000):  FA-18(2147), MS(1994), KC-135(1424), SU-35(1317), C-130(1297)
               F-15(1265), TU-95(1114), F-16(1017), C-17(998)

Medium (200-1000):  SU-34(933), P-3C(895), B-1B(762), SU-24(752), B-52(750)
                    QHS(641), TU-22(583), E-3(547), C-5(500), F-22(493)
                    E-8(432)

Tail (<200):  FSC(402), TU-160(361), KC-10(262), LQS(30), HM(17)
```

## 评估指标

竞赛要求：
- **召回率 ≥ 85%**: TP / (TP + FN) ≥ 0.85
- **虚警率 ≤ 20%**: FP / (TP + FP) ≤ 0.20
- **推理时间 ≤ 20秒**: 单张10,000×10,000图像

评估脚本:
```bash
python src/evaluate.py --predictions results.json --ground-truth val.json
```

## 参考文献

1. Kang et al., "Few-Shot Object Detection via Feature Reweighting", ICCV 2019
2. Wang et al., "Frustratingly Simple Few-Shot Object Detection", ICML 2020  
3. Li et al., "Overcoming Classifier Imbalance for Long-Tail Object Detection", CVPR 2020
4. Zhou et al., "BAGS: Balancing Classifier for Long-Tailed Object Detection", CVPR 2020
5. Akyon et al., "SAHI: Slicing Aided Hyper Inference", 2022

## 作者

竞赛团队: XH-202625项目组
更新日期: 2026-06-26
