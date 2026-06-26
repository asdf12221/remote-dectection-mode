#!/usr/bin/env python3
"""
Stage 1 训练脚本 - 服务器版
支持YOLO11l/m/s，自动适配GPU显存
运行: python train_stage1.py --model l --imgsz 800 --batch 16
"""
import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# 禁用MLflow避免错误
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'


def get_gpu_memory():
    """获取GPU显存(GB)"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0


def recommend_config(gpu_mem_gb):
    """根据GPU显存推荐配置"""
    if gpu_mem_gb >= 24:
        return {'model': 'l', 'imgsz': 800, 'batch': 16, 'mosaic': 1.0}
    elif gpu_mem_gb >= 16:
        return {'model': 'l', 'imgsz': 800, 'batch': 8, 'mosaic': 1.0}
    elif gpu_mem_gb >= 12:
        return {'model': 'm', 'imgsz': 640, 'batch': 8, 'mosaic': 1.0}
    elif gpu_mem_gb >= 8:
        return {'model': 's', 'imgsz': 640, 'batch': 4, 'mosaic': 0.0}
    else:
        return {'model': 'n', 'imgsz': 512, 'batch': 4, 'mosaic': 0.0}


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Training')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='数据集YAML路径 (默认: dataset.yaml)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='模型大小 (n/s/m/l/x), 不指定则自动选择')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='输入图像尺寸, 不指定则自动选择')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size, 不指定则自动选择')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--output', type=str, default='runs/stage1',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练权重路径')
    args = parser.parse_args()

    # 检查数据集
    if not os.path.exists(args.data):
        print(f"错误: 数据集文件不存在: {args.data}")
        print("请创建 dataset.yaml 文件, 内容如下:")
        print("""
names:
  0: HM
  1: LQS
  2: QHS
  3: MS
  4: A1_SU-35
  5: A2_C-130
  6: A3_C-17
  7: A4_C-5
  8: A5_F-16
  9: A6_TU-160
  10: A7_E-3
  11: A8_B-52
  12: A9_P-3C
  13: A10_B-1B
  14: A11_E-8
  15: A12_TU-22
  16: A13_F-15
  17: A14_KC-135
  18: A15_F-22
  19: A16_FA-18
  20: A17_TU-95
  21: A18_KC-10
  22: A19_SU-34
  23: A20_SU-24
  24: FSC
nc: 25
path: /path/to/your/dataset
train: images/train
val: images/val
""")
        sys.exit(1)

    # 检测GPU并推荐配置
    gpu_mem = get_gpu_memory()
    print(f"检测到GPU显存: {gpu_mem:.1f} GB")

    if args.model is None or args.imgsz is None or args.batch is None:
        recommended = recommend_config(gpu_mem)
        if args.model is None:
            args.model = recommended['model']
        if args.imgsz is None:
            args.imgsz = recommended['imgsz']
        if args.batch is None:
            args.batch = recommended['batch']
        print(f"自动推荐配置: model={args.model}, imgsz={args.imgsz}, batch={args.batch}")

    # 加载模型
    model_name = f'yolo11{args.model}.pt'
    print(f"\n加载模型: {model_name}")
    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(model_name)

    # 训练配置
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 1.0,           # 提高分类损失权重
        'dfl': 1.5,
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'flipud': 0.0,
        'mosaic': 1.0 if gpu_mem >= 12 else 0.0,
        'mixup': 0.15 if gpu_mem >= 12 else 0.0,
        'copy_paste': 0.3 if gpu_mem >= 16 else 0.0,
        'close_mosaic': 10,
        # 训练控制
        'patience': 30,
        'save': True,
        'save_period': 5,
        'project': os.path.dirname(args.output),
        'name': os.path.basename(args.output),
        'device': 0,
        'workers': 4,
        'amp': True,
        'exist_ok': True,
        'cache': False,
        'val': True,
        'plots': True,
    }

    print("\n" + "=" * 70)
    print("训练配置:")
    for k in ['data', 'epochs', 'imgsz', 'batch', 'optimizer', 'lr0',
              'mosaic', 'mixup', 'copy_paste']:
        print(f"  {k}: {config[k]}")
    print("=" * 70)

    # 开始训练
    results = model.train(**config)

    # 保存最终结果
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳权重: {args.output}/weights/best.pt")
    print(f"最后权重: {args.output}/weights/last.pt")
    print(f"结果CSV: {args.output}/results.csv")
    print("=" * 70)

    if hasattr(results, 'results_dict'):
        print("\n最终指标:")
        for k, v in results.results_dict.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
