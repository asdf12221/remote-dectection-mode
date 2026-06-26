#!/usr/bin/env python3
"""
Stage 2 训练脚本 - 尾类增强微调
使用Copy-Paste增强数据训练，冻结backbone
运行: python train_stage2.py --weights runs/stage1/weights/best.pt
"""
import os
import sys
import argparse
import torch
from ultralytics import YOLO

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0


def main():
    parser = argparse.ArgumentParser(description='Stage 2 尾类增强微调')
    parser.add_argument('--weights', type=str, required=True,
                        help='Stage 1最佳权重路径')
    parser.add_argument('--data', type=str, default='dataset_cp.yaml',
                        help='Copy-Paste增强数据集YAML')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--freeze', type=int, default=10,
                        help='冻结前N层 (默认10, 只训练head)')
    parser.add_argument('--output', type=str, default='runs/stage2')
    args = parser.parse_args()

    gpu_mem = get_gpu_memory()
    if args.imgsz is None:
        args.imgsz = 800 if gpu_mem >= 12 else 640
    if args.batch is None:
        if gpu_mem >= 24:
            args.batch = 16
        elif gpu_mem >= 12:
            args.batch = 8
        else:
            args.batch = 2

    print("=" * 70)
    print("Stage 2: 尾类增强微调")
    print(f"权重: {args.weights}")
    print(f"数据: {args.data}")
    print(f"冻结前 {args.freeze} 层")
    print(f"GPU显存: {gpu_mem:.1f} GB")
    print(f"imgsz={args.imgsz}, batch={args.batch}")
    print("=" * 70)

    # 检查数据集
    if not os.path.exists(args.data):
        print(f"错误: 数据集文件不存在: {args.data}")
        print("请创建 dataset_cp.yaml, 将train路径指向Copy-Paste增强后的数据:")
        print("""
names:
  0: HM
  1: LQS
  ... (同dataset.yaml)
nc: 25
path: /path/to/your/dataset
train: images/train_cp
val: images/val
""")
        sys.exit(1)

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer='SGD',
        lr0=0.005,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        freeze=args.freeze,
        # 增强配置 - 轻量级
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # 损失
        box=7.5,
        cls=2.0,           # 进一步提高分类损失
        dfl=1.5,
        # 控制
        patience=15,
        save=True,
        save_period=5,
        project=os.path.dirname(args.output),
        name=os.path.basename(args.output),
        device=0,
        workers=4,
        amp=True,
        exist_ok=True,
        cache=False,
        val=True,
        plots=True,
    )

    print("\n" + "=" * 70)
    print("Stage 2 训练完成!")
    print(f"最佳权重: {args.output}/weights/best.pt")
    print("=" * 70)


if __name__ == '__main__':
    main()
