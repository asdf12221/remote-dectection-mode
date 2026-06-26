"""
正式训练 Stage 1: YOLOv11s + 过采样平衡数据集
100 epochs, patience=20, 适配 RTX 5060 8GB
运行: python scripts/train_stage1_final.py
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

DATA_YAML = r"C:\Users\lenovo\Desktop\数据集\data\dataset_balanced.yaml"
OUTPUT_DIR = r"C:\Users\lenovo\Desktop\XH-202625\runs"

def main():
    print("=" * 70)
    print("STAGE 1 正式训练: YOLOv11s, 100 epochs, imgsz=640, batch=2")
    print(f"数据集: {DATA_YAML}")
    print(f"输出目录: {OUTPUT_DIR}/stage1_final")
    print("=" * 70)

    model = YOLO('yolo11s.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,          # 增大分辨率，提升小目标检测
        batch=2,            # batch=4+640会OOM，用batch=2
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=1.0,            # 提高分类损失权重，改善类别区分
        dfl=1.5,
        # 数据增强 - 适配低显存
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,        # 关闭mosaic避免OOM
        mixup=0.0,         # 关闭mixup
        copy_paste=0.0,    # 关闭copy_paste (离线已做)
        close_mosaic=0,
        # 训练控制
        patience=30,       # 增加耐心，让模型充分训练
        save=True,
        save_period=5,
        project=OUTPUT_DIR,
        name='stage1_final',
        device=0,
        workers=0,
        amp=True,
        exist_ok=True,
        cache=False,
        nbs=64,
        erasing=0.0,
        auto_augment=None,
        val=True,
        plots=True,
    )

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳权重: {OUTPUT_DIR}/stage1_final/weights/best.pt")
    print(f"最后权重: {OUTPUT_DIR}/stage1_final/weights/last.pt")
    print("=" * 70)

    if hasattr(results, 'results_dict'):
        print("\n最终指标:")
        for k, v in results.results_dict.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
