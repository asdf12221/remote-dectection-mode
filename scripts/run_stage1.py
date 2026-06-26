"""
Stage 1 Training - Optimized for RTX 5060 Laptop (8GB VRAM)
YOLOv11-Small, imgsz=512, batch=4
"""
import sys
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = r"C:\Users\lenovo\Desktop\数据集\data\dataset_balanced.yaml"
OUTPUT_DIR = r"C:\Users\lenovo\Desktop\XH-202625\runs"

def main():
    import os
    # Disable MLflow to avoid filesystem backend issues on Windows
    os.environ['MLFLOW_SKIP_Callback'] = '1'
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    
    print("=" * 70)
    print("STAGE 1: YOLOv11-Small + balanced dataset")
    print("=" * 70)

    model = YOLO('yolo11s.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=512,
        batch=4,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Minimal augmentation for memory
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        # Training config
        patience=20,
        save=True,
        save_period=10,
        project=OUTPUT_DIR,
        name='stage1',
        device=0,
        workers=0,            # Windows: use 0 workers to avoid multiprocessing issues
        amp=True,
        exist_ok=True,
        cache=False,
        nbs=16,
        erasing=0.0,
        auto_augment=None,
    )

    print(f"\nStage 1 complete!")
    print(f"Best weights: {OUTPUT_DIR}/stage1/weights/best.pt")

if __name__ == '__main__':
    main()
