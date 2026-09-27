# Remote Sensing Detection: InternImage + BiFPN + Cascade R-CNN + cRT

This repository contains the detector side of the FPBA-Syn project. It packages
the custom InternImage backbone, BiFPN neck, Cascade R-CNN configuration, cRT
(classifier re-training) implementation, DCNv3 build sources, training scripts,
evaluation code, and large-image inference utilities.

## Model pipeline

```text
synthetic pretraining
        ↓
train-only fine-tuning (InternImage-L + BiFPN + Cascade R-CNN)
        ↓
cRT classifier re-training (only fc_cls is unfrozen)
        ↓
validation evaluation / large-image inference
```

The detector has 25 classes: four ship classes, twenty aircraft classes, and
FSC (launch vehicle).

This repository also keeps the earlier YOLO/SAHI competition baseline under
`src/`, `scripts/`, `configs/` and `runs/`. The `detector/` subtree is the
separate MMDetection implementation of the final InternImage + BiFPN + Cascade
R-CNN + cRT experiment.

## Reported train → val benchmark

The public benchmark uses the **train-only A0 split** and evaluates on the
untouched `finaldatav5` validation split. No trainval images are used for these
numbers.

| Model | Training data | Evaluation data | mAP | AP50 | AP75 |
| --- | --- | --- | ---: | ---: | ---: |
| Cascade R-CNN + InternImage-L + BiFPN | train-only, 36 epochs | finaldatav5 val | 0.759 | 0.946 | 0.908 |
| + cRT | train-only, 10 classifier epochs | finaldatav5 val | **0.758** | **0.944** | **0.909** |

For the cRT model, the same validation run reported APs 0.268, APm 0.747 and
APl 0.788. At score threshold 0.70, the recorded overall miss rate / false
alarm rate were 4.94% / 5.09%.

These are detector metrics, not image-generation metrics. The evaluation split,
IoU definition, class list and post-processing are documented in
[`detector/docs/train_only_val_metrics.md`](detector/docs/train_only_val_metrics.md).

## Final trainval checkpoint lineage

The supplied final checkpoint name, `bifpn_trainval_fp16_finaltrain_crt.pth`,
belongs to the **reference-only trainval → cRT lineage**. Its original artifact
was saved as `best_coco_bbox_mAP_epoch_1.pth` in the trainval cRT work directory
and reached mAP 0.863 on the recorded validation loader. Because trainval
includes the validation images, that score is not used as the public benchmark
above. The matching configs and scripts are retained under:

```text
detector/configs/*trainval_fp16_finaltrain*
detector/scripts/train_trainval_fp16_finaltrain*.sh
```

The complete checkpoint graph is in
[`detector/docs/checkpoint_lineage.md`](detector/docs/checkpoint_lineage.md).

## Installation

The detector was developed with Python 3.10, PyTorch 2.5, CUDA 12.4,
MMCV 2.1, MMEngine 0.10.7 and MMDetection 3.3.0. Install versions compatible
with your CUDA build, then build the custom DCNv3 operator:

```bash
pip install torch torchvision mmcv==2.1.0 mmengine==0.10.7 mmdet==3.3.0 timm pycocotools
cd detector/ops_dcnv3
python setup.py build_ext --inplace
cd ../..
```

Set the project and data paths before using the configs:

```bash
export FPBA_DATA_ROOT=/data/finaldatav5
export FPBA_TRAIN_ONLY_ROOT=/data/train_only
export FPBA_TRAIN_ONLY_ANN=annotations_train_abl.json
export FPBA_INTERNIMAGE_CKPT=/models/internimage_l_22k_192to384.pth
export FPBA_SYNTH_CKPT=/models/synth_pretrain_epoch12.pth
export FPBA_WORK_ROOT=$PWD/detector/work_dirs
```

Model weights and datasets are intentionally not included. See
[`detector/weights/README.md`](detector/weights/README.md).

## Training

From the repository root:

```bash
bash detector/scripts/train_1_synth_pretrain.sh
bash detector/scripts/train_train_only_finaltrain.sh
bash detector/scripts/train_train_only_crt.sh
```

Override checkpoints when needed:

```bash
S1_CKPT=/models/synth_pretrain_epoch12.pth \
  bash detector/scripts/train_train_only_finaltrain.sh

FT_CKPT=/models/a0_best_epoch35.pth \
  bash detector/scripts/train_train_only_crt.sh
```

## Evaluation and inference

```bash
python detector/scripts/evaluate.py \
  --config detector/configs/cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_crt_config.py \
  --ckpt /models/a0_crt_best.pth \
  --val-json /data/finaldatav5/annotations_val.json \
  --image-root /data/finaldatav5/images/all \
  --output detector/eval_results.json
```

Single-image inference:

```bash
python detector/scripts/infer_single.py \
  --image image.png \
  --config detector/configs/cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_crt_config.py \
  --ckpt /models/a0_crt_best.pth
```

For large remote-sensing scenes, `infer_large.py` uses 1152×1152 tiles with
152-pixel overlap, resizes tiles to 800×800, maps boxes back to global
coordinates, and applies same-class global NMS.

## Repository layout

```text
detector/configs/       MMDetection config chain
detector/scripts/       train, evaluate and inference entry points
detector/mmdet_custom/  InternImage, BiFPN, cRT and sampling components
detector/ops_dcnv3/     source for the custom CUDA operator
detector/docs/          metric provenance and reproducibility notes
```

## License and attribution

The repository contains adapted InternImage and DCNv3 components. Preserve the
upstream licenses and attribution notices when redistributing. Third-party
datasets and pretrained weights remain subject to their own licenses.
