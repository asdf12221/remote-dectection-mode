# Remote Sensing Detection

### InternImage-L · BiFPN · Cascade R-CNN · cRT

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.5.1">
  <img src="https://img.shields.io/badge/MMDetection-3.3.0-0B72B9" alt="MMDetection 3.3.0">
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.4">
</p>

An end-to-end remote-sensing detection framework for small objects, large
scenes, and long-tailed categories. The repository packages the final
InternImage-L + BiFPN + Cascade R-CNN + cRT pipeline into a clean, source-only
project that is easy to reproduce and extend.

## At a glance

<p align="center"><b>Model architecture</b></p>
<p align="center">
  <img src="assets/architecture.png" alt="Model architecture" width="100%">
</p>

<p align="center"><b>Training pipeline</b></p>
<p align="center">
  <img src="assets/training_pipeline.png" alt="Training pipeline" width="92%">
</p>

- **Multi-scale features:** InternImage-L backbone + BiFPN neck.
- **Progressive localization:** three-stage Cascade R-CNN head.
- **Long-tail adaptation:** cRT retrains only the classifier layers.
- **Large-scene inference:** tiling, coordinate remapping, and global NMS.

## Pipeline

```text
1) synthetic pretraining
        ↓
2) train-only fine-tuning
        ↓
3) cRT classifier re-training (fc_cls only)
        ↓
4) validation / large-image inference
```

The detector has 25 classes: four ship classes, twenty aircraft classes, and
FSC (launch vehicle).

Synthetic pretraining uses the companion generation project
[FPBA-Syn](https://github.com/asdf12221/FPBA-Syn). This repository starts from
the exported synthetic images/annotations or the resulting pretraining
checkpoint; the generation code is intentionally kept in that separate
project.

This repository is intentionally source-only: generated training outputs,
legacy YOLO/SAHI experiments, and large checkpoints are excluded so the public
project focuses on the final detector pipeline.

## Results

The numbers below use the train-only A0 split and the `finaldatav5` validation
split. Validation images are not used for gradient updates.

| Model | mAP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| InternImage-L + BiFPN + Cascade R-CNN | 0.759 | 0.946 | 0.908 |
| **+ cRT** | **0.758** | **0.944** | **0.909** |

At confidence `0.70` and matching IoU `0.50`:

| Precision | Recall | F1 | Miss rate | False-alarm rate |
| ---: | ---: | ---: | ---: | ---: |
| **94.91%** | **95.06%** | **94.98%** | **4.94%** | **5.09%** |

### cRT impact

| Metric @ score 0.70 | pre-cRT | cRT | Change |
| --- | ---: | ---: | ---: |
| Miss rate (MR) | 4.29% | 4.94% | +0.66 pp |
| False-alarm rate (FAR) | 6.44% | **5.09%** | **−1.35 pp (−21.02%)** |

## Checkpoint notes

The supplied `bifpn_trainval_fp16_finaltrain_crt.pth` belongs to the
**reference-only trainval → cRT lineage**. Its recorded mAP was 0.863, but it is
not used as the headline result because trainval includes validation images.
Matching configs and scripts are kept under:

```text
detector/configs/*trainval_fp16_finaltrain*
detector/scripts/train_reference*.sh
```

Datasets and large `.pth` files are not committed to the repository.

## Setup

The detector was developed with Python 3.10, PyTorch 2.5, CUDA 12.4,
MMCV 2.1, MMEngine 0.10.7 and MMDetection 3.3.0. Install versions compatible
with your CUDA build, then build the custom DCNv3 operator:

```bash
pip install -r detector/requirements.txt
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
bash detector/scripts/train_synthetic.sh
bash detector/scripts/train_finetune.sh
bash detector/scripts/train_crt.sh
```

Override checkpoints when needed:

```bash
S1_CKPT=/models/synth_pretrain_epoch12.pth \
  bash detector/scripts/train_finetune.sh

FT_CKPT=/models/a0_best_epoch35.pth \
  bash detector/scripts/train_crt.sh
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
detector/scripts/       clean training, evaluation, and inference entry points
detector/mmdet_custom/  InternImage, BiFPN, cRT and sampling components
detector/ops_dcnv3/     source for the custom CUDA operator
detector/weights/       checkpoint download and placement instructions
assets/                 public architecture and training-strategy figures
```

## License and attribution

The repository contains adapted InternImage and DCNv3 components. Preserve the
upstream licenses and attribution notices when redistributing. Third-party
datasets and pretrained weights remain subject to their own licenses. No single
umbrella license is asserted for all repository contents; review the upstream
notices and dataset/checkpoint terms before redistribution.
