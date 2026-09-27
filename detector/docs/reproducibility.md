# Reproducibility checklist

This document records the minimum information required to reproduce the
published validation-set report.

## Environment

- Python 3.10
- PyTorch 2.5.1 / torchvision 0.20.1
- CUDA 12.4 (or a compatible CUDA build)
- MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0
- Custom DCNv3 extension compiled from `detector/ops_dcnv3`

Install dependencies with `pip install -r detector/requirements.txt`, then
compile the extension before launching MMDetection.

## Data and checkpoints

Set the data and checkpoint environment variables described in the root README.
The public chain requires synthetic pretraining initialization, the train-only
A0 annotations, and the validation COCO annotations. Dataset files and model
weights are deliberately not redistributed in this repository.

## Training chain

Run the stages in order:

1. `detector/scripts/train_1_synth_pretrain.sh`
2. `detector/scripts/train_train_only_finaltrain.sh`
3. `detector/scripts/train_train_only_crt.sh`

The finaltrain stage uses FSC-only repeat-factor sampling with threshold 0.08.
The cRT stage uses threshold 0.15 and freezes all modules except the three
classifier layers (`fc_cls`).

## Evaluation chain

Run `detector/scripts/evaluate.py` on the validation COCO annotations. The
script reports COCO AP and the explicitly defined operating-point MR/FAR at
confidence thresholds 0.30, 0.50 and 0.70. The headline operating point uses
confidence 0.70 and same-class greedy matching at IoU 0.50.

## Reproducibility caveats

- The recorded benchmark uses one fixed split and one recorded training run.
- The configuration does not encode a project-specific random seed; exact
  bitwise replication is therefore not guaranteed.
- Validation AP is evaluated every epoch and used by
  `CheckpointHook(save_best='coco/bbox_mAP')`; the report is a validation-set
  model-selection result, not an unbiased held-out test estimate.
- For a stronger scientific claim, add a sealed test split or repeated runs
  with confidence intervals before comparing against other methods.
