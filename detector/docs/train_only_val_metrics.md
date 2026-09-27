# Train-only → validation metrics

This is the metric provenance used on the repository homepage.

## Split protocol

- **Training:** A0 train-only split (`annotations_train_abl.json`), 3,138
  source images, with the recorded FSC repeat-factor sampling.
- **Validation:** untouched `finaldatav5/annotations_val.json`, 1,343 images
  and 7,022 ground-truth boxes.
- **Architecture:** InternImage-L backbone, custom BiFPN neck, Cascade R-CNN,
  25 classes.
- **cRT:** only the three Cascade R-CNN `fc_cls` layers are trainable; the
  backbone, BiFPN, RPN and regression branches are frozen. Repeat-factor
  sampling uses threshold 0.15 for classifier re-training.
- **Matching:** COCO bbox AP plus same-class greedy matching at IoU 0.5 for
  miss-rate / false-alarm-rate calculations.

## Results

| Stage | Best epoch | mAP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train-only finaltrain | 35 | 0.759 | 0.946 | 0.908 | 0.279 | 0.748 | 0.787 |
| Train-only finaltrain + cRT | 1 | **0.758** | **0.944** | **0.909** | **0.268** | **0.747** | **0.788** |

The cRT model's deployment operating point at score threshold 0.70 was recorded
as MR 4.94% and FAR 5.09% overall.

Definitions for precision, recall, miss rate (MR), false-alarm rate (FAR), and
the IoU/confidence operating point are in [`metrics.md`](metrics.md).

## Checkpoint provenance

The cRT checkpoint used for these metrics was the best checkpoint from the
train-only A0 cRT work directory, originally named
`best_coco_bbox_mAP_epoch_1.pth`. The separate user-supplied
`bifpn_trainval_fp16_finaltrain_crt.pth` belongs to the trainval lineage and is
documented separately; it must not be used to claim an independent validation
score.
