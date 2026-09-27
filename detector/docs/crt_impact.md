# cRT impact report

This report compares the two checkpoints from the same train-only A0 chain:

| Stage | Checkpoint artifact | Evaluation record |
| --- | --- | --- |
| pre-cRT A0 | `best_coco_bbox_mAP_epoch_35.pth` | `abl_A0_ap.json`, `abl_A0_mr_far.json` |
| cRT C1 | `best_coco_bbox_mAP_epoch_1.pth` | `abl_C1_ap.json`, `abl_C1_mr_far.json` |

Both records use the same `finaldatav5` validation split (1,343 images, 7,022
ground-truth boxes), same-class greedy matching at IoU 0.50, and score
threshold 0.70 for the operating-point comparison.

## Operating-point counts and rates

| Stage | TP | FP | FN | MR | FAR | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-cRT A0 | 6,721 | 463 | 301 | 4.29% | 6.44% | 93.56% | 95.71% | 94.62% |
| cRT C1 | 6,675 | 358 | 347 | 4.94% | 5.09% | 94.91% | 95.06% | 94.98% |
| cRT − pre-cRT | −46 | −105 | +46 | +0.66 pp | **−1.35 pp** | +1.35 pp | −0.66 pp | +0.36 pp |

Relative to pre-cRT, cRT lowers FAR by 21.02% while MR increases by 15.28%.
At this threshold the practical effect is fewer false alarms, slightly more
misses, and a net F1 increase.

## COCO AP deltas

| Metric | pre-cRT A0 | cRT C1 | Change |
| --- | ---: | ---: | ---: |
| mAP | 0.75853 | 0.75791 | −0.00063 |
| AP50 | 0.94626 | 0.94435 | −0.00191 |
| AP75 | 0.90846 | 0.90939 | +0.00093 |
| APs | 0.27940 | 0.26822 | −0.01119 |
| APm | 0.74788 | 0.74732 | −0.00056 |
| APl | 0.78693 | 0.78837 | +0.00145 |

The rounded values in the README are derived from these records. The raw JSON
files remain on the experiment server and are identified above for auditability;
they are not copied into the source-only repository.
