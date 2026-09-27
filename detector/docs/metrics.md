# Detection metrics

This project reports two complementary metric families:

1. COCO bounding-box AP (`mAP`, `AP50`, `AP75`, `APs`, `APm`, `APl`) for the
   standard detector benchmark.
2. An operating-point summary for deployment, computed with same-class greedy
   matching at IoU 0.50 and a confidence threshold of 0.70.

## Definitions

With true positives (TP), false positives (FP), and false negatives (FN):

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
miss rate = FN / (TP + FN) = 1 - recall
false-alarm rate = FP / (TP + FP) = 1 - precision
```

The reported cRT operating point is therefore:

| Quantity | Value | Protocol |
| --- | ---: | --- |
| Confidence threshold | 0.70 | detections below this score are discarded |
| Matching IoU | 0.50 | same class, greedy one-to-one matching |
| Precision | 95.06% | complement of recorded FAR |
| Recall | 95.06% | complement of recorded MR |
| Miss rate (MR) | **4.94%** | FN / (TP + FN) |
| False-alarm rate (FAR) | **5.09%** | FP / (TP + FP) |

These values are aggregate validation-set rates, not per-class averages. For
class-wise analysis, export the matched TP/FP/FN counts from
`detector/scripts/evaluate.py` and report the same formulas per category.

## Split and provenance

- Training: A0 train-only split (`annotations_train_abl.json`).
- Validation: untouched `finaldatav5/annotations_val.json` (1,343 images,
  7,022 ground-truth boxes).
- Architecture: InternImage-L + BiFPN + Cascade R-CNN, 25 classes.
- cRT: only the three classifier layers are retrained; backbone, neck, RPN and
  regression branches remain frozen.

Do not use the separate `bifpn_trainval_fp16_finaltrain_crt.pth` lineage to
claim an independent validation score: it was trained with trainval images and
is retained only as checkpoint lineage metadata.
