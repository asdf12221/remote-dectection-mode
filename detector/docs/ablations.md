# Selected ablations

These runs come from the same train-only A0 validation protocol and use the
same `finaldatav5` validation split. The operating-point columns use score
threshold 0.70 and same-class greedy matching at IoU 0.50.

| Run | Change from A0 | mAP | AP50 | AP75 | MR | FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **A0** | full recipe | **0.759** | 0.946 | **0.908** | 4.29% | 6.44% |
| A1 | remove FSC repeat-factor sampling | 0.756 | 0.947 | 0.905 | 4.19% | 7.20% |
| A2 | freeze InternImage stages 1–2 | 0.750 | 0.944 | 0.904 | 4.36% | 7.07% |
| **A3** | standard AMP backbone, no fp16 backbone constructor | 0.757 | **0.948** | 0.906 | **4.14%** | **6.12%** |
| C1 | A0 + cRT classifier re-training | 0.758 | 0.944 | **0.909** | 4.94% | **5.09%** |

## What the runs suggest

- A0 remains the strongest overall mAP configuration in this set.
- Removing FSC repeat-factor sampling (A1) keeps mAP close but increases FAR,
  which supports retaining the tail-class sampling strategy.
- Freezing early backbone stages (A2) costs about 0.009 mAP relative to A0.
- A3 is a strong near-baseline alternative: it has the best AP50 and the lowest
  MR/FAR among the non-cRT A0-family runs, but slightly lower mAP/AP75.
- cRT (C1) gives the lowest FAR and the best AP75/F1 trade-off, while accepting
  a small increase in MR at the fixed confidence threshold.

## Provenance

The records used for this table are the server-side experiment files
`abl_A0_ap.json`, `abl_A1_ap.json`, `abl_A2_ap.json`, `abl_A3_ap.json`,
`abl_C1_ap.json` and their corresponding `*_mr_far.json` files. The raw files
are not copied into the source-only repository, but the names are retained for
auditability.
