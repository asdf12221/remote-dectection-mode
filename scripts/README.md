# Training and inference entry points

| Script | Purpose |
| --- | --- |
| `train_synthetic.sh` | Synthetic-data pretraining initialization |
| `train_finetune.sh` | Train-only A0 fine-tuning of the full detector |
| `train_crt.sh` | cRT classifier re-training on the A0 checkpoint |
| `evaluate.py` | COCO AP and MR/FAR evaluation |
| `infer_single.py` | Single-image inference |
| `infer_large.py` | Tiled large-scene inference |

The three public benchmark stages are `train_synthetic.sh`,
`train_finetune.sh`, and `train_crt.sh`.

The matching config map is documented in [`../configs/README.md`](../configs/README.md).
