# Configuration map

The config filenames preserve the experiment lineage so a checkpoint can be
traced back to its training recipe.

| Pattern | Meaning |
| --- | --- |
| `*_v5_config.py` | Base finaldatav5 detector recipe |
| `*_synthinit_config.py` | Starts from synthetic pretraining |
| `*_finaltrain_abl_a0*.py` | Public train-only A0 benchmark |
| `*_trainval_*.py` | Reference-only trainval lineage |
| `*_crt_config.py` | cRT classifier re-training stage |

For the public benchmark, use:

```text
cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_config.py
cascade_merged_25cls_v3_ce_v5_bifpn_v5_fp16_finaltrain_abl_a0_crt_config.py
```

The long names are intentional: they encode the dataset split, backbone/neck
recipe, precision mode, and training stage. The shell entry points in
`detector/scripts/` use shorter, user-facing names.
