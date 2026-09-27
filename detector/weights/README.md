# Weights are not tracked

GitHub's regular file limit is 100 MB, while the InternImage detector
checkpoints are approximately 1.1–4.4 GB. They are therefore intentionally
excluded from this repository.

Expected artifacts:

- `synth_pretrain_epoch12.pth` — synthetic pretraining initialization.
- `a0_best_epoch35.pth` — train-only finaltrain checkpoint.
- `a0_crt_best_epoch1.pth` — checkpoint used for the published train→val cRT
  metrics.
- `bifpn_trainval_fp16_finaltrain_crt.pth` — user-supplied final trainval cRT
  checkpoint; reference lineage only, not the independent benchmark.

Place the files in a local `weights/` directory and set `FPBA_*_CKPT` variables
as described in the root README. If you want to distribute a checkpoint, use a
dedicated model registry or Git LFS with an appropriate storage quota.
