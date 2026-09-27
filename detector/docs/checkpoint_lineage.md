# Checkpoint lineage

## Supplied final checkpoint

The user-supplied artifact name `bifpn_trainval_fp16_finaltrain_crt.pth` maps
to the server artifact:

```text
trainval finaltrain best_coco_bbox_mAP_epoch_24.pth
  └── cRT best_coco_bbox_mAP_epoch_1.pth
        └── exported/renamed as bifpn_trainval_fp16_finaltrain_crt.pth
```

The trainval finaltrain stage starts from synthetic pretraining epoch 12 and
trains the full InternImage-L + BiFPN + Cascade R-CNN model. The cRT stage then
freezes the backbone, BiFPN, RPN and regression branches and retrains only the
three classifier layers with repeat-factor sampling.

The original trainval cRT run reported mAP 0.863, AP50 0.995 and AP75 0.987 on
its recorded validation loader. Since trainval contains the validation images,
those values are retained as lineage metadata only.

## Public benchmark lineage

The train-only → validation numbers in the root README come from a separate
train-only A0 chain:

```text
synth_pretrain_epoch12.pth
  → train-only finaltrain best_coco_bbox_mAP_epoch_35.pth
  → train-only cRT best_coco_bbox_mAP_epoch_1.pth
  → finaldatav5 val evaluation
```

This is the chain used for the reported validation-set mAP 0.758 / AP50 0.944 /
AP75 0.909. It is independent of the trainval leakage lineage, but it is not a
test-set estimate because validation AP is used for checkpoint selection.
