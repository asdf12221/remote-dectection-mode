# References and implementation basis

## Detection methods

1. Wang, W. et al. *InternImage: Exploring Large-Scale Vision Foundation
   Models with Deformable Convolutions*. CVPR, 2023.
   [arXiv:2211.05778](https://arxiv.org/abs/2211.05778)
2. Tan, M., Pang, R., and Le, Q. V. *EfficientDet: Scalable and Efficient
   Object Detection*. CVPR, 2020.
   [arXiv:1911.09070](https://arxiv.org/abs/1911.09070)
3. Cai, Z. and Vasconcelos, N. *Cascade R-CNN: Delving into High Quality
   Object Detection*. CVPR, 2018.
   [arXiv:1712.00726](https://arxiv.org/abs/1712.00726)
4. Kang, B. et al. *Decoupling Representation and Classifier for Long-Tailed
   Recognition*. ICLR, 2020.
   [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)

## Software

- PyTorch 2.5.1 and torchvision 0.20.1
- MMCV 2.1.0 and MMEngine 0.10.7
- MMDetection 3.3.0
- timm, pycocotools, OpenCV and NumPy

The repository contains adapted InternImage/DCNv3 components. Their upstream
licenses and attribution notices must be preserved when redistributing a
derived implementation.
