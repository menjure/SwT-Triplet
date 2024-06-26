# A Method for Recognition of Cattle Noseprint based Fusing Swin Transformer and Triplet Network
## Introduction

With the rapid global development of agriculture and animal husbandry, the recognition of cattle has become a crucial issue in ranch management. Previous studies have commonly relied on extracting features such as cattle back patterns and body structures to identify individual cattle. However, not all cattle possess distinct body pattern features, making them unsuitable for certain breeds. To address this issue, this paper proposes the use of cattle noseprint as a distinguishing feature for individual cattle, as noseprint offer uniqueness and universality. Furthermore, this architecture fusing Swin Transformer and triplet network is proposed for noseprint classification, achieving a remarkable accuracy of 98.61% on a newly curated dataset of 432 images. These results align with the expected outcomes and demonstrate the effectiveness of the proposed approach. This method holds significant implications for cattle traceability, insurance claims, and financial loans.

## Installation

Simply clone this repository to your desired local directory: `git clone https://github.com/menjure/SwT-Triplet.git` and install any missing requirements. Requires python=3.8, pytorch=1.11.0, torchvision=0.12.0, scikit-learn=1.1.3, seaborn=0.12.1, pillow=9.1.1, tqdm=4.64.1, opencv-python=4.6.0.66

## Data preparation

Download the cattle noseprint dataset from [here](https://pan.baidu.com/s/1K9o-KhLYBdJtu-H6Okbm2Q), password: `rj6y`. Unzip the file and put it in `datasets/Cattle_12/crop_nose_images/`.

## Train

Download the weights file from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) and put it in folder `weights/`. Then train with the following code:

```(python)
python train.py --model_path weights/swin_small_patch4_window7_224.pth --save_folder weights/
```

## Test

The weights generated by the training are saved in `weights/best_model_state.pth`. You can test with the following code:

```(python)
python test.py --model_path weights/best_model_state.pth
```

You can also download our best weights file from [here](https://pan.baidu.com/s/138cc9BxpF6e2E6qXFqImhg), password: `8435`. Then test with the following code:

```(python)
python test.py --model_path weights/best_acc_weights.pth
```

## Citation

If you use our code or dataset in your research, please cite with:

```(python)
@inproceedings{10.1145/3652628.3652716,
author = {Zhong, Minyue and Tan, Yao and Yu, Siyi},
title = {A Method for Recognition of Cattle Noseprint based Fusing Swin Transformer and Triplet Network},
year = {2024},
isbn = {9798400708831},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 4th International Conference on Artificial Intelligence and Computer Engineering},
pages = {523–527},
numpages = {5},
series = {ICAICE '23}
}
```
