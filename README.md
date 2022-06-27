
# Official Implementation of "SGTR: End-to-end Scene Graph Generation with Transformer"

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/Scarecrow0/SGTR/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.10.0-%237732a8)

Our paper [SGTR: End-to-end Scene Graph Generation with Transformer](https://arxiv.org/abs/2112.12970) has been accepted by CVPR 2022.

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Model Zoo 
SGTR performance:

The methods implemented in our toolkit and reported results are given in [Model Zoo.md](MODELZOO.md)

## Training **(IMPORTANT)**

Our codebase is developed based on cvpods, you can refer to [usage of cvpods](https://github.com/Megvii-BaseDetection/cvpods#usage) and [tutorial](
https://github.com/Megvii-BaseDetection/cvpods/blob/master/docs/tutorials/cvpods%20tutorials.ipynb).

### Prepare Faster-RCNN Detector
- You can download the pretrained DETR on ResNet-101 we used in the paper: 
  - [VG](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfJK_InTsk9Hq9RgXEui4gsBsk3pekuzPYk4gTR8coBYAA?e=fAo647), 
  - [OIv6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/Edrab9pd0O1NuVoz9RPmZtoBwxnSyl-NVIFCjxf6yUZ7FA?e=uPQZ1A), 

- Unzip the checkpoint into the folder

- Then, you need to modify the pre-trained weight parameter `MODEL.WEIGHTS` in config.py `playground/experiment/path/config_xx.py` to the path of corresponding pre-trained detector weight to make sure you load the detection weight parameter correctly.

- Besides, you can also train your own detector weight by the provide configs in  [Model Zoo.md](MODELZOO.md)

### Scene Graph Generation Model
- You can follow the following instructions to train your own, which takes 4 GPUs for train each SGG model. The results should be very close to the reported results given in paper.

You can run training by:
``` bash
# activate the environments
conda activate sgtr
cvpods_env

# move to experiment directory
cd playground/sgg/detr.res101.c5.one_stage_rel_tfmer
# link the config file
rm config.py; ln -s config_vg.py config.py 

pods_train --num-gpus 4
```

- We also provide the trained model .pth and config.json of [SGTR(vg)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/ESr2lrq30vdLg2nv-d32J6EBRhng0PHsmmdttrAPaSut3g?e=BDlKBg),[SGTR(oiv6)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/ETWYQ6nGWyRGgZAOfS7jOQwB4M19rTobmpCn0RBR0mZFCA?e=lGMhoK)

### More Training args:
Single Machine 4-GPU Run
``` bash
pods_train --num-gpus 4
```
If you want check the machine with available gpus:
``` bash
pods_train --num-gpus 4 --gpu-check
```


# Test

- First, you need download and unzip the provided trained model parameters.
- By replacing the parameter of `MODEL.TEST_WEIGHTS` in `config.py` to the trained model weight you can directly eval the model on validation or test set.
  - The program will load the config file (`config.json`) in same directory of model weights.
- You can use the following scripts for directly produce the results from the checkpoint provide by us.
``` bash
# visual genome dataset
pods_test --num-gpus 4 DATASETS.TEST "(vg_test,)"
pods_test --num-gpus 4 DATASETS.TEST "(vg_val,)"

# openimage dataset
pods_test --num-gpus 4 DATASETS.TEST "(oiv6_test,)"
pods_test --num-gpus 4 DATASETS.TEST "(oiv6_val,)"
```


## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@InProceedings{Li_2022_CVPR,
    author    = {Li, Rongjie and Zhang, Songyang and He, Xuming},
    title     = {SGTR: End-to-end Scene Graph Generation with Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19486-19496}
}
```

# Q&A

- Import ipdb in anywhere in your code will cause the multi-process initialization error, try pdb when you debug in multi-process mode.


## Acknowledgment
This repository borrows code from scene graph benchmarking framework developed by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [Rongjie Li](https://github.com/SHTUPLUS/PySGG)


