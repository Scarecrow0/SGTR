
# Model Zoo
We provide several models refereed in our paper.
Here we list the experiments directories in `playground/sgg` for each model.

## Performance
All models use ResNet-101 backbone.

### VG
| Model(SGGen) | mR@50 | mR@100 | R@50 | R@100 | head | body | tail |
|--------------|:-----:|:------:|:----:|:-----:|------|------|------|
| RelDN        |  6.0  |  7.3   | 31.4 |  35.9 | 34.1 | 6.6  | 1.1  |
| BGNN         |  8.4  |   9.8  | 29.8 |  33.7 | 29.8 | 11.2 | 2.1  |
| BGNN+DETR    |  5.2  |  6.5   | 22.8 |  26.8 | 26.5 | 5.5  | 1.2  |
| SGTR         |  12.6 |  16.4  | 24.8 |  27.9 | 27.6 | 20.1 | 9.4  |

<!-- ### OIv6
| Model(SGGen) | mR@50 | R@50 | wmAP_rel | wmAP_phr | score_wtd |
|---|:---:|:---:|:---:|:---:|---|
| RelDN | 33.98 | 73.08 | 32.16 | 33.39 | 40.84
| BGNN | 41.71 | 74.96 | 33.83 | 34.87 | 42.47 |
| SGTR | 41.71 | 74.96 | 33.83 | 34.87 | 42.47 | -->

# Experiment Directories
## SGG model
- End-to-End SGG (SGTR)
  - detr.res101.c5.one_stage_rel_tfmer


- Tranditional two-stage
  - Experiments Directory
    - detr.res101.c5.detr_two_stage_baseline
    - faster_rcnn.res101X.fpn.600size/two_stage_sgg
    - faster_rcnn.res101.c5.600size/two_stage_sgg
  - Models
    - BGNN
    - RelDN

- Adopt from End-to-End HOI model   
  - HOTR
    - hotr
  - AS-Net[TODO]

## Object Detection For SGG
  - Experiments Directory
    - detr.res101.c5.multiscale.150e.bs16/det_pretrain
    - faster_rcnn.res101.c5.600size/det_pretrain
    - faster_rcnn.res101X.fpn.coco.600size/det_pretrain
  - Models
    - BGNN
    - RelDN