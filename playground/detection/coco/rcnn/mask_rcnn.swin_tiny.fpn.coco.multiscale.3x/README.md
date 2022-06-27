# mask_rcnn.swin_tiny.fpn.coco.multiscale.3x  

seed: 51672125

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.675
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.723
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 45.679 | 67.518 | 50.140 | 29.831 | 48.449 | 59.080 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 56.810 | bicycle      | 35.078 | car            | 47.112 |  
| motorcycle    | 46.501 | airplane     | 67.821 | bus            | 68.454 |  
| train         | 66.457 | truck        | 39.839 | boat           | 31.193 |  
| traffic light | 31.044 | fire hydrant | 70.022 | stop sign      | 68.580 |  
| parking meter | 50.112 | bench        | 28.363 | bird           | 41.074 |  
| cat           | 72.305 | dog          | 65.939 | horse          | 61.197 |  
| sheep         | 54.488 | cow          | 59.642 | elephant       | 64.415 |  
| bear          | 72.445 | zebra        | 67.204 | giraffe        | 68.266 |  
| backpack      | 19.975 | umbrella     | 43.321 | handbag        | 19.685 |  
| tie           | 37.725 | suitcase     | 46.348 | frisbee        | 69.604 |  
| skis          | 27.550 | snowboard    | 38.284 | sports ball    | 49.999 |  
| kite          | 45.141 | baseball bat | 36.864 | baseball glove | 41.952 |  
| skateboard    | 58.550 | surfboard    | 44.303 | tennis racket  | 51.706 |  
| bottle        | 43.025 | wine glass   | 39.533 | cup            | 46.875 |  
| fork          | 39.551 | knife        | 26.738 | spoon          | 22.459 |  
| bowl          | 45.589 | banana       | 26.848 | apple          | 24.678 |  
| sandwich      | 40.845 | orange       | 36.297 | broccoli       | 24.480 |  
| carrot        | 24.054 | hot dog      | 39.791 | pizza          | 55.121 |  
| donut         | 50.562 | cake         | 41.709 | chair          | 32.595 |  
| couch         | 45.800 | potted plant | 32.149 | bed            | 47.418 |  
| dining table  | 30.443 | toilet       | 63.597 | tv             | 60.513 |  
| laptop        | 64.073 | mouse        | 64.694 | remote         | 39.149 |  
| keyboard      | 55.281 | cell phone   | 40.504 | microwave      | 63.862 |  
| oven          | 38.783 | toaster      | 44.391 | sink           | 41.490 |  
| refrigerator  | 58.814 | book         | 18.500 | clock          | 51.269 |  
| vase          | 39.843 | scissors     | 34.299 | teddy bear     | 50.280 |  
| hair drier    | 9.629  | toothbrush   | 33.390 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 41.795 | 64.858 | 45.125 | 22.844 | 44.234 | 59.375 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 50.140 | bicycle      | 22.077 | car            | 44.542 |  
| motorcycle    | 37.478 | airplane     | 55.434 | bus            | 67.900 |  
| train         | 67.334 | truck        | 39.220 | boat           | 27.754 |  
| traffic light | 30.841 | fire hydrant | 67.120 | stop sign      | 67.389 |  
| parking meter | 50.691 | bench        | 21.056 | bird           | 35.459 |  
| cat           | 71.722 | dog          | 62.471 | horse          | 45.837 |  
| sheep         | 48.979 | cow          | 51.590 | elephant       | 59.547 |  
| bear          | 69.907 | zebra        | 59.052 | giraffe        | 53.289 |  
| backpack      | 20.156 | umbrella     | 50.674 | handbag        | 20.210 |  
| tie           | 35.458 | suitcase     | 47.470 | frisbee        | 67.021 |  
| skis          | 5.595  | snowboard    | 28.751 | sports ball    | 50.353 |  
| kite          | 32.427 | baseball bat | 30.267 | baseball glove | 45.188 |  
| skateboard    | 37.207 | surfboard    | 38.197 | tennis racket  | 58.802 |  
| bottle        | 41.954 | wine glass   | 35.290 | cup            | 47.325 |  
| fork          | 21.024 | knife        | 18.330 | spoon          | 15.583 |  
| bowl          | 42.457 | banana       | 22.334 | apple          | 23.881 |  
| sandwich      | 43.920 | orange       | 36.651 | broccoli       | 23.341 |  
| carrot        | 21.461 | hot dog      | 32.312 | pizza          | 53.441 |  
| donut         | 52.149 | cake         | 42.712 | chair          | 23.500 |  
| couch         | 39.696 | potted plant | 27.076 | bed            | 34.982 |  
| dining table  | 18.402 | toilet       | 60.623 | tv             | 63.168 |  
| laptop        | 65.362 | mouse        | 64.771 | remote         | 37.391 |  
| keyboard      | 53.914 | cell phone   | 39.784 | microwave      | 64.936 |  
| oven          | 35.806 | toaster      | 49.807 | sink           | 39.484 |  
| refrigerator  | 60.522 | book         | 13.082 | clock          | 52.934 |  
| vase          | 39.913 | scissors     | 25.993 | teddy bear     | 48.145 |  
| hair drier    | 12.354 | toothbrush   | 23.220 |                |        |
