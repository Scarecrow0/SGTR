# mask_rcnn.swin_tiny.fpn.coco.multiscale.1x  

seed: 23548468

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.658
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 43.136 | 65.758 | 47.343 | 27.539 | 46.314 | 56.469 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 54.395 | bicycle      | 33.094 | car            | 44.302 |  
| motorcycle    | 43.221 | airplane     | 65.895 | bus            | 66.168 |  
| train         | 62.273 | truck        | 36.468 | boat           | 28.392 |  
| traffic light | 30.535 | fire hydrant | 66.476 | stop sign      | 66.221 |  
| parking meter | 51.397 | bench        | 25.223 | bird           | 38.797 |  
| cat           | 68.564 | dog          | 62.934 | horse          | 57.132 |  
| sheep         | 51.970 | cow          | 55.673 | elephant       | 63.833 |  
| bear          | 72.599 | zebra        | 64.367 | giraffe        | 67.258 |  
| backpack      | 18.097 | umbrella     | 40.095 | handbag        | 15.737 |  
| tie           | 33.595 | suitcase     | 41.422 | frisbee        | 67.209 |  
| skis          | 24.640 | snowboard    | 39.250 | sports ball    | 49.284 |  
| kite          | 44.859 | baseball bat | 32.072 | baseball glove | 39.844 |  
| skateboard    | 54.207 | surfboard    | 41.535 | tennis racket  | 47.280 |  
| bottle        | 41.153 | wine glass   | 36.463 | cup            | 44.381 |  
| fork          | 35.661 | knife        | 19.415 | spoon          | 18.052 |  
| bowl          | 42.560 | banana       | 24.936 | apple          | 24.149 |  
| sandwich      | 40.218 | orange       | 35.716 | broccoli       | 22.324 |  
| carrot        | 24.542 | hot dog      | 37.552 | pizza          | 53.582 |  
| donut         | 48.230 | cake         | 40.770 | chair          | 29.266 |  
| couch         | 41.571 | potted plant | 28.273 | bed            | 46.247 |  
| dining table  | 26.946 | toilet       | 59.448 | tv             | 60.121 |  
| laptop        | 59.662 | mouse        | 61.818 | remote         | 33.940 |  
| keyboard      | 49.899 | cell phone   | 37.047 | microwave      | 59.909 |  
| oven          | 34.481 | toaster      | 45.217 | sink           | 40.291 |  
| refrigerator  | 52.046 | book         | 16.947 | clock          | 51.510 |  
| vase          | 37.222 | scissors     | 33.471 | teddy bear     | 46.148 |  
| hair drier    | 13.393 | toothbrush   | 29.966 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 39.931 | 62.799 | 42.954 | 21.083 | 42.402 | 57.285 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 48.130 | bicycle      | 19.572 | car            | 41.668 |  
| motorcycle    | 34.549 | airplane     | 53.905 | bus            | 66.632 |  
| train         | 64.075 | truck        | 36.908 | boat           | 25.386 |  
| traffic light | 30.018 | fire hydrant | 65.156 | stop sign      | 66.199 |  
| parking meter | 52.568 | bench        | 18.420 | bird           | 34.365 |  
| cat           | 70.234 | dog          | 60.034 | horse          | 42.722 |  
| sheep         | 47.431 | cow          | 48.826 | elephant       | 59.489 |  
| bear          | 71.653 | zebra        | 56.220 | giraffe        | 52.355 |  
| backpack      | 17.463 | umbrella     | 47.518 | handbag        | 16.929 |  
| tie           | 33.353 | suitcase     | 43.269 | frisbee        | 65.145 |  
| skis          | 4.183  | snowboard    | 24.349 | sports ball    | 49.892 |  
| kite          | 33.311 | baseball bat | 27.897 | baseball glove | 43.047 |  
| skateboard    | 33.222 | surfboard    | 35.141 | tennis racket  | 55.862 |  
| bottle        | 40.643 | wine glass   | 32.140 | cup            | 45.601 |  
| fork          | 17.969 | knife        | 13.562 | spoon          | 13.094 |  
| bowl          | 40.227 | banana       | 21.009 | apple          | 24.589 |  
| sandwich      | 43.906 | orange       | 36.691 | broccoli       | 22.446 |  
| carrot        | 22.316 | hot dog      | 29.431 | pizza          | 52.826 |  
| donut         | 49.680 | cake         | 42.309 | chair          | 20.698 |  
| couch         | 36.519 | potted plant | 25.361 | bed            | 35.715 |  
| dining table  | 16.059 | toilet       | 58.976 | tv             | 62.563 |  
| laptop        | 61.668 | mouse        | 61.897 | remote         | 32.092 |  
| keyboard      | 51.821 | cell phone   | 37.712 | microwave      | 60.752 |  
| oven          | 33.881 | toaster      | 51.245 | sink           | 37.652 |  
| refrigerator  | 56.512 | book         | 12.177 | clock          | 53.389 |  
| vase          | 38.058 | scissors     | 26.303 | teddy bear     | 46.632 |  
| hair drier    | 9.262  | toothbrush   | 21.992 |                |        |
