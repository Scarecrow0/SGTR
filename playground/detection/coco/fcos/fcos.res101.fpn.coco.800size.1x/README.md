# fcos.res101.fpn.coco.800size.1x  

seed: 30064528

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.490
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.655
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 31.245 | 49.036 | 33.328 | 17.802 | 34.577 | 39.781 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 48.202 | bicycle      | 22.239 | car            | 36.257 |  
| motorcycle    | 29.137 | airplane     | 53.204 | bus            | 56.483 |  
| train         | 51.025 | truck        | 28.297 | boat           | 18.123 |  
| traffic light | 21.605 | fire hydrant | 56.408 | stop sign      | 54.285 |  
| parking meter | 36.056 | bench        | 12.910 | bird           | 28.928 |  
| cat           | 57.645 | dog          | 54.878 | horse          | 43.252 |  
| sheep         | 41.605 | cow          | 45.852 | elephant       | 54.749 |  
| bear          | 60.805 | zebra        | 61.395 | giraffe        | 60.315 |  
| backpack      | 11.014 | umbrella     | 28.954 | handbag        | 9.123  |  
| tie           | 20.784 | suitcase     | 23.851 | frisbee        | 55.720 |  
| skis          | 12.243 | snowboard    | 14.181 | sports ball    | 39.466 |  
| kite          | 35.922 | baseball bat | 16.546 | baseball glove | 30.248 |  
| skateboard    | 36.916 | surfboard    | 20.330 | tennis racket  | 36.504 |  
| bottle        | 30.627 | wine glass   | 27.132 | cup            | 34.353 |  
| fork          | 13.261 | knife        | 7.800  | spoon          | 6.135  |  
| bowl          | 33.506 | banana       | 18.334 | apple          | 16.347 |  
| sandwich      | 23.183 | orange       | 26.130 | broccoli       | 20.329 |  
| carrot        | 15.945 | hot dog      | 20.364 | pizza          | 39.350 |  
| donut         | 37.123 | cake         | 25.860 | chair          | 19.260 |  
| couch         | 30.669 | potted plant | 20.092 | bed            | 36.180 |  
| dining table  | 20.369 | toilet       | 47.115 | tv             | 45.387 |  
| laptop        | 45.358 | mouse        | 53.150 | remote         | 17.163 |  
| keyboard      | 33.146 | cell phone   | 27.176 | microwave      | 46.618 |  
| oven          | 22.072 | toaster      | 7.159  | sink           | 25.305 |  
| refrigerator  | 36.936 | book         | 10.682 | clock          | 45.771 |  
| vase          | 31.220 | scissors     | 11.793 | teddy bear     | 36.623 |  
| hair drier    | 0.662  | toothbrush   | 8.478  |                |        |
