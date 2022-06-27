# detr.res101.c5.coco.multiscale.150e.bs16  

seed: 19500164

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
```  
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  
|:------:|:------:|:------:|:-----:|:------:|:------:|  
| 29.204 | 50.152 | 29.140 | 9.602 | 31.578 | 46.908 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 41.721 | bicycle      | 21.472 | car            | 24.274 |  
| motorcycle    | 34.068 | airplane     | 50.395 | bus            | 49.276 |  
| train         | 60.700 | truck        | 22.498 | boat           | 14.839 |  
| traffic light | 8.485  | fire hydrant | 52.582 | stop sign      | 46.813 |  
| parking meter | 32.976 | bench        | 17.909 | bird           | 21.029 |  
| cat           | 65.172 | dog          | 55.984 | horse          | 48.271 |  
| sheep         | 40.870 | cow          | 38.468 | elephant       | 54.641 |  
| bear          | 65.693 | zebra        | 58.215 | giraffe        | 61.495 |  
| backpack      | 7.341  | umbrella     | 29.262 | handbag        | 6.645  |  
| tie           | 17.430 | suitcase     | 20.842 | frisbee        | 34.595 |  
| skis          | 12.472 | snowboard    | 18.457 | sports ball    | 14.223 |  
| kite          | 20.214 | baseball bat | 20.121 | baseball glove | 24.651 |  
| skateboard    | 34.704 | surfboard    | 24.619 | tennis racket  | 32.662 |  
| bottle        | 17.560 | wine glass   | 19.601 | cup            | 24.432 |  
| fork          | 18.820 | knife        | 7.525  | spoon          | 8.369  |  
| bowl          | 29.233 | banana       | 18.055 | apple          | 11.428 |  
| sandwich      | 27.387 | orange       | 21.261 | broccoli       | 16.294 |  
| carrot        | 9.153  | hot dog      | 22.806 | pizza          | 42.152 |  
| donut         | 28.607 | cake         | 25.807 | chair          | 16.386 |  
| couch         | 37.979 | potted plant | 18.614 | bed            | 40.462 |  
| dining table  | 27.215 | toilet       | 56.244 | tv             | 46.152 |  
| laptop        | 50.363 | mouse        | 35.805 | remote         | 16.658 |  
| keyboard      | 37.711 | cell phone   | 16.453 | microwave      | 42.929 |  
| oven          | 26.095 | toaster      | 2.281  | sink           | 22.564 |  
| refrigerator  | 48.962 | book         | 5.768  | clock          | 31.013 |  
| vase          | 21.980 | scissors     | 20.660 | teddy bear     | 39.134 |  
| hair drier    | 8.892  | toothbrush   | 11.422 |                |        |
