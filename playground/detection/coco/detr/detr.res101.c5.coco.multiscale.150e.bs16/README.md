# detr.res101.c5.coco.multiscale.150e.bs16  

seed: 59144370

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.212
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.622
```  
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  
|:------:|:------:|:------:|:-----:|:------:|:------:|  
| 20.241 | 38.350 | 19.188 | 5.671 | 21.250 | 33.304 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 31.177 | bicycle      | 14.391 | car            | 12.786 |  
| motorcycle    | 22.637 | airplane     | 42.511 | bus            | 39.122 |  
| train         | 47.411 | truck        | 13.772 | boat           | 8.375  |  
| traffic light | 6.122  | fire hydrant | 39.619 | stop sign      | 38.184 |  
| parking meter | 26.216 | bench        | 11.475 | bird           | 15.386 |  
| cat           | 49.656 | dog          | 43.734 | horse          | 33.683 |  
| sheep         | 26.690 | cow          | 27.471 | elephant       | 43.516 |  
| bear          | 52.872 | zebra        | 49.072 | giraffe        | 49.086 |  
| backpack      | 1.783  | umbrella     | 17.924 | handbag        | 2.336  |  
| tie           | 9.681  | suitcase     | 12.605 | frisbee        | 23.172 |  
| skis          | 7.002  | snowboard    | 12.427 | sports ball    | 6.464  |  
| kite          | 9.838  | baseball bat | 12.843 | baseball glove | 10.113 |  
| skateboard    | 23.770 | surfboard    | 14.460 | tennis racket  | 24.045 |  
| bottle        | 10.929 | wine glass   | 13.646 | cup            | 14.087 |  
| fork          | 11.604 | knife        | 2.090  | spoon          | 2.569  |  
| bowl          | 22.016 | banana       | 10.095 | apple          | 4.971  |  
| sandwich      | 19.975 | orange       | 17.500 | broccoli       | 9.664  |  
| carrot        | 4.542  | hot dog      | 9.194  | pizza          | 33.796 |  
| donut         | 17.161 | cake         | 15.262 | chair          | 10.066 |  
| couch         | 22.896 | potted plant | 10.588 | bed            | 29.154 |  
| dining table  | 20.986 | toilet       | 42.573 | tv             | 35.427 |  
| laptop        | 38.604 | mouse        | 26.721 | remote         | 6.587  |  
| keyboard      | 31.294 | cell phone   | 12.630 | microwave      | 26.802 |  
| oven          | 18.324 | toaster      | 3.600  | sink           | 12.325 |  
| refrigerator  | 35.295 | book         | 2.694  | clock          | 22.718 |  
| vase          | 13.177 | scissors     | 12.389 | teddy bear     | 25.408 |  
| hair drier    | 1.733  | toothbrush   | 4.756  |                |        |
