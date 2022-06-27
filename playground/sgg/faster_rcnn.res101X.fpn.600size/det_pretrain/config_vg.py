import os
import os.path as osp

import cvpods
from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = osp.realpath(__file__)[:-9]

_config_dict = dict(
    DEBUG=False,
    EXPERIMENT_NAME="from-imagenet-512-roihead_bz-top80det-0.5bg_thresmatch",
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        WEIGHTS="detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl",
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.coco.600size.det_pretrain/log/coco_det_model_final_68b088_X101_8_32.pkl", # x101 coco detectron2 pretrain
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.coco.600size.det_pretrain/log/2021-01-14_07-14-from-coco-earlier-stoping-less-iter-larger-roihead_bz/model_0019999.pth",
        # WEIGHTS="/group/rongjie/output/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.coco.600size.det_pretrain/2021-01-17_03-05-from-coco-larger-roihead_bz-top80det-0.5bg_thresmatch/model_0035999.pth",
        MASK_ON=False,
        PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_MEAN=[103.530, 116.280, 123.675],
        RESNETS=dict(
            DEPTH=101,
            NUM_GROUPS=32,
            WIDTH_PER_GROUP=8,
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32], [64], [128], [256], [512]],
            ASPECT_RATIOS=[[0.23232838, 0.63365731, 1.28478321, 3.15089189]],
        ),
        RPN=dict(
            PRE_NMS_TOPK_TEST=6000,
            PRE_NMS_TOPK_TRAIN=12000,
            POST_NMS_TOPK_TEST=1500,
            POST_NMS_TOPK_TRAIN=3000
        ),
        ROI_HEADS=dict(
            NUM_CLASSES=150,
            IOU_THRESHOLDS=[0.5, ],
            # IOU_THRESHOLDS=[0.3, 0.5,],
            # IOU_LABELS=[0, -1, 1],
            BATCH_SIZE_PER_IMAGE=512,
            POSITIVE_FRACTION=0.4,
        ),
        ROI_BOX_HEAD=dict(
            FC_DIM=2048,
        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=False,
            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False
        ),
        PROPOSAL_GENERATOR=dict(
            FREEZE=False,
        ),
    ),
    DATASETS=dict(
        TRAIN=("vgs_train",),
        TEST=("vgs_val",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True

    ),

    DATALOADER=dict(
        NUM_WORKERS=3,
    ),

    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(600,), max_size=1000, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=600, max_size=1000, sample_style="choice")),
            ],
        ),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            BASE_LR=0.01,  # over整个batchsize的LR
        ),
        LR_SCHEDULER=dict(
            STEPS=(35000, 52000),
            MAX_ITER=70000,
        ),
        IMS_PER_BATCH=16,  # 四卡时候的batchsize
        IMS_PER_DEVICE=4,
        CHECKPOINT_PERIOD=2000,
    ),
    TEST=dict(
        EVAL_PERIOD=4000,
        DETECTIONS_PER_IMAGE=80,
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
    GLOBAL=dict(
        DUMP_TEST=True,
        LOG_INTERVAL=200
    ),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
