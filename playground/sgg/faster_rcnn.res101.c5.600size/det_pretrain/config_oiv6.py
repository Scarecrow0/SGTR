import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs import RCNNConfig
from cvpods.configs.vrd_config import VRDConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="oiv6-det",
    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
        },

        WEIGHTS_FIXED=[
        ],
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        # PIXEL_STD=[57.375, 57.120, 58.395],
        # PIXEL_MEAN=[103.530, 116.280, 123.675],

        WEIGHTS = "/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101.c5.600size/det_pretrain/log/model_coco_101c4.pkl",

        MASK_ON=False,

        RESNETS=dict(
            DEPTH=101,
        ),
        ANCHOR_GENERATOR=dict(  # coco params
            SIZES=[[32, 64, 128, 256, 512]], ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
        ),
        RPN=dict(
            PRE_NMS_TOPK_TEST=6000,
            PRE_NMS_TOPK_TRAIN=12000,
            POST_NMS_TOPK_TEST=1500,
            POST_NMS_TOPK_TRAIN=3000
        ),
        ROI_HEADS=dict(
            IN_FEATURES=["res4"],
            NUM_CLASSES=601,
            IOU_THRESHOLDS=[0.5, ],
            BATCH_SIZE_PER_IMAGE=512,
            POSITIVE_FRACTION=0.25,
        ),
        ROI_BOX_HEAD=dict(
            FC_DIM=2048,
        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=False
        )
    ),
    DATASETS=dict(

        TRAIN=("oi_v6_train",),
        TEST=("oi_v6_val",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True

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
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
        RELATION=dict(
            MULTIPLE_PREDS=False,
            IOU_THRESHOLD=0.5,
            EVAL_POST_PROC=True,

        )
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
    GLOBAL=dict(
        DUMP_TEST=True,
        LOG_INTERVAL=100
    ),
    EXT_KNOWLEDGE=dict(
        GLOVE_DIR="/group/rongjie/cvpods/datasets/vg/glove",
    )
)


class FasterRCNNConfig(RCNNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
