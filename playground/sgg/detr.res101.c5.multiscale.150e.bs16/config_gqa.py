import os
import os.path as osp

import cvpods
from cvpods.configs.base_detection_config import BaseDetectionConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

# use epoch rather than iteration in this model

_config_dict = dict(
    DEBUG=False,
    EXPERIMENT_NAME="detr_gqa",
    MODEL=dict(
        # PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_STD=[1.0,1.0,1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675], # detectron2 pixel normalization config
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),
        DETR=dict(
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                NUM_ENC_LAYERS=6,
                NUM_DEC_LAYERS=6,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=False,
                RETURN_INTERMEDIATE_DEC=True,
            ),
            IN_FEATURES="res5",
            TEMPERATURE=10000,
            POSITION_EMBEDDING="sine",  # choice: [sine, learned]
            NUM_QUERIES=100,
            NO_AUX_LOSS=False,
            COST_CLASS=1.0,
            COST_BBOX=5.0,
            COST_GIOU=2.0,
            CLASS_LOSS_COEFF=1.0,
            BBOX_LOSS_COEFF=5.0,
            GIOU_LOSS_COEFF=2.0,
            EOS_COEFF=0.1,  # Relative classification weight of the no-object class
            NUM_CLASSES=1703,  # for GQA

        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=False,
            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,

            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.1,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,

                ENTITY={
                    "ENABLED": True,
                    "REPEAT_FACTOR": 0.2,
                    "INSTANCE_DROP_RATE": 1.1,
                    "REPEAT_DICT_DIR": None,
                },

            ),
        ),
    ),
    DATASETS=dict(
        TRAIN=("gqa_train",),
        TEST=("gqa_test",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True,
        ENTITY_LONGTAIL_DICT=['t', 't', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 'b', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 'b', 't', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 'b', 't', 'b', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 'h', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 'b', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 'b', 'b', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 'h', 't', 'b', 't', 'b', 't', 't', 'h', 't', 't', 't', 't', 't', 'b', 't', 't', 'h', 't', 't', 't', 't', 't', 't', 't', 't', 'h', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 'b', 'h', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 'h', 't', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'h', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 'h', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 'b', 'b', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 'h', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 't', 'h', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 'h', 't', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 'b', 'h', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 't', 't', 't', 'b', 't', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 't', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 'b', 'b', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 'b', 't', 'b', 't', 'b', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 't', 't', 't', 'b', 't', 't', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 'b', 'b', 't', 'h', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 't', 't', 't', 'b', 't', 'b', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 'h', 't', 't', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b', 't', 't', 't', 't', 't', 't', 't', 'b', 'b', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'h', 't', 'b', 'b', 'b', 't', 't', 'b', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 't', 'b', 't', 'b', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'b', 't']
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_EPOCH=None,
            MAX_ITER=2e6,
            # MAX_ITER=1000,
            WARMUP_ITERS=400,
            STEPS=(3e5,),
        ),
        OPTIMIZER=dict(
            NAME="DETRAdamWBuilder",
            BASE_LR=1e-4,
            BASE_LR_RATIO_BACKBONE=0.1,
            WEIGHT_DECAY=1e-4,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True,
            CLIP_VALUE=0.1,
            CLIP_TYPE="norm",
            NORM_TYPE=2.0,
        ),
        IMS_PER_BATCH=12,
        IMS_PER_DEVICE=3,
        CHECKPOINT_PERIOD=10000,
    ),
    DATALOADER=dict(
        NUM_WORKERS=1,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(480, 496, 512, 536, 552, 576, 600,),
                    max_size=1000, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=600, max_size=1000, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=2500,
    ),
    GLOBAL=dict(
        DUMP_TEST=False,
        DUMP=False,
        LOG_INTERVAL=300
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
)


class DETRConfig(BaseDetectionConfig):
    def __init__(self):
        super(DETRConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DETRConfig()
