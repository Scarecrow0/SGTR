import os
import os.path as osp

import cvpods
from cvpods.configs.rcnn_config import RCNNConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = osp.realpath(__file__)[:-9]

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        RPN=dict(
            PRE_NMS_TOPK_TRAIN=2000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=1000,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            # NAME="StandardROIHeads",
            # IN_FEATURES=["p2", "p3", "p4", "p5"],
            NUM_CLASSES=20,
        ),
    ),
    DATASETS=dict(
        TRAIN=('voc_2007_trainval', 'voc_2012_trainval',),
        TEST=("voc_2007_test",)),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(12000, 16000),
            MAX_ITER=18000,  # 17.4 epochs
            WARMUP_ITERS=100,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(
                     480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                     max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=curr_folder.replace(cvpods_home, os.getenv("CVPODS_OUTPUT")),
)


class FasterRCNNConfig(RCNNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
