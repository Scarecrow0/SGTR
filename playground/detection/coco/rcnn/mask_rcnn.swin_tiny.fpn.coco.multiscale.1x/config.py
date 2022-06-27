import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/data/swin_ckpts/swin_tiny_patch4_window7_224_cvpods_det.pth",
        MASK_ON=True,
        PIXEL_MEAN=[123.675, 116.28, 103.53],  # RGB
        PIXEL_STD=[58.395, 57.12, 57.375],
        BACKBONE=dict(
            FREEZE_AT=-1,
        ),
        SWINT=dict(
            EMBED_DIM=96,
            DEPTHS=[2, 2, 6, 2],
            NUM_HEADS=[3, 6, 12, 24],
            WINDOW_SIZE=7,
            MLP_RATIO=4.,
            PATCH_SIZE=4,
            DROP_PATH_RATE=0.2,
            APE=False,
            IN_CHANS=3,
            QKV_BIAS=True,
            QK_SCALE=None,
            PATCH_NORM=True,
            OUT_FEATURES=["res2", "res3", "res4", "res5"]
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            WEIGHT_DECAY=0.05,
            BETAS=(0.9, 0.999),
            AMSGRAD=False,
            NAME="AdamW",
            BASE_LR=0.0001,
        ),
        IMS_PER_BATCH=16,
    ),
    TRAINER=dict(
        FP16=dict(
            ENABLED=True,
        )
    ),
    INPUT=dict(
        FORMAT="RGB",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
    ),
    # OUTPUT_DIR=osp.join(
    #     '/data/Outputs/model_logs/cvpods_playground',
    #     osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
    OUTPUT_DIR="./log",
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
