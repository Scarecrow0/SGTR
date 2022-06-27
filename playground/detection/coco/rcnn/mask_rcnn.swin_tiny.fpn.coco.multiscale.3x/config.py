import os.path as osp
import os
from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig
import cvpods

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])


_config_dict = dict(

    DEBUG=False,
    DUMP_INTERMEDITE=False,
    DUMP_RATE=dict(
        TRAIN=0.0001,
        TEST=0.1
    ),
    # EXPERIMENT_NAME=f"vg_rel_detr_v1-fix_ent_head-entities_awareloss-cross_decoder-split-fc-{loss_coeff}-cost_{match_cost}-rank_w_mult",
    EXPERIMENT_NAME=f"mask_rcnn.swin_tiny.fpn.coco.multiscale.3x",
    # EXPERIMENT_NAME=f"vg_rel_detr_v1-fix_ent_head-eos_{rel_eos_coef}",
    OVERIDE_CFG_DIR="",

    MODEL=dict(

        WEIGHTS="/group/rongjie/cvpods/playground/detection/coco/rcnn/mask_rcnn.swin_tiny.fpn.coco.multiscale.3x/log/swin_tiny_patch4_window7_224_cvpods_det.pth",
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
            STEPS=(210000, 250000),
            MAX_ITER=270000,
        ),
        OPTIMIZER=dict(
            WEIGHT_DECAY=0.05,
            BETAS=(0.9, 0.999),
            AMSGRAD=False,
            NAME="AdamW",
            BASE_LR=0.00005,
        ),
        IMS_PER_BATCH=8,
        IMS_PER_DEVICE=2,
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

    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
