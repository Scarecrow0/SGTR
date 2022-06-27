import os
import os.path as osp

import cvpods
from cvpods.configs import RCNNConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="R101-C5-vg_det",
    # EXPERIMENT_NAME="freq_bias_only_baseline-test",
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        # WEIGHTS="detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl",
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.600size.naive_baseline/log/2021-02-24_12-14-2stage_rel_naive_baseline/model_0055999.pth",
        WEIGHTS_LOAD_MAPPING={
            # "relation_head.entities_head": "roi_heads.box_head",
            # "relation_head.rel_feat_head.rel_convfc_feat_head": "roi_heads.box_head",
        },

        WEIGHTS_FIXED=[
            # "backbone",
            # "proposal_generator",
            # "roi_heads"

        ],
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.coco.600size.det_pretrain/log/faster_rcnn.res101X.fpn.vg.pth",
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101.fpn.coco.600size.det_pretrain/log/model_final_f6e8b1.pkl",
        MASK_ON=False,
        # PIXEL_STD=[57.375, 57.120, 58.395],
        # PIXEL_MEAN=[103.530, 116.280, 123.675],
        RESNETS=dict(
            DEPTH=101,
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32, 64,128,256,512]],
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
            BATCH_SIZE_PER_IMAGE=512,
            POSITIVE_FRACTION=0.25,
        ),
        ROI_BOX_HEAD=dict(
            FC_DIM=2048,
        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=False,
            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,
            BATCH_SIZE_PER_IMAGE=768,  # max relationship proposals in training time
            MAX_PROPOSAL_PAIR=4096,  # max proposals number in inference time
            POSITIVE_FRACTION=0.4,
            SCORE_THRESH_TEST=0.02,
            FG_IOU_THRESHOLD=0.5,
            NUM_CLASSES=50,
            FREQUENCY_BIAS=True,
            IGNORE_VALUE=255,
            PAIR_MATCH_CONDITION="det",  # det loc
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.11,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
            ),

            ENTITIES_CONVFC_FEAT=dict(
                POOLER_RESOLUTION=7,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=3,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=2048,
                NORM="BN"
            ),

            UNION_FEAT=dict(
                POOLER_RESOLUTION=8,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=3,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=4096,
                NORM="BN"
            ),
            PAIRWISE_REL_FEATURE=dict(
                HIDDEN_DIM=512,
                OUTPUT_DIM=4096,
                WORD_EMBEDDING_FEATURES=True,
                WORD_EMBEDDING_FEATURES_DIM=300,
            ),

            LONGTAIL_PART_DICT=[None, 'b', 't', 't', 't', 'b', 'b', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 't', 'b',
                                't', 't', 'b', 'h', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 'h', 'h', 'h', 't', 'b',
                                't', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 'b', 'b', 'h', 'b', 'b'],

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
            STEPS=(40000, 70000),
            MAX_ITER=80000,
        ),
        IMS_PER_BATCH=12,  # 四卡时候的batchsize
        IMS_PER_DEVICE=3,
        CHECKPOINT_PERIOD=4000,
    ),
    TEST=dict(
        EVAL_PERIOD=4000,
        RELATION=dict(
            MULTIPLE_PREDS=False,
            IOU_THRESHOLD=0.5

        )
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
    GLOBAL=dict(
        DUMP_TEST=False,
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
