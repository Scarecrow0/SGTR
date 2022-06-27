import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.vrd_config import VRDConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="oiv6-det_match_condi-rel_feature-update_eval",
    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
            # "relation_head.entities_head": "roi_heads.box_head",
            # "relation_head.rel_feat_head.rel_convfc_feat_head": "roi_heads.box_head",
        },

        WEIGHTS_FIXED=[
            "backbone",
            "proposal_generator",
            "roi_heads",

        ],
        WEIGHTS = "/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.600size.naive_baseline/log/2021-03-08_11-35-oiv6-fixed_freq_bias-loc_match_condi-rel_feature/model_0047999.pth",
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/oi/faster_rcnn.res101X.fpn.oiv6.600size.det_pretrain/log/2021-02-08_04-56-oiv6-from-coco-512-roihead_bz-top80det/model_0067999.pth",
        # WEIGHTS="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101.fpn.coco.600size.det_pretrain/log/model_final_f6e8b1.pkl",
        MASK_ON=False,
        PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_MEAN=[103.530, 116.280, 123.675],
        RESNETS=dict(
            DEPTH=101,
            NUM_GROUPS=32,
            WIDTH_PER_GROUP=8,
        ),
        ANCHOR_GENERATOR=dict(  # coco params
            SIZES=[[32], [64], [128], [256], [512]],
            ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
        ),
        RPN=dict(
            PRE_NMS_TOPK_TEST=6000,
            PRE_NMS_TOPK_TRAIN=12000,
            POST_NMS_TOPK_TEST=1500,
            POST_NMS_TOPK_TRAIN=3000
        ),
        ROI_HEADS=dict(
            NUM_CLASSES=601,
            IOU_THRESHOLDS=[0.5, ],
            BATCH_SIZE_PER_IMAGE=512,
            POSITIVE_FRACTION=0.25,
        ),
        ROI_BOX_HEAD=dict(
            FC_DIM=2048,
        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.1,
                INSTANCE_DROP_RATE=1.3,
                REPEAT_DICT_DIR=None,
            ),

            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,
            BATCH_SIZE_PER_IMAGE=768,  # max relationship proposals in training time
            MAX_PROPOSAL_PAIR=4096,  # max proposals number in inference time
            POSITIVE_FRACTION=0.4,
            SCORE_THRESH_TEST=0.02,
            FG_IOU_THRESHOLD=0.5,
            NUM_CLASSES=30,
            FREQUENCY_BIAS=True,
            IGNORE_VALUE=255,
            PAIR_MATCH_CONDITION="det",  # det loc
            ENTITIES_CONVFC_FEAT=dict(
                POOLER_RESOLUTION=7,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=3,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=2048,
                NORM=""
            ),

            UNION_FEAT=dict(
                POOLER_RESOLUTION=8,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=3,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=2048,
                NORM=""
            ),
            PAIRWISE_REL_FEATURE=dict(
                HIDDEN_DIM=512,
                OUTPUT_DIM=4096,
                WORD_EMBEDDING_FEATURES=True,
                WORD_EMBEDDING_FEATURES_DIM=300,
            ),

            LONGTAIL_PART_DICT=[None, 'h', 'h', 'h', 't', 'b', 't', 't', 'h', 'h', 'b', 't', 't', 't', 'b', 't', 't',
                                'h', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'h', 't'],

        ),
        PROPOSAL_GENERATOR=dict(
            FREEZE=False,
        ),
    ),
    DATASETS=dict(
        # TRAIN=("vgs_train",),
        # TEST=("vgs_val",),
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
        OPTIMIZER=dict(
            BASE_LR=0.008,  # over整个batchsize的LR
        ),
        LR_SCHEDULER=dict(
            STEPS=(10000, 20000),
            MAX_ITER=80000,
        ),
        IMS_PER_BATCH=16,  # 四卡时候的batchsize
        IMS_PER_DEVICE=4,
        CHECKPOINT_PERIOD=4000,
    ),
    TEST=dict(
        EVAL_PERIOD=2,
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


class TwoStageVRDFasterRCNNConfig(VRDConfig):
    def __init__(self):
        super(TwoStageVRDFasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = TwoStageVRDFasterRCNNConfig()
