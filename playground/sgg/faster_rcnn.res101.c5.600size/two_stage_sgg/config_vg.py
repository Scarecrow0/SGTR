import os
import os.path as osp

import cvpods
from cvpods.configs import RCNNConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="reldn-ent_retrain-rsmp",
    LOAD_FROM_SHM=False,
    # EXPERIMENT_NAME="freq_bias_only_baseline-test",
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        # WEIGHTS="detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl",
        WEIGHTS_LOAD_MAPPING={
            # "relation_head.entities_head": "roi_heads.box_head",
            # "relation_head.rel_feat_head.rel_convfc_feat_head": "roi_heads.box_head",
        },

        # WEIGHTS_FIXED=[
        #     "backbone",
        #     "proposal_generator",
        #     "roi_heads",
        #     "relation_head.entities_head",
        #     "relation_head.rel_feat_head",
        #     "relation_head.relation_feature_neck",
        # ],

        # for entities head re-training
        WEIGHTS_FIXED=[
            "backbone",
            "proposal_generator",
            "roi_heads.pooler",
            "roi_heads.res5",
            "roi_heads.box_predictor.bbox_pred",
            "relation_head",
        ],
        TEST_WEIGHTS='/public/home/lirj2/projects/cvpods/playground/sgg/vg/faster_rcnn.res101.c5.600size/naive_baseline/log/2022-01-29_14-26-reldn-ent_retrain-rsmp/model_0000399.pth',
        # TEST_WEIGHTS="/public/home/lirj2/projects/cvpods/playground/sgg/vg/faster_rcnn.res101.c5.600size/naive_baseline/log/2022-01-27_22-58-reldn-rel_pres-rsmp/model_0007999.pth",
        # WEIGHTS="/public/home/lirj2/R101-C5-fasterrcnn-det-vg.pth",
        # for ent retrain
        WEIGHTS="/public/home/lirj2/projects/cvpods/playground/sgg/vg/faster_rcnn.res101.c5.600size/naive_baseline/log/2022-01-27_22-58-reldn-rel_pres-rsmp/model_0007999.pth",
        # for pre retrain
        # WEIGHTS="/public/home/lirj2/projects/cvpods/playground/sgg/vg/faster_rcnn.res101.c5.600size/naive_baseline/log/2022-01-27_17-56-reldn-/model_0007999.pth",
        MASK_ON=False,
        # PIXEL_STD=[57.375, 57.120, 58.395],
        # PIXEL_MEAN=[103.530, 116.280, 123.675],
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res4",],
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32, 64, 128, 256, 512]],
            ASPECT_RATIOS=[[0.23232838, 0.63365731, 1.28478321, 3.15089189]],
        ),
        RPN=dict(
            PRE_NMS_TOPK_TEST=6000,
            PRE_NMS_TOPK_TRAIN=12000,
            POST_NMS_TOPK_TEST=1500,
            POST_NMS_TOPK_TRAIN=3000
        ),
        ROI_HEADS=dict(
            IN_FEATURES=["res4"],
            NUM_CLASSES=150,
            IOU_THRESHOLDS=[0.5, ],
            BATCH_SIZE_PER_IMAGE=512,
            POSITIVE_FRACTION=0.25,
        ),
        ROI_BOX_HEAD=dict(
            FC_DIM=2048,
        ),
        ROI_RELATION_HEAD=dict(
            NUM_CLASSES=50,  # for VG
            # NUM_CLASSES=30, # for OIV6
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.1,
                INSTANCE_DROP_RATE=2.5,
                REPEAT_DICT_DIR=None,
                ENTITY={
                    "ENABLED": True,
                    "REPEAT_FACTOR": 0.03,
                    "INSTANCE_DROP_RATE": 1.2,
                    "REPEAT_DICT_DIR": None,
                },
            ),

            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,
            BATCH_SIZE_PER_IMAGE=768,  # max relationship proposals in training time
            MAX_PROPOSAL_PAIR=4096,  # max proposals number in inference time
            POSITIVE_FRACTION=0.4,
            SCORE_THRESH_TEST=0.02,
            FG_IOU_THRESHOLD=0.5,
            FREQUENCY_BIAS=True,
            IGNORE_VALUE=255,
            PAIR_MATCH_CONDITION="det",  # det loc

            ENTITIES_CONVFC_FEAT=dict(
                POOLER_RESOLUTION=7,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=2,
                CONV_DIM=256,
                NUM_FC=1,
                FC_DIM=1024,
                NORM="BN"
            ),

            UNION_FEAT=dict(
                POOLER_RESOLUTION=9,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=2,
                CONV_DIM=256,
                NUM_FC=1,
                FC_DIM=1024,
                NORM="BN"
            ),
            PAIRWISE_REL_FEATURE=dict(
                HIDDEN_DIM=512,
                OUTPUT_DIM=2048,
                WORD_EMBEDDING_FEATURES=True,
                WORD_EMBEDDING_FEATURES_DIM=300,
            ),
            FEATURE_NECK=dict(
                NAME='naive',
            ),
            BGNN_MODULE=dict(
                RELATION_CONFIDENCE_AWARE=True,
                MP_VALID_PAIRS_NUM=512,
                ITERATE_MP_PAIR_REFINE=3,
                APPLY_GT=False,
                GATING_WITH_RELNESS_LOGITS=False,
                SHARE_RELATED_MODEL_ACROSS_REFINE_ITER=False,
                SHARE_PARAMETERS_EACH_ITER=True,
                RELNESS_MP_WEIGHTING=True,
                RELNESS_MP_WEIGHTING_SCORE_RECALIBRATION_METHOD='minmax',
                MP_ON_VALID_PAIRS=True,
                SKIP_CONNECTION_ON_OUTPUT=False
            ),
            RELATION_PROPOSAL_MODEL=dict(
                REL_AWARE_PREDICTOR_TYPE="single",
                VISUAL_FEATURES_ON=True,
                IGNORE_FOREGROUND_BOXES_PAIRS=True,
                PRE_CLSER_LOSS='focal_fgbg_norm', # fgbg_norm
                FOCAL_LOSS_GAMMA=2,
                FOCAL_LOSS_ALPHA=0.2
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
        FILTER_DUPLICATE_RELS=True,
        ENTITY_LONGTAIL_DICT = ['t', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 'b', 'h', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 'b', 'b', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b', 'b', 't', 'h', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 'b', 'h', 't', 'b', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't', 't', 'b', 'h', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'h', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'h', 't', 't', 't', 'h', 't'],
        ENTITY_SORTED_CLS_LIST =  [77, 135, 144, 110, 90, 21, 148, 114, 73, 60, 98, 56, 57, 125, 25, 75, 89, 86, 111, 39, 37, 44, 72, 27, 53, 65, 96, 123, 2, 143, 120, 126, 43, 59, 113, 112, 103, 3, 47, 58, 19, 133, 104, 61, 138, 134, 83, 52, 20, 99, 16, 129, 66, 74, 95, 128, 142, 42, 48, 9, 137, 63, 92, 22, 109, 18, 10, 40, 51, 76, 82, 13, 29, 17, 36, 80, 64, 136, 94, 146, 107, 79, 32, 87, 54, 149, 147, 30, 12, 14, 24, 4, 62, 97, 33, 116, 31, 70, 117, 124, 81, 23, 11, 26, 6, 108, 93, 145, 68, 121, 7, 84, 8, 46, 71, 28, 34, 15, 141, 102, 45, 131, 115, 41, 127, 132, 101, 88, 91, 122, 139, 5, 49, 100, 1, 85, 35, 119, 106, 38, 118, 105, 69, 130, 50, 78, 55, 140, 67]

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
            BASE_LR=0.0008,  # over整个batchsize的LR
        ),
        LR_SCHEDULER=dict(
            STEPS=(10000, 20000),
            MAX_ITER=80000,
        ),
        IMS_PER_BATCH=20,  # 四卡时候的batchsize
        IMS_PER_DEVICE=5,
        CHECKPOINT_PERIOD=400,
    ),
    TEST=dict(
        EVAL_PERIOD=10,
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
        GLOVE_DIR="/public/home/lirj2/projects/cvpods/datasets/vg/vg_motif_anno/glove",
    )
)


class TwoStageVRDFasterRCNNConfig(RCNNConfig):
    def __init__(self):
        super(TwoStageVRDFasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = TwoStageVRDFasterRCNNConfig()
