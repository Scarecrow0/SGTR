import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.detr_config import DETRConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="vg_bgnn-rsmp",
    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
        },

        WEIGHTS_FIXED=[
            "backbone",
            "transformer",
            "class_embed",
            "bbox_embed",
            "query_embed",
            "input_proj",
        ],
        ########## weight for test #######
        TEST_WEIGHTS="",
        # TEST_WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-22_09-42-oiv6_rel_detr_v1_baseline_transformer_more_param-n-query/model_0025283.pth",
        ######### weight for train ##############
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        WEIGHTS="/public/home/lirj2/ckpts/vg_detr.pth",
        # WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-30_09-20-oiv6_rel_detr_v1-fix_detr-12lyer_dec_160_query-top1-learnable_query/model_0108335.pth",
        PIXEL_STD=[1.0, 1.0, 1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675],  # detectron2 pixel normalization config
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
        ),

        DETR=dict(  # entities DETR
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
            NUM_CLASSES=150,  # for VG
            # NUM_CLASSES=601,  # for OIv6
        ),

        ROI_HEADS=dict(
            IN_FEATURES=["res5", ],
            # IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            NUM_CLASSES=150,
        ),

        ROI_BOX_HEAD= dict(
            POOLER_SAMPLING_RATIO=0,
            POOLER_TYPE='ROIAlignV2'
        ),

        REL_HEAD_IN_FEAT_DIM=512,

        ROI_RELATION_HEAD=dict(
            NUM_CLASSES=50,  # for VG
            # NUM_CLASSES=30, # for OIV6
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.15,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
                ENTITY={
                    "ENABLED": False,
                    "REPEAT_FACTOR": 0.01,
                    "INSTANCE_DROP_RATE": 1.0,
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
                NAME='bgnn',
            ),
            BGNN_MODULE=dict(
                RELATION_CONFIDENCE_AWARE=True,
                MP_VALID_PAIRS_NUM=128,
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
                VISUAL_FEATURES_ON=False,
                IGNORE_FOREGROUND_BOXES_PAIRS=True,
                PRE_CLSER_LOSS='focal', # fgbg_norm
                FOCAL_LOSS_GAMMA=2,
                FOCAL_LOSS_ALPHA=0.5
            ),

            # LONGTAIL_PART_DICT=[None, 'h', 'h', 'h', 't', 'b', 't', 't', 'h', 'h', 'b', 't', 't', 't', 'b', 't', 't',
            #                     'h', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'h', 't'],

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
        # TRAIN=("oi_v6_train",),
        # TEST=("oi_v6_val",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True,
        ENTITY_LONGTAIL_DICT=['t', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't',
                              't', 'b', 'b', 'h', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't',
                              't', 'b', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                              't', 't', 'b', 'b', 'b', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't',
                              'b', 'b', 'b', 'b', 't', 'h', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 'b',
                              'h', 't', 'b', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't',
                              't', 'b', 'h', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 'b',
                              'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'h', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                              'h', 't', 't', 't', 'h', 't'],
        ENTITY_SORTED_CLS_LIST=[77, 135, 144, 110, 90, 21, 148, 114, 73, 60, 98, 56, 57, 125, 25, 75, 89, 86, 111, 39,
                                37, 44, 72, 27, 53, 65, 96, 123, 2, 143, 120, 126, 43, 59, 113, 112, 103, 3, 47, 58, 19,
                                133, 104, 61, 138, 134, 83, 52, 20, 99, 16, 129, 66, 74, 95, 128, 142, 42, 48, 9, 137,
                                63, 92, 22, 109, 18, 10, 40, 51, 76, 82, 13, 29, 17, 36, 80, 64, 136, 94, 146, 107, 79,
                                32, 87, 54, 149, 147, 30, 12, 14, 24, 4, 62, 97, 33, 116, 31, 70, 117, 124, 81, 23, 11,
                                26, 6, 108, 93, 145, 68, 121, 7, 84, 8, 46, 71, 28, 34, 15, 141, 102, 45, 131, 115, 41,
                                127, 132, 101, 88, 91, 122, 139, 5, 49, 100, 1, 85, 35, 119, 106, 38, 118, 105, 69, 130,
                                50, 78, 55, 140, 67]

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
            BASE_LR=0.001,  # over整个batchsize的LR
        ),
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            STEPS=(10000, 20000),
            MAX_ITER=80000,
            MAX_EPOCH=None,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=12,  # 四卡时候的batchsize
        IMS_PER_DEVICE=3,
        CHECKPOINT_PERIOD=4000,
    ),
    TEST=dict(
        EVAL_PERIOD=1000,
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
        DUMP_TEST=False,
        LOG_INTERVAL=100
    ),
    EXT_KNOWLEDGE=dict(
        GLOVE_DIR="datasets/vg/vg_motif_anno/glove",
    )
)


class OneStageRelDetrConfig(DETRConfig):
    def __init__(self):
        super(OneStageRelDetrConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageRelDetrConfig()
