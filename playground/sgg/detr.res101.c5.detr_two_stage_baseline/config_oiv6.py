import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.detr_config import DETRConfig
from cvpods.configs.vrd_config import VRDConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    EXPERIMENT_NAME="oiv6_bgnn",
    OVERIDE_CFG_DIR = "",
    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
            # "relation_head.entities_head": "roi_heads.box_head",
            # "obj_class_embed": "class_embed",
            # "obj_bbox_embed": "bbox_embed",
            # "sub_class_embed": "class_embed",
            # "sub_bbox_embed": "bbox_embed",
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
        TEST_WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-29_07-14-oiv6_rel_detr_v1_baseline-finetune_detr-12lyer_dec_160_query-top1-learnable_query-cross_rel_decoder/model_0028439.pth",
        # TEST_WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-22_09-42-oiv6_rel_detr_v1_baseline_transformer_more_param-n-query/model_0025283.pth",
        ######### weight for train ##############
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        WEIGHTS="/p300/outputs/cvpods/playground/sgg/oi/detr.res101.c5.multiscale.150e.bs16/2021-02-06_16-28-detr_oiv6_from_imagenet/model_1000444.pth",
        # WEIGHTS="/p300/outputs/cvpods/playground/sgg/oi/detr.res101.c5.multiscale.150e.bs16/2021-02-06_16-28-detr_oiv6_from_imagenet/model_1316374.pth",
        # WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-30_09-20-oiv6_rel_detr_v1-fix_detr-12lyer_dec_160_query-top1-learnable_query/model_0108335.pth",
        PIXEL_STD=[1.0, 1.0, 1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675],  # detectron2 pixel normalization config
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
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
            # NUM_CLASSES=150,  # for VG
            NUM_CLASSES=601,  # for OIv6
        ),

        ROI_HEADS=dict(
            IN_FEATURES=["res5", ],
            # IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            NUM_CLASSES=601,
        ),
        ROI_BOX_HEAD=dict(
            NUM_FC=2,
            FC_DIM=1024,
            NUM_CONV=0,
            CONV_DIM=256,

            POOLER_RESOLUTION=14,
            POOLER_SAMPLING_RATIO=0,
            POOLER_TYPE="ROIAlignV2",
        ),

        REL_HEAD_IN_FEAT_DIM=512,

        ROI_RELATION_HEAD=dict(
            # NUM_CLASSES=50, # for VG
            NUM_CLASSES=30,  # for OIV6
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.13,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
            ),

            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,
            BATCH_SIZE_PER_IMAGE=768,  # max relationship proposals in training time
            MAX_PROPOSAL_PAIR=2048,  # max proposals number in inference time
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
                NUM_FC=2,
                FC_DIM=1024,
                NORM="BN"
            ),

            UNION_FEAT=dict(
                POOLER_RESOLUTION=8,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=2,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=2048,
                NORM="BN"
            ),
            PAIRWISE_REL_FEATURE=dict(
                HIDDEN_DIM=512,
                OUTPUT_DIM=4096,
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
            BASE_LR=0.01,  # over整个batchsize的LR
        ),
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            STEPS=(10000, 20000),
            MAX_ITER=80000,
            MAX_EPOCH=None,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=8,  # 四卡时候的batchsize
        IMS_PER_DEVICE=2,
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
        DUMP_TEST=True,
        LOG_INTERVAL=100
    ),
    EXT_KNOWLEDGE=dict(
        GLOVE_DIR="/group/rongjie/cvpods/datasets/vg/glove",
    )
)


class OneStageRelDetrConfig(DETRConfig):
    def __init__(self):
        super(OneStageRelDetrConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageRelDetrConfig()
