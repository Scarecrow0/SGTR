import os
from cvpods.configs.detr_config import DETRConfig


_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    SAMPLED_DUMP=True, # random drop the image during the dump for strorage
    EXPERIMENT_NAME="",
    OVERIDE_CFG_DIR="",
    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
            # "relation_head.entities_head": "roi_heads.box_head",
            # "obj_class_embed": "class_embed",
            # "obj_bbox_embed": "bbox_embed",
            # "sub_class_embed": "class_embed",
            # "sub_bbox_embed": "bbox_embed",
        },

        WEIGHTS_FIXED=[
            # "backbone",
            # "transformer",
        ],
        ########## weight for test #######
        TEST_WEIGHTS="",
        ######### weight for train ##############
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        PIXEL_STD=[1.0, 1.0, 1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675],  # detectron2 pixel normalization config
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),
        ANCHOR_GENERATOR=dict(  # coco params
            SIZES=[[32], [64], [128], [256], [512]],
            ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
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

        REL_DETR=dict(  # relationship DETR
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                NUM_ENC_LAYERS=None,  # share the encoder with the entities part
                NUM_DEC_LAYERS=12,
                DIM_FFN=4096,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,
            ),

            CROSS_DECODER=dict( # k_generator
                ENABLED=False,
                N_HEAD=8,
                NUM_DEC_LAYERS=6,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,
            ),
            ENTITIES_AWARE_HEAD=dict(
                ENABLED=False,
                ENTITIES_AWARE_MATCHING=False,  # gt pred matching in training
                ENTITIES_AWARE_RANKING=False,  # pred entities pred relationship matching and ranking in test.

                USE_ENTITIES_PRED=False,

                ENTITIES_INDEXING=False,
                USE_ENTITIES_INDEXING_RANKING=False,

                CROSS_DECODER=False,
                N_HEAD=8,
                NUM_DEC_LAYERS=6,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,

                ENT_CLS_LOSS_COEFF=1.0,
                ENT_BOX_L1_LOSS_COEFF=1.0,
                ENT_BOX_GIOU_LOSS_COEFF=1.0,
                ENT_INDEXING_LOSS_COEFF=1.0,


                COST_ENT_CLS=1.0,
                COST_BOX_L1=1.5,
                COST_BOX_GIOU=2.5,
                COST_INDEXING=1.0 ,


            ),
            PAIRWISE_DECODER=dict(
                ENABLED=False,
                QUERY_GENERATOR_DEC=dict(
                    N_HEAD=8,
                    NUM_DEC_LAYERS=6,  # share the encoder with the entities part
                    DIM_FFN=2048,
                    DROPOUT_RATE=0.1,
                    ACTIVATION="relu",
                    PRE_NORM=True,
                ),
                ENTITIES_DECODER=dict(
                    D_MODEL=256,
                    N_HEAD=8,
                    NUM_DEC_LAYERS=6,
                    DIM_FFN=2048,
                    DROPOUT_RATE=0.1,
                    ACTIVATION="relu",
                    PRE_NORM=True,
                    RETURN_INTERMEDIATE_DEC=True,
                )
            ),



            LOSSES=["relation_vec",  "cardinality"],  # "rel_labels"
            TEMPERATURE=10000,
            POSITION_EMBEDDING="sine",  # choice: [sine, learned]
            NUM_QUERIES=160,
            NUM_PRED_EDGES=1,
            NO_AUX_LOSS=True,

            COST_CLASS=1.0,
            COST_REL_VEC=5.0,

            CLASS_LOSS_COEFF=1.0,
            REL_VEC_LOSS_COEFF=1.0,
            EOS_COEFF=0.1,  # Relative classification weight of the no-object class

            FOCAL_LOSS=dict(
                ENABLED=False,
                DUAL_STAGE_PRED=False,
                ALPHA=1.0,
                GAMMA=0,
            ),


        ),

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
            MAX_PROPOSAL_PAIR=4096,  # max proposals number in inference time
            POSITIVE_FRACTION=0.4,
            SCORE_THRESH_TEST=0.02,
            FG_IOU_THRESHOLD=0.5,
            FREQUENCY_BIAS=False,
            IGNORE_VALUE=255,
            PAIR_MATCH_CONDITION="det",  # det loc
            ENTITIES_CONVFC_FEAT=dict(
                POOLER_RESOLUTION=7,
                POOLER_SAMPLING_RATIO=0,
                NUM_CONV=3,
                CONV_DIM=256,
                NUM_FC=2,
                FC_DIM=1024,
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
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_EPOCH=150,
            MAX_ITER=None,
            WARMUP_ITERS=4000,
            STEPS=(100,),
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
        IMS_PER_BATCH=28,  # 四卡时候的batchsize
        IMS_PER_DEVICE=7,
        CHECKPOINT_PERIOD=3,
    ),
    TEST=dict(
        EVAL_PERIOD=1,
        RELATION=dict(
            MULTIPLE_PREDS=False,
            IOU_THRESHOLD=0.5,
            EVAL_POST_PROC=True,

        )
    ),
    GLOBAL=dict(
        DUMP_TEST=True,
        LOG_INTERVAL=100
    ),
    EXT_KNOWLEDGE=dict(
        GLOVE_DIR="/group/rongjie/cvpods/datasets/vg/glove",
    )
)


class OneStageRelDetrBASEConfig(DETRConfig):
    def __init__(self):
        super(OneStageRelDetrBASEConfig, self).__init__()
        self._register_configuration(_config_dict)


