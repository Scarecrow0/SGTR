import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.rel_detr_config import OneStageRelDetrBASEConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])



aux_loss_weight = 0.5

rel_dec_layer = 6
ent_dec_layer = 6

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,

    EXPERIMENT_NAME=f"oiv6-SGTR-rel_dec-{rel_dec_layer}",
    # the program will use this config to uptate the initial config(over write the existed, ignore doesnt existed)
    OVERIDE_CFG_DIR="",


    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
            "obj_class_embed": "class_embed",
            "obj_bbox_embed": "bbox_embed",
            "sub_class_embed": "class_embed",
            "sub_bbox_embed": "bbox_embed",
        },

        WEIGHTS_FIXED=[
            "backbone",
            # "transformer.encoder",
            # "transformer.decoder",
        ],

        # detection pretrain weights
        WEIGHTS="/storage/data/lirj2/ckpts/detr_oiv6.pth",

        TEST_WEIGHTS="/public/home/lirj2/projects/sgtr_release/playground/sgg/detr.res101.c5.one_stage_rel_tfmer/log/2022-06-24_17-09-oiv6-SGTR-rel_dec-6/model_0107999.pth",

        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),

        DETR=dict(  # entities DETR
            IN_FEATURES="res5",
            NUM_CLASSES=601,  # for OIv6
        ),

        REL_DETR=dict(  # relationship DETR
            USE_GT_ENT_BOX=False,
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                SHARE_ENC_FEAT_LAYERS=-1,
                NUM_ENC_LAYERS=3,  # set None will share the encoder with the entities part
                NUM_DEC_LAYERS=rel_dec_layer,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,
            ),


            ENTITIES_AWARE_HEAD=dict(
                ENTITIES_AWARE_MATCHING=True,  # gt pred matching in training
                ENTITIES_AWARE_RANKING=True,  # pred entities pred relationship matching and ranking in test.
                CROSS_DECODER=True,
                ENABLED=True,


                # SGTR
                USE_ENTITIES_PRED=False,

                INTERACTIVE_REL_DECODER=dict(
                    ENT_DEC_EACH_LVL=True,
                    UPDATE_QUERY_BY_REL_HS=False,
                ),

                ENTITIES_INDEXING=True,
                USE_ENTITIES_INDEXING_RANKING=False,
                INDEXING_TYPE="rule_base",  # feat_att, pred_att rule_base
                INDEXING_TYPE_INFERENCE="rule_base",  # rel_vec
                
                INDEXING_FOCAL_LOSS=dict(
                    ALPHA=0.8,
                    GAMMA=0.0,
                ),


                NUM_FUSE_LAYER=ent_dec_layer,  # for cross encoder

                NUM_DEC_LAYERS=ent_dec_layer,

                ENT_CLS_LOSS_COEFF=0.3,
                ENT_BOX_L1_LOSS_COEFF=0.3,  
                ENT_BOX_GIOU_LOSS_COEFF=1.,
                ENT_INDEXING_LOSS_COEFF=0.0,

                COST_ENT_CLS=0.5,
                COST_BOX_L1=0.6,
                COST_BOX_GIOU=1.25,
                COST_INDEXING=0.00,
                COST_FOREGROUND_ENTITY=0.1,

                REUSE_ENT_MATCH=False,

                USE_REL_VEC_MATCH_ONLY=False,

            ),

            NUM_PRED_EDGES=1,

            NO_AUX_LOSS=False,
            USE_FINAL_MATCH=False,
            USE_SAME_MATCHER=True,

            AUX_LOSS_WEIGHT=aux_loss_weight,

            NUM_QUERIES=180,

            COST_CLASS=1.0,
            COST_REL_VEC=1.0,

            CLASS_LOSS_COEFF=1.0,
            REL_VEC_LOSS_COEFF=1.0,

            EOS_COEFF=0.08,  # Relative classification weight of the no-object class
            OVERLAP_THRES=0.8,
            NUM_ENTITIES_PAIRING=3,
            NUM_ENTITIES_PAIRING_TRAIN=40,
            NUM_MAX_REL_PRED=4096,
            MATCHING_RANGE=4096,

            NUM_MATCHING_PER_GT=1,

            DYNAMIC_QUERY=True,
            DYNAMIC_QUERY_AUX_LOSS_WEIGHT=None,

            NORMED_REL_VEC_DIST=False,
            FOCAL_LOSS=dict(ENABLED=False, ALPHA=0.25, GAMMA=2.0, ),
        ),


        ROI_RELATION_HEAD=dict(
            NUM_CLASSES=30,  # for OIV6
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.13,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
            ),

            LONGTAIL_PART_DICT=[None, 'h', 'h', 'h', 't', 'b', 't', 't', 'h', 'h', 'b', 't', 't', 't', 'b', 't', 't',
                                'h', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'h', 't'],

        ),
        PROPOSAL_GENERATOR=dict(
            FREEZE=False,
        ),
    ),
    DATASETS=dict(
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
                    short_edge_length=[480, 512, 544, 576, 608, 640, 672, 704, 720],
                    max_size=1000, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=720, max_size=1000, sample_style="choice")),
            ],
        )
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            GAMMA=0.1,
            MAX_EPOCH=None,
            MAX_ITER=2.4e5,
            WARMUP_ITERS=300,
            STEPS=(6e4, 1.1e5),
        ),
        OPTIMIZER=dict(
            NAME="DETRAdamWBuilder",
            BASE_LR=1e-4,
            BASE_LR_RATIO_BACKBONE=1e-4,
            WEIGHT_DECAY=1e-4,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True, CLIP_VALUE=0.1, CLIP_TYPE="norm", NORM_TYPE=2.0,
        ),
        IMS_PER_BATCH=24,  # 四卡时候的batchsize
        IMS_PER_DEVICE=6,
        CHECKPOINT_PERIOD=12000,
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
        RELATION=dict(MULTIPLE_PREDS=False, IOU_THRESHOLD=0.5, EVAL_POST_PROC=True, ),
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),

    GLOBAL=dict(
        DUMP_TEST=True,
        LOG_INTERVAL=100,
    ),

)


class OneStageRelDetrConfig(OneStageRelDetrBASEConfig):
    def __init__(self):
        super(OneStageRelDetrConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageRelDetrConfig()
