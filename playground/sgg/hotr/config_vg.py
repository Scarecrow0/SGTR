import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.rel_detr_config import OneStageRelDetrBASEConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

loss_coeff = 0.3
match_cost = 0.4
loss_indexing_coeff = 0.5
ent_pair = 3
topk_gt = 2

aux_loss_weight = 0.5

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,
    DUMP_RATE=dict(
        TRAIN=0.0006,
        TEST=0.004
    ),

    EXPERIMENT_NAME=f"HOTR-init",
    # the program will use this config to uptate the initial config(over write the existed, ignore doesnt existed)
    OVERIDE_CFG_DIR="",
    # OVERIDE_CFG_DIR=f"/group/rongjie/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/log/{exp_name}/config.json",

    MODEL=dict(

        LOAD_PROPOSALS=False,
        WEIGHTS_LOAD_MAPPING={
            "interaction_decoder": "transformer.decoder"
        },

        WEIGHTS_FIXED=[
            "backbone",
            "transformer.encoder",
            "transformer.decoder",
        ],

        #####
        TEST_WEIGHTS="",
        WEIGHTS="/path/to/detr/weights",
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),
        DETR=dict(  # entities DETR
            IN_FEATURES="res5",
            NUM_CLASSES=150,  # for VG
            CLASS_LOSS_COEFF=1.0,
            BBOX_LOSS_COEFF=1.0,
            GIOU_LOSS_COEFF=1.0,
        ),

        REL_DETR=dict(  # relationship DETR
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                SHARE_ENC_FEAT_LAYERS=-1,
                NUM_ENC_LAYERS=3,  # set None will share the encoder with the entities part
                FOREGROUND_AWARENESS=dict(
                    ENABLED=False
                ),
                NUM_DEC_LAYERS=12,
                DIM_FFN=4096,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,
            ),

            CROSS_DECODER=dict(  # k_generator
                ENABLED=False,
                N_HEAD=8,
                NUM_DEC_LAYERS=4,
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

                USE_ENTITIES_PRED=False,

                SPACONDI_REL_DECODER=dict(
                    ENABLED=False
                ),

                ENTITIES_INDEXING=True,
                USE_ENTITIES_INDEXING_RANKING=False,
                INDEXING_TYPE="rule_base",  # feat_att, pred_att rule_base
                INDEXING_FOCAL_LOSS=dict(
                    ALPHA=0.01,
                    GAMMA=2.0,
                ),
                NUM_FUSE_LAYER=8,  # for cross encoder

                NUM_DEC_LAYERS=4,

                ENT_CLS_LOSS_COEFF=loss_coeff,
                ENT_BOX_L1_LOSS_COEFF=loss_coeff,
                ENT_BOX_GIOU_LOSS_COEFF=loss_coeff,
                ENT_INDEXING_LOSS_COEFF=loss_coeff,

                COST_ENT_CLS=match_cost,
                COST_BOX_L1=0.8,
                COST_BOX_GIOU=1.0,
                COST_INDEXING=0.01,
                COST_FOREGROUND_ENTITY=0.2,


            ),

            CONFINDENCE_ESTIMATOR=dict(
                ENABLED=False,
                DUAL_STAGE_PRED=False,
                SEMANTICS_INPUT=False,
                CONFINDENCE_GATING=False,  # use the confidence score to multiply on cls prediction
                FOCAL_LOSS=dict(
                    ALPHA=0.06,
                    GAMMA=2.5,
                ),
            ),

            NUM_PRED_EDGES=1,

            NO_AUX_LOSS=False,
            USE_FINAL_MATCH=True,
            USE_SAME_MATCHER=True,


            AUX_LOSS_WEIGHT=aux_loss_weight,

            NUM_QUERIES=160,

            COST_CLASS=1.0,
            COST_REL_VEC=3,
            # COST_REL_VEC=4,

            CLASS_LOSS_COEFF=1.0,
            REL_VEC_LOSS_COEFF=1.1,

            EOS_COEFF=0.08,  # Relative classification weight of the no-object class
            OVERLAP_THRES=0.8,
            NUM_ENTITIES_PAIRING=ent_pair,
            NUM_ENTITIES_PAIRING_TRAIN=80,
            NUM_MAX_REL_PRED=4096,
            MATCHING_RANGE=4096,

            NUM_MATCHING_PER_GT=topk_gt,

            NORMED_REL_VEC_DIST=False,
            FOCAL_LOSS=dict(ENABLED=True, ALPHA=0.25, GAMMA=2.0, ),
        ),

        ROI_RELATION_HEAD=dict(
            NUM_CLASSES=50,  # for VG
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.13,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
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
        PROPOSAL_FILES_TEST="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.600size.naive_baseline/log/2021-03-04_13-54-fixed_freq_bias-det_match_condi-more_param++_on_rel_feature/inference/vgs_val",
        PROPOSAL_FILES_TRAIN="/group/rongjie/cvpods/playground/sgg/vg/faster_rcnn.res101X.fpn.600size.naive_baseline/log/2021-03-04_13-54-fixed_freq_bias-det_match_condi-more_param++_on_rel_feature/inference_new/vgs_train"

    ),

    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(480, 496, 512, 536, 552, 576, 600,),
                    # short_edge_length=(600,),
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
            MAX_EPOCH=None,
            MAX_ITER=1.4e6,
            WARMUP_ITERS=100,
            STEPS=(7e5, 1e6),
        ),
        OPTIMIZER=dict(
            NAME="DETRAdamWBuilder",
            BASE_LR=8e-5,
            BASE_LR_RATIO_BACKBONE=1e-4,
            WEIGHT_DECAY=1e-4,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True, CLIP_VALUE=0.1, CLIP_TYPE="norm", NORM_TYPE=2.0,
        ),
        IMS_PER_BATCH=16,  # 四卡时候的batchsize
        IMS_PER_DEVICE=4,
        CHECKPOINT_PERIOD=6000,
    ),
    TEST=dict(
        EVAL_PERIOD=2000,
        RELATION=dict(MULTIPLE_PREDS=True, IOU_THRESHOLD=0.5, EVAL_POST_PROC=False, ),
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
    GLOBAL=dict(
        DUMP_TEST=False,
        DUMP_TRAIN=False,
        LOG_INTERVAL=100
    ),

)


class OneStageRelDetrConfig(OneStageRelDetrBASEConfig):
    def __init__(self):
        super(OneStageRelDetrConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageRelDetrConfig()


