import os
import os.path as osp

import cvpods
from cvpods.configs.rel_detr_config import OneStageRelDetrBASEConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])


aux_loss_weight = 0.5

rel_dec_layer = 6
ent_dec_layer = 6

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,

    EXPERIMENT_NAME=f"(top1)VG-SGTR-dec_layer{rel_dec_layer}",

    # the program will use this config to uptate the initial config(over write the existed, ignore doesnt existed)
    OVERIDE_CFG_DIR="",
    # OVERIDE_CFG_DIR=f"/group/rongjie/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/log/{exp_name}/config.json",

    MODEL=dict(

        LOAD_PROPOSALS=False,
        WEIGHTS_LOAD_MAPPING={
            "obj_class_embed": "class_embed",
            "obj_bbox_embed": "bbox_embed",
            "sub_class_embed": "class_embed",
            "sub_bbox_embed": "bbox_embed",
        },

        WEIGHTS_FIXED=[
            "backbone",
            "transformer.encoder",
            "transformer.decoder",
        ],

        #####
        TEST_WEIGHTS="/public/home/lirj2/projects/sgtr_release/playground/sgg/detr.res101.c5.one_stage_rel_tfmer/log/2022-06-20_11-25-(top1)VG-SGTR-dec_layer6_add_ent_hs/model_0059999.pth",

        WEIGHTS="/storage/data/lirj2/ckpts/vg_detr.pth",

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
                INTERACTIVE_REL_DECODER=dict(
                    ENT_DEC_EACH_LVL=True,
                    UPDATE_QUERY_BY_REL_HS=False,
                ),

                USE_ENTITIES_PRED=False, # use the entity prediction from predicate entity sub decoder rather than graph assembling 
                ENTITIES_INDEXING=True, # training time Graph assembling

                USE_ENTITIES_INDEXING_RANKING=False,
                INDEXING_TYPE="rule_base",  # the entity  matching method for graph assembeling; other options: feat_att rule_base 
                INDEXING_TYPE_INFERENCE="rule_base",  # rel_vec
                
                INDEXING_FOCAL_LOSS=dict(
                    ALPHA=0.8,
                    GAMMA=0.0,
                ),


                NUM_FUSE_LAYER=ent_dec_layer,  # for cross encoder

                NUM_DEC_LAYERS=ent_dec_layer,

                ENT_CLS_LOSS_COEFF=0.3,
                ENT_BOX_L1_LOSS_COEFF=0.5,  
                ENT_BOX_GIOU_LOSS_COEFF=0.5,
                ENT_INDEXING_LOSS_COEFF=0.5,

                COST_ENT_CLS=0.5,
                COST_BOX_L1=0.6,
                COST_BOX_GIOU=1.0,
                COST_INDEXING=0.01,
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
            COST_REL_VEC=0.6,

            CLASS_LOSS_COEFF=1.0,
            REL_VEC_LOSS_COEFF=1.0,

            EOS_COEFF=0.08,  # Relative classification weight of the no-object class
            OVERLAP_THRES=0.8,
            NUM_ENTITIES_PAIRING=3, # topk entity selection for graph assembling
            NUM_ENTITIES_PAIRING_TRAIN=25,
            NUM_MAX_REL_PRED=4096,

            MATCHING_RANGE=4096, # efficient matching
            NUM_MATCHING_PER_GT=1,

            DYNAMIC_QUERY=True, # dynamic query generation
            DYNAMIC_QUERY_AUX_LOSS_WEIGHT=None,

            NORMED_REL_VEC_DIST=False,
            FOCAL_LOSS=dict(ENABLED=True, ALPHA=0.25, GAMMA=2.0, ),
        ),

        ROI_RELATION_HEAD=dict(
            NUM_CLASSES=50,  # for VG
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,

                METHOD="bilvl",
                REPEAT_FACTOR=0.2,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,

                ENTITY={
                    "ENABLED": False,
                    "REPEAT_FACTOR": 0.2,
                    "INSTANCE_DROP_RATE": 1.5,
                    "REPEAT_DICT_DIR": None,
                },

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
        TEST=("vgs_val",), # ("vgs_test",), # test dataset
        PROPOSAL_FILES_TEST="/public/home/lirj2/vg_rcnnfpn_precompute_box/vgs_val",
        PROPOSAL_FILES_TRAIN="/public/home/lirj2/vg_rcnnfpn_precompute_box/train_new",
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
            GAMMA=0.1,
            MAX_EPOCH=None,
            MAX_ITER=2.6e5,
            WARMUP_ITERS=100,
            # STEPS=(9e4, 1.3e5), # SGTR
            STEPS=(5e4, 1.2e5),  # SGTR +
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
        IMS_PER_BATCH=24,  # 四卡时候的batchsize
        IMS_PER_DEVICE=6,
        CHECKPOINT_PERIOD=12000,
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
        RELATION=dict(MULTIPLE_PREDS=True, IOU_THRESHOLD=0.5, EVAL_POST_PROC=False, ),
    ),

    TRAINER=dict(
        FP16=dict(
            ENABLED=False,
        )
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
