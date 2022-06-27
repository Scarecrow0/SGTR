import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs import BaseDetectionConfig
from cvpods.configs.detr_config import DETRConfig
from cvpods.configs.vrd_config import VRDConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

_config_dict = dict(
    DEBUG=False,
    EXPERIMENT_NAME="oiv6_rel_detr_v1_baseline-finetune_detr-12lyer_dec-128-query",
    MODEL=dict(
        TEST_WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-22_09-42-oiv6_rel_detr_v1_baseline_transformer_more_param-n-query/model_0025283.pth",
        # TEST_WEIGHTS="/p300/outputs/cvpods/playground/sgg/vg/detr.res101.c5.one_stage_rel_tfmer/2021-03-22_13-37-vg_rel_detr_v1_baseline-12lyer_dec-128-query/model_0056315.pth",
    ),

    DATASETS=dict(
        TRAIN=("vgs_train",),
        TEST=("vgs_val",),
        # TRAIN=("oi_v6_train",),
        # TEST=("oi_v6_val",),
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
    TEST=dict(
        EVAL_PERIOD=1,
        RELATION=dict(
            MULTIPLE_PREDS=False,
            IOU_THRESHOLD=0.5,
            EVAL_POST_PROC=False,

        )
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
)


class OneStageDetrVRDConfig(DETRConfig):
    def __init__(self):
        super(OneStageDetrVRDConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageDetrVRDConfig()
