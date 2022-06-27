

from .rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(

        MASK_ON=False,
        PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_MEAN=[103.530, 116.280, 123.675],

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
            FREQUENCY_BIAS=False,
            IGNORE_VALUE=255,
            PAIR_MATCH_CONDITION="loc",  # det loc


        ),
    ),
)


class VRDConfig(RCNNFPNConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = VRDConfig()