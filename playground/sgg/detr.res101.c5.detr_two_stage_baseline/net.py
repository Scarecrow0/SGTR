from cvpods.layers import ShapeSpec
from cvpods.modeling import build_resnet_backbone
from cvpods.modeling.backbone import Backbone

import logging
import sys

from cvpods.modeling.roi_heads.twostg_vrd_head import TwoStageRelationROIHeads

sys.path.append("..")

from modeling.meta_arch.detr_twostg_vrd import TwoStageDETRVRD
from modeling.meta_arch.relation_head import  TwoStageRelationROIHeadsBGNN

def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone

def build_relation_head(cfg, input_shape):
    if cfg.MODEL.ROI_RELATION_HEAD.FEATURE_NECK.NAME == 'bgnn':
        return TwoStageRelationROIHeadsBGNN(cfg, input_shape)
    else:
        return TwoStageRelationROIHeads(cfg, input_shape)



def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_relation_head = build_relation_head
    model = TwoStageDETRVRD(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
