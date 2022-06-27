from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.proposal_generator import RPN

import sys
sys.path.append("..")


from cvpods.modeling.roi_heads.entities_roi_head import *
from cvpods.modeling.roi_heads.twostg_vrd_head import *
from cvpods.modeling.meta_arch.twostg_vrd import *

def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    return RelationEntitiesROIHeads(cfg, input_shape)


def build_relation_head(cfg, input_shape):
    return TwoStageRelationROIHeads(cfg, input_shape)


def build_box_head(cfg, input_shape):
    return FastRCNNConvFCHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator
    cfg.build_roi_heads = build_roi_heads
    cfg.build_box_head = build_box_head
    cfg.build_relation_head = build_relation_head

    model = TwoStageVRD(cfg)
    return model
