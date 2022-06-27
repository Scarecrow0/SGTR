from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone.fpn import (
    build_resnet_backbone,
    build_shufflenetv2_backbone,
    build_mobilenetv2_backbone,
    FPN,
    LastLevelMaxPool
)
from .swin_transformer import build_swint_backbone
from .timm_backbone import build_timm_backbone


# TODO refine backbone name into cfg
def build_fpn_backbone(cfg, input_shape: ShapeSpec, backbone_name="resnet"):
    """
    Args:
        cfg: a cvpods CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if backbone_name == "resnet":
        bottom_up = build_resnet_backbone(cfg, input_shape)
    elif backbone_name == "shufflev2":
        bottom_up = build_shufflenetv2_backbone(cfg, input_shape)
    elif backbone_name == "mobilev2":
        bottom_up = build_mobilenetv2_backbone(cfg, input_shape)
    elif backbone_name == "timm":
        bottom_up = build_timm_backbone(cfg, input_shape)
    elif backbone_name == "swin":
        bottom_up = build_swint_backbone(cfg, input_shape)
    else:
        raise ValueError("No such backbone: {}".format(backbone_name))

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

def build_swin_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_fpn_backbone(cfg, input_shape, backbone_name='swin')
