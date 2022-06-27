import math

import numpy as np
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F

from cvpods.layers import ShapeSpec, Conv2d
from cvpods.layers.batch_norm import get_norm
from cvpods.layers.roi_align import ROIAlign
from cvpods.modeling.nn_utils import weight_init
from cvpods.modeling.poolers import (
    assign_boxes_to_levels,
    convert_boxes_to_pooler_format,
)
from cvpods.structures import Boxes
from cvpods.structures.boxes import union_box


class RelationGeometryFeatHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pooler_resolution = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.POOLER_RESOLUTION

        self.rect_size = pooler_resolution * 4 - 1

        out_channels = 128

        self._output_size = (
            out_channels,
            None,
            None,
        )


        self.rect_geo_feat_conv = nn.Sequential(
            *[
                nn.Conv2d(
                    2, out_channels // 2, kernel_size=7, stride=2, padding=3, bias=True
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // 2, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(
                    out_channels // 2,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels, momentum=0.01),
            ]
        )

    @property
    def output_size(self):
        return self._output_size

    def forward(self, entities_box, image_size, rel_pair_idxs):
        rect_inputs = []

        for proposal_box, img_sz, rel_pair_idx in zip(
                entities_box, image_size, rel_pair_idxs
        ):
            # use range to construct rectangle, sized (rect_size, rect_size)
            device = proposal_box.device  # keep same device with the model param
            num_rel = len(rel_pair_idx)
            dummy_x_range = (
                torch.arange(self.rect_size, device=device)
                    .view(1, 1, -1)
                    .expand(num_rel, self.rect_size, self.rect_size)
            )
            dummy_y_range = (
                torch.arange(self.rect_size, device=device)
                    .view(1, -1, 1)
                    .expand(num_rel, self.rect_size, self.rect_size)
            )
            # resize bbox to the scale rect_size

            head_proposal_box = proposal_box[rel_pair_idx[:, 0]]
            head_proposal_box.scale(
                self.rect_size / img_sz[0], self.rect_size / img_sz[1]
            )

            def build_rect(proposal_box):
                ret = (
                        (dummy_x_range >= proposal_box.tensor[:, 0].floor().view(-1, 1, 1).long())
                        & (dummy_x_range <= proposal_box.tensor[:, 2].ceil().view(-1, 1, 1).long())
                        & (dummy_y_range >= proposal_box.tensor[:, 1].floor().view(-1, 1, 1).long())
                        & (dummy_y_range <= proposal_box.tensor[:, 3].ceil().view(-1, 1, 1).long())
                )
                return ret.float()

            head_rect = build_rect(head_proposal_box)

            tail_proposal_box = proposal_box[rel_pair_idx[:, 1]]
            tail_proposal_box.scale(
                self.rect_size / img_sz[0], self.rect_size / img_sz[1]
            )
            tail_rect = build_rect(tail_proposal_box)

            # (num_rel, 4, rect_size, rect_size)
            rect_input = torch.stack((head_rect, tail_rect), dim=1)
            rect_inputs.append(rect_input)

        rect_inputs = torch.cat(rect_inputs, dim=0).float().to(device)
        rect_features = self.rect_geo_feat_conv(rect_inputs)

        # try:
        #     rect_features = self.rect_geo_feat_conv(rect_inputs)
        #
        # except RuntimeError:
        #     pdb.set_trace()

        return rect_features


class RelationConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, geo_feat_input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.NUM_CONV
        conv_dim = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.CONV_DIM
        num_fc = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.NUM_FC
        fc_dim = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.FC_DIM
        norm = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.NORM
        # fmt: on
        assert num_conv + num_fc > 0
        assert num_conv > 0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        input_channels = self._output_size[0] + geo_feat_input_shape[0]
        for k in range(num_conv):
            if k > 0:
                input_channels = conv_dim
            conv = Conv2d(
                input_channels,
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, geo_feat):
        x = torch.cat((geo_feat, x), dim=1)

        for layer in self.conv_norm_relus:
            x = layer(x)


        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


class RelationROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (tuple[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.
                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.level_poolers = nn.ModuleList(
            ROIAlign(
                output_size,
                spatial_scale=scale,
                sampling_ratio=sampling_ratio,
                aligned=True,
            )
            for scale in scales
        )
        """
            elif pooler_type == "ROIPool":
                self.level_poolers = nn.ModuleList(
                    RoIPool(output_size, spatial_scale=scale) for scale in scales
                )
            elif pooler_type == "ROIAlignRotated":
                self.level_poolers = nn.ModuleList(
                    ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                    for scale in scales
                )
            elif pooler_type == "PSROIPool":
                self.level_poolers = nn.ModuleList(
                    PSRoIPool(output_size, spatial_scale=scale)
                    for scale in scales
                )
            elif pooler_type == "PSROIAlign":
                self.level_poolers = nn.ModuleList(
                    PSRoIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                    for scale in scales
                )
            else:
                raise ValueError("Unknown pooler type: {}".format(pooler_type))
        """

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
                len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            # When there is only one feature map, canonical_level is redundant and we should not
            # require it to be a sensible value. Therefore we skip this assertion
            assert (
                    self.min_level <= canonical_level and canonical_level <= self.max_level
            )
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )
        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size,
                device=x[0].device,
                dtype=x[0].dtype,
            )
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        level_assignments = assign_boxes_to_levels(
            box_lists,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level,
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]

            # output_cpu = pooler(x_level.cpu(), pooler_fmt_boxes_level.cpu())
            # output[inds] = output_cpu.cuda()

            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output


class RelationshipFeatureHead(nn.Module):
    def __init__(self, cfg, rel_pooler, input_shape):
        super(RelationshipFeatureHead, self).__init__()
        self.cfg = cfg

        self.rel_pooler = rel_pooler
        self.geo_rect_feat_head = RelationGeometryFeatHead(cfg)
        self.rel_convfc_feat_head = RelationConvFCHead(cfg, input_shape, 
                                                        self.geo_rect_feat_head.output_size)

        self._output_size = self.rel_convfc_feat_head.output_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x, rel_proposals):
        #   build the union box
        phr_union_boxes = []
        ent_boxes = []
        pair_indxs = []
        image_size = []
        rel_batch_sizes = [len(p) for p in rel_proposals]
        for rel_prop in rel_proposals:
            phr_union_boxes.append(
                union_box(
                    rel_prop.instances.pred_boxes[rel_prop.rel_pair_tensor[:, 0]],
                    rel_prop.instances.pred_boxes[rel_prop.rel_pair_tensor[:, 1]],
                )
            )

            ent_boxes.append(rel_prop.instances.pred_boxes)
            image_size.append(rel_prop.instances.image_size)
            pair_indxs.append(rel_prop.rel_pair_tensor)
        #   extract union feat
        phr_union_feats = self.rel_pooler(x, phr_union_boxes)

        #   extract the geometry feature
        phr_geo_feats = self.geo_rect_feat_head(ent_boxes, image_size, pair_indxs)

        # fuse the features by additional modules
        phr_union_feats = self.rel_convfc_feat_head(phr_union_feats, phr_geo_feats)

        return torch.split(phr_union_feats, rel_batch_sizes)


class ConvFCHeadWithAttention(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, param_dicts=None):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        if param_dicts is None:
            param_dicts = cfg.MODEL.ROI_BOX_HEAD

        num_conv = param_dicts.NUM_CONV
        conv_dim = param_dicts.CONV_DIM
        num_fc = param_dicts.NUM_FC
        fc_dim = param_dicts.FC_DIM
        norm = param_dicts.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for idx, layer in enumerate(self.conv_norm_relus):
            weight_init.c2_msra_fill(layer)

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size
