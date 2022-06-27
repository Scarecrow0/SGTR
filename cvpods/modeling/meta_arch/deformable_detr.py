#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

"""
DETR model and criterion classes.
"""
import copy
import math
from typing import List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from cvpods.layers import ShapeSpec, position_encoding_dict
from cvpods.layers.deformable_transformer import DeformableTransformer, NestedTensor
# from cvpods.modeling.matcher import HungarianMatcher
from cvpods.structures import Boxes, ImageList, Instances
from cvpods.structures import boxes as box_ops
from cvpods.structures.boxes import box_cxcywh_to_xyxy
from cvpods.structures.boxes import generalized_box_iou
from cvpods.structures.relationship import squeeze_tensor
from cvpods.utils.metrics import accuracy


class Joiner(nn.Module):
    def __init__(self, cfg, position_embedding):
        super().__init__()
        self.position_embedding = position_embedding
        self.backbone_feats_keys = cfg.MODEL.DETR.DEFORMABLE_MODEL.IN_FEATURES

    def forward(self, backbone_feats, masks):

        out: List[NestedTensor] = []
        pos = []
        for name in self.backbone_feats_keys:
            x = backbone_feats[name]
            resized_mask = F.interpolate(masks[None].float(), size=x.shape[-2:]).to(
                torch.bool
            )[0]
            out.append(NestedTensor(x, resized_mask))

        # position encoding
        for x in out:
            pos.append(self.position_embedding(x.tensors, x.mask))
        return out, pos


class DeformableDETR(nn.Module):
    def __init__(self, cfg):
        super(DeformableDETR, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg
        # Build Backbone
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        backbone_out_shapes = self.backbone.output_shape()
        self.in_features_name = cfg.MODEL.DETR.DEFORMABLE_MODEL.IN_FEATURES
        self.num_feature_levels = cfg.MODEL.DETR.DEFORMABLE_MODEL.FEAT_LEVEL_NUM
        self.num_backbone_outs = len(self.in_features_name)

        # Build Transformer

        d_model = cfg.MODEL.DETR.TRANSFORMER.D_MODEL
        nhead = cfg.MODEL.DETR.TRANSFORMER.N_HEAD
        num_encoder_layers = cfg.MODEL.DETR.TRANSFORMER.NUM_ENC_LAYERS
        num_decoder_layers = cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS
        dim_feedforward = cfg.MODEL.DETR.TRANSFORMER.DIM_FFN
        dropout = cfg.MODEL.DETR.TRANSFORMER.DROPOUT_RATE
        activation = cfg.MODEL.DETR.TRANSFORMER.ACTIVATION
        normalize_before = cfg.MODEL.DETR.TRANSFORMER.PRE_NORM
        return_intermediate_dec = cfg.MODEL.DETR.TRANSFORMER.RETURN_INTERMEDIATE_DEC
        self.two_stage = cfg.MODEL.DETR.DEFORMABLE_MODEL.TWO_STAGE
        two_stage_num_proposals = (
            cfg.MODEL.DETR.DEFORMABLE_MODEL.TWO_STAGE_NUM_PROPOSALS
        )
        self.transformer = DeformableTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=self.two_stage,
            two_stage_num_proposals=two_stage_num_proposals,
        )

        self.aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_queries = cfg.MODEL.DETR.NUM_QUERIES

        # pre transformer modules
        hidden_dim = self.transformer.d_model
        self.position_embedding = position_encoding_dict[
            cfg.MODEL.DETR.POSITION_EMBEDDING
        ](
            num_pos_feats=hidden_dim // 2,
            temperature=cfg.MODEL.DETR.TEMPERATURE,
            normalize=True if cfg.MODEL.DETR.POSITION_EMBEDDING == "sine" else False,
            scale=None,
        )

        self.joiner = Joiner(cfg, self.position_embedding)

        input_proj_list = []

        for idx_out, out_name in enumerate(self.in_features_name):
            in_channels = self.backbone.output_shape()[out_name].channels
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )

        for _ in range(self.num_feature_levels - self.num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim

        self.input_proj = nn.ModuleList(input_proj_list)

        # Build FFN
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.two_stage
            else self.transformer.decoder.num_layers
        )
        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # FFN initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.two_stage
            else self.transformer.decoder.num_layers
        )
        self.with_box_refine = self.cfg.MODEL.DETR.DEFORMABLE_MODEL.BOX_REFINE
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

        if self.two_stage:
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self._setup_two_stage_predictor()


        # Build Object Queries
        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }

        if self.aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                self.aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(self.aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]

        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DETR.COST_CLASS,
            cost_bbox=cfg.MODEL.DETR.COST_BBOX,
            cost_giou=cfg.MODEL.DETR.COST_GIOU,
        )

        self.criterion = SetCriterion(
            self.num_classes, matcher, weight_dict=self.weight_dict, losses=losses
        )

        # self.criterion = SetCriterion(
        #     self.num_classes,
        #     matcher=matcher,
        #     weight_dict=self.weight_dict,
        #     eos_coef=cfg.MODEL.DETR.EOS_COEFF,
        #     losses=losses,
        # )

        self.post_processors = {"bbox": PostProcess()}

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        if not cfg.MODEL.RESNETS.STRIDE_IN_1X1:
            # Custom or torch pretrain weights
            self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        else:
            # MSRA pretrain weights
            self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def _setup_two_stage_predictor(self):
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        # futhermore, there shares the same class_embed and bbox_embed between 
        # the first stage prediction output from the encoder 
        # and the second stage prediction output from the decoder.
        if self.with_box_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        if self.two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)

        B, C, H, W = images.tensor.shape
        device = images.tensor.device

        mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
        for img_shape, m in zip(images.image_sizes, mask):
            m[: img_shape[0], : img_shape[1]] = False

        backbone_feats = self.backbone(images.tensor)
        features, pos = self.joiner(backbone_feats, mask)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, curr_mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(curr_mask)
            assert curr_mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                curr_mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.position_embedding(src, curr_mask).to(src.dtype)
                srcs.append(src)
                masks.append(curr_mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        """
        src, masks, pos: lvl, bz, chnl, w, h
        """
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        box = outputs_coord[-1]
        box_for_check = box_cxcywh_to_xyxy(box)

        vaild_x = box_for_check[:, :, 2:] >= box_for_check[:, :, :2]
        vaild_y = box_for_check[:, :, 2:] >= box_for_check[:, :, :2]
        vaild_box_idx = torch.logical_or(vaild_x, vaild_y)

        invaild_box_idx = torch.logical_not(torch.logical_and(vaild_box_idx[:, :, 0], vaild_box_idx[:, :, 1]))

        if len(squeeze_tensor(torch.nonzero(invaild_box_idx))) > 0:
            print(invaild_box_idx)
            print(outputs_coord[:, invaild_box_idx, :])

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        enc_outputs = None
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            enc_outputs = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        if self.training:

            targets = self.convert_anno_format(batched_inputs)

            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

            loss_dict = self.criterion(out, targets)

            if enc_outputs is not None:
                loss_encoder_dict = self.criterion(enc_outputs, targets)
                loss_encoder_dict = {
                    k + f"_enc": v for k, v in loss_encoder_dict.items()
                }
                loss_dict.update(loss_encoder_dict)

            for k, v in loss_dict.items():
                loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
            return loss_dict
        else:
            target_sizes = torch.stack(
                [
                    torch.tensor(
                        [bi.get("height", img_size[0]), bi.get("width", img_size[1])],
                        device=self.device,
                    )
                    for bi, img_size in zip(batched_inputs, images.image_sizes)
                ]
            )
            res = self.post_processors["bbox"](out, target_sizes)

            processed_results = []
            # for results_per_image, input_per_image, image_size in zip(
            for det_res_per_image, _, image_size in zip(
                    res, batched_inputs, images.image_sizes
            ):
                result = Instances(image_size)
                result.pred_boxes = Boxes(det_res_per_image["boxes"].float())
                result.scores = det_res_per_image["scores"].float()
                result.pred_classes = det_res_per_image["labels"]
                result.pred_score_dist = det_res_per_image["prob"]
                processed_results.append({"instances": result})

            return processed_results

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(img) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]
            boxes = box_ops.box_xyxy_to_cxcywh(
                bi["instances"].gt_boxes.tensor
                / torch.tensor([w, h, w, h], dtype=torch.float32)
            )
            target["boxes"] = boxes.to(self.device)
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor(
                [bi["height"], bi["width"]], device=self.device
            )
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)
            targets.append(target)

        return targets


def sigmoid_focal_loss(
        inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
                cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1
            )  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                    (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_giou * cost_giou
            )
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        assert self.num_classes == src_logits.shape[2]

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, topk=100):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values_all, topk_indexes_all = torch.sort(prob.view(out_logits.shape[0], -1), dim=-1, descending=True)
        topk_values = topk_values_all[:, :topk]
        topk_indexes = topk_indexes_all[:, :topk]
        topk_indexes_all = topk_indexes_all[:, :out_logits.shape[1]] // out_logits.shape[2]

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        out_bbox_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes_norm = torch.gather(out_bbox_xyxy, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        topk_inst_prob = torch.gather(prob, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, prob.shape[-1]))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes_norm * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b, "prob": p, "boxes_norm": b_n, "init_idx": idx}
            for s, l, b, p, idx, b_n in zip(scores, labels, boxes, topk_inst_prob, topk_indexes_all, boxes_norm)
        ]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
