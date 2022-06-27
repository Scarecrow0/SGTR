# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone.resnet import BottleneckBlock, make_stage
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from cvpods.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from cvpods.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference, keypoint_rcnn_loss
from cvpods.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from cvpods.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints, \
    ROIHeads, Res5ROIHeads
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import get_event_storage

logger = logging.getLogger(__name__)

"""
This module is for implementation of entities detection in SGG tasks.
The rois head is modified from standard ROI heads, the behavior is changed 
for relationship detection 

"""


class RelationEntitiesROIHeads(ROIHeads):
    """
    It's "RelationEntities" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(RelationEntitiesROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        # self._init_mask_head(cfg)
        # self._init_keypoint_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # use the box predict by the roi head for the mask or keypoint prediction
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If RelationEntitiesROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = cfg.build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    @torch.no_grad()
    def label_proposals(
            self, proposals: List["Instances"], targets: List["Instances"]
    ) -> List["Instances"]:
        """
        In relation mode we just add the GT on to the proposals

        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # assign GT labels
            gt_classes = targets_per_image.gt_classes
            has_gt = gt_classes.numel() > 0

            # Get the corresponding GT for each proposal
            if has_gt:
                gt_classes = gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[matched_labels == 0] = self.num_classes
                # Label ignore proposals (-1 label)
                gt_classes[matched_labels == -1] = -1
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

            # Set target attributes of the sampled proposals:
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").

            if has_gt:
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[matched_idxs])

                proposals_per_image.matched_gt_idx = matched_idxs
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if targets is not None:
            proposals = self.label_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]

        # in relationship detection model, the detector works in inference mode
        # if we have gt, we can compute the losses
        pred_entity_instances, losses = self._forward_box(features_list, proposals)

        return pred_entity_instances, losses

    def _forward_box(
            self, features: List[torch.Tensor], proposals: List[Instances]
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        loss = None
        if proposals[0].has("gt_boxes"):
            loss = outputs.losses()

        pred_instances, keep_inds = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_nms_type,
            self.test_detections_per_img
        )

        # add the gt to the prediction if GT is existed
        for idx, prop in enumerate(pred_instances):
            if proposals[0].has("gt_boxes"):
                prop.gt_boxes = proposals[idx].gt_boxes[keep_inds[idx]]
            if proposals[0].has("gt_classes"):
                prop.gt_classes = proposals[idx].gt_classes[keep_inds[idx]]

        return pred_instances, loss

class RelationEntitiesRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """


    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled


        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )


        loss = {}
        if proposals[0].has("gt_boxes"):
            loss = outputs.losses()

        pred_instances, keep_inds = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_nms_type,
            self.test_detections_per_img
        )

        # add the gt to the prediction if GT is existed
        for idx, prop in enumerate(pred_instances):
            if proposals[0].has("gt_boxes"):
                prop.gt_boxes = proposals[idx].gt_boxes[keep_inds[idx]]
            if proposals[0].has("gt_classes"):
                prop.gt_classes = proposals[idx].gt_classes[keep_inds[idx]]

        return pred_instances, loss

