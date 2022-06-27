from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.data.datasets.builtin_meta import get_dataset_statistics
from cvpods.layers import ShapeSpec
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from cvpods.structures import Instances, pairwise_iou
from cvpods.structures.relationship import Relationships, squeeze_tensor
from cvpods.modeling.roi_heads.relation_head.rel_classifier import build_rel_classifier, FrequencyBias
from cvpods.modeling.roi_heads.twostg_vrd_head import TwoStageRelationROIHeads, RelationshipOutput
from cvpods.evaluation.boxlist import BoxList
from .bgnn.rce_loss import RelAwareLoss
from .bgnn.bgnn_rel_feat_neck import BGNNFeatureNeck


class TwoStageRelationROIHeadsBGNN(TwoStageRelationROIHeads):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(TwoStageRelationROIHeadsBGNN, self).__init__(cfg, input_shape)

        self.relation_feature_neck = BGNNFeatureNeck(
            cfg,
            rel_feat_input_shape=self.rel_feat_head.output_size,
            ent_feat_input_shape=cfg.MODEL.DETR.TRANSFORMER.D_MODEL,
        )

    def forward(self, features, entities_instance, gt_relationships, entities_features=None):
        """[summary]

        Args:
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            entities_instance  (list[Instances]):
                fields:
                 ['pred_boxes', 'scores', 'pred_classes',
                      'pred_boxes_per_cls', 'pred_score_dist',
                      (in training time)'gt_boxes', 'gt_classes']

                length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".

            gt_relationships (list[Relationships]):
                contains GT instance, which is the gt_entities
                (list[Instances]):
                    ['gt_boxes', 'gt_classes']

        """

        features_list = [features[f] for f in self.in_features]

        # the feature is in splited form
        # list(Tensor) -> list (B x Tensor(N, D))
        if entities_features is None:
            entities_features = self.entities_feat_extraction(
                features_list, entities_instance
            )

        # build pairs, fg_bg matching and resampling in training
        relationship_proposal = self.relation_proposal_gen(
            entities_instance, gt_relationships
        )

        rel_features = self.rel_feat_head(features_list, relationship_proposal)
        rel_features, entities_features, add_outputs = self.relation_feature_neck(
            relationship_proposal, rel_features, entities_features
        )
        (rel_pred_logits, entities_pred_logits) = self.relation_predictor(
            relationship_proposal, rel_features
        )

        outputs = RelationshipOutputRCE(
            self.cfg, relationship_proposal, rel_pred_logits, entities_pred_logits, 
            add_outputs,

        )

        if self.training:
            losses = outputs.losses()

            return [], losses
        else:
            pred_relationships = outputs.inference()
            return pred_relationships, {}


class RelationshipOutputRCE(RelationshipOutput):
    """calculate the loss in training and post processing while testing."""

    def __init__(
        self, cfg, relationship_proposal, rel_pred_logits, entities_logits, add_outputs
    ):
        super(RelationshipOutputRCE, self).__init__(cfg, relationship_proposal, rel_pred_logits, entities_logits,)
        self.rel_conf_loss = RelAwareLoss(cfg)

        rce_logits=add_outputs['conf_est_logits_each_iter']
        rce_pred_mat=add_outputs['relatedness_each_iters']

        if self.rel_proposals[0].has("rel_label"):
            self.gt_rel_labels_rce = torch.cat(
                [each.rel_label for each in self.rel_proposals]
            )
            self.gt_rel_labels_rce[self.gt_rel_labels_rce > 0] = 1
        self.rce_logits = rce_logits
        self.rce_pred_mat = rce_pred_mat

    def RCELoss(self):
        loss_dict = {
            f"rce_loss_{i}": self.rel_conf_loss(rce_loss, self.gt_rel_labels_rce)
            for i, rce_loss in enumerate(self.rce_logits)
        }
        return loss_dict

    def losses(self):
        loss_dict = {}
        loss_dict.update(self.predicate_cls_softmax_ce_loss())
        loss_dict.update(self.RCELoss())
        # todo entities loss
        return loss_dict


