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
from cvpods.modeling.roi_heads.relation_head.rel_feat_neck import cat
from cvpods.modeling.roi_heads.relation_head .rel_roi_feat import RelationROIPooler, RelationshipFeatureHead
from cvpods.modeling.roi_heads.relation_head.bgnn.rce_loss import RelAwareLoss

class TwoStageRelationROIHeads(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.cfg = cfg
        print(input_shape)
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.in_features = [k  for k, v in input_shape.items()]
        # cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.batch_size_per_image = cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_RELATION_HEAD.SCORE_THRESH_TEST

        # for entites pooler, we keep the same with the original detection
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        entity_pooler_resolution = (
            cfg.MODEL.ROI_RELATION_HEAD.ENTITIES_CONVFC_FEAT.POOLER_RESOLUTION
        )
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)

        self.entities_pooler = ROIPooler(
            output_size=entity_pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == len(in_channels)
        in_channels = in_channels[0]

        self.entities_head = FastRCNNConvFCHead(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=entity_pooler_resolution,
                width=entity_pooler_resolution,
            ),
            param_dicts=cfg.MODEL.ROI_RELATION_HEAD.ENTITIES_CONVFC_FEAT
        )

        rel_uni_feat_pooler_resolution = (
            cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.POOLER_RESOLUTION
        )
        # pooler_type       = cfg.MODEL.ROI_RELATION_HEAD.POOLER_TYPE
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.UNION_FEAT.POOLER_SAMPLING_RATIO

        self.rel_pooler = RelationROIPooler(
            output_size=rel_uni_feat_pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,  # share with the entities boxes
            canonical_box_size=300,  # a little bit larger than ent det
        )
        self.rel_feat_head = RelationshipFeatureHead(
            cfg,
            self.rel_pooler,
            ShapeSpec(
                channels=in_channels,
                height=rel_uni_feat_pooler_resolution,
                width=rel_uni_feat_pooler_resolution,
            ),
        )

        self.relation_proposal_gen = RelationshipProposalGenerator(cfg)

        if cfg.MODEL.ROI_RELATION_HEAD.FEATURE_NECK.NAME == 'bgnn':
            from cvpods.modeling.roi_heads.relation_head.bgnn.bgnn_rel_feat_neck import BGNNFeatureNeck as RelationshipFeatureNeck
        else:
            from cvpods.modeling.roi_heads.relation_head.rel_feat_neck import RelationshipFeatureNeck

        self.relation_feature_neck = RelationshipFeatureNeck(
            cfg,
            rel_feat_input_shape=self.rel_feat_head.output_size,
            ent_feat_input_shape=self.entities_head.output_size,
        )
        self.relation_predictor = RelationshipPredictor(
            cfg,
            in_channels=self.relation_feature_neck.output_size,
            num_cls=cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
        )

        self.rel_loss = nn.CrossEntropyLoss()

    def entities_feat_extraction(self, features_list: list, entities_instance: list):
        """extract the entities feature from the feature maps

        Args:
            features_list (list): [description]
            entities_instance (list): [description]

        Returns:
            entities features list(Tensor): the feature is splitted into list form
        """
        boxes_sizes = [len(x) for x in entities_instance]
        entities_box_features = self.entities_pooler(
            features_list, [x.pred_boxes for x in entities_instance]
        )
        entities_box_features = self.entities_head(entities_box_features)

        return torch.split(entities_box_features, boxes_sizes)

    def forward(self, features, entities_instance, gt_relationships):
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

        RelationshipOutputFactory =  RelationshipOutput
        if self.cfg.MODEL.ROI_RELATION_HEAD.FEATURE_NECK.NAME == 'bgnn':
            RelationshipOutputFactory =  RelationshipOutputRCE

        outputs = RelationshipOutputFactory(
            self.cfg, relationship_proposal, rel_pred_logits, entities_pred_logits, add_outputs
        )

        if self.training:
            losses = outputs.losses()

            return [], losses
        else:
            pred_relationships = outputs.inference()
            return pred_relationships, {}


class RelationshipProposalGenerator(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.rel_matcher = RelMatcher(cfg)

        self.batch_size_per_image = cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION
        self.use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        self.max_proposal_pairs = cfg.MODEL.ROI_RELATION_HEAD.MAX_PROPOSAL_PAIR

    def label_and_sample_proposals(
        self,
        cand_matrix: torch.Tensor,
        entit_prop: Instances,
        rel_targets: Relationships,
    ):
        """
        match the fully connect relation proposal with the ground truth
            and sampling the foreground background for training

        Args:
            cand_matrix: N x N matrix indicates the entities associated with each other
            entit_prop: Instances predicted entities proposal
            rel_targets: Gt relationship

        Returns:
            rel_proposal

        """
        device = entit_prop.pred_boxes.device
        rel_prop_pairing_ind = torch.nonzero(cand_matrix)
        corrsp_gt_rel_idx, matching_qualities, binary_rel_matrixs = self.rel_matcher(
            rel_prop_pairing_ind, entit_prop, rel_targets,
        )
        # fgbg matching

        num_fg_samples = int(self.batch_size_per_image * self.positive_fraction)

        fg_rel_inds = squeeze_tensor(torch.nonzero(corrsp_gt_rel_idx >= 0))
        fg_rel_labs = rel_targets.rel_label[corrsp_gt_rel_idx[corrsp_gt_rel_idx >= 0]]
        fg_rel_triplets = torch.cat(
            (rel_prop_pairing_ind[fg_rel_inds], fg_rel_labs.view(-1, 1)), dim=1
        ).long()

        bg_rel_inds = squeeze_tensor(torch.nonzero(corrsp_gt_rel_idx < 0))
        bg_rel_labs = torch.zeros(
            bg_rel_inds.shape[0], dtype=torch.int64, device=device
        )
        bg_rel_triplets = torch.cat(
            (rel_prop_pairing_ind[bg_rel_inds], bg_rel_labs.view(-1, 1)), dim=-1
        ).long()

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), device=device).long()
        else:
            if fg_rel_triplets.shape[0] > num_fg_samples:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[
                    :num_fg_samples
                ]
                fg_rel_triplets = fg_rel_triplets[perm]
        

        num_neg_per_img = min(
            self.batch_size_per_image - fg_rel_triplets.shape[0],
            bg_rel_triplets.shape[0],
        )
        if bg_rel_triplets.shape[0] > 0:
            # samples from the pairs grouped by the high quality proposals
            pairs_qualities = matching_qualities[bg_rel_inds]
            # take the both entities boxes match with the foreground entities as the bg relationship
            pairs_qualities = pairs_qualities[
                pairs_qualities > self.rel_matcher.fg_thres ** 2 * 1.2
            ]

            _, sorted_idx = torch.sort(pairs_qualities, dim=0, descending=True)
            bg_rel_triplets = bg_rel_triplets[sorted_idx][: int(num_neg_per_img * 1.5)]
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[
                :num_neg_per_img
            ]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            _, idx = torch.sort(matching_qualities[bg_rel_inds], descending=True)
            selected_bg_rel_inds = bg_rel_inds[idx[:3]]
            
            selected_bg_rel_inds = idx[:5]
            bg_rel_triplets = torch.zeros((len(selected_bg_rel_inds), 3), dtype=torch.int64, device=device)
            bg_rel_triplets[:, :2] = rel_prop_pairing_ind[selected_bg_rel_inds]
            bg_rel_triplets[:, 2] = 0

        # print(f"fg {fg_rel_triplets.shape[0]}, bg {bg_rel_triplets.shape[0]}")
        rel_proposal = Relationships(
            instances=entit_prop,
            rel_pair_tensor=torch.cat(
                (fg_rel_triplets[:, :2], bg_rel_triplets[:, :2]), dim=0
            ),
            rel_label=torch.cat((fg_rel_triplets[:, 2], bg_rel_triplets[:, 2]), dim=0),
        )
        rel_proposal.add_meta_info('fg_pair_matrixs', binary_rel_matrixs)
        return rel_proposal

    def forward(self, entities_instance: list, rel_targets: list = None):

        rel_proposals = []

        for img_idx, entit_proposal in enumerate(entities_instance):

            device = entit_proposal.pred_boxes.device
            num_prp = len(entit_proposal)

            # fully connect the entities for relationship pairs instead of self-connection
            cand_matrix = torch.ones((num_prp, num_prp), device=device) - torch.eye(
                num_prp, device=device
            )

            if self.training:
                assert rel_targets
                rel_target = rel_targets[img_idx]  # take from the batch
                rel_proposal = self.label_and_sample_proposals(
                    cand_matrix, entit_proposal, rel_target
                )

            else:
                device = entit_proposal.pred_boxes.device
                # naive ranking mechanism by the entities detection score production
                rel_prop_pairing_ind = torch.nonzero(cand_matrix)
                rel_prop_quality = (
                    entit_proposal.scores[rel_prop_pairing_ind[:, 0]]
                    * entit_proposal.scores[rel_prop_pairing_ind[:, 1]]
                )
                _, top_idx = torch.sort(rel_prop_quality, descending=True)
                if len(rel_prop_pairing_ind) > 0:
                    rel_prop_pairing_ind = rel_prop_pairing_ind[top_idx[: self.max_proposal_pairs]]
                else:
                    rel_prop_pairing_ind = torch.zeros((3, 2), dtype=torch.int64, device=device)

                rel_proposal = Relationships(
                    instances=entit_proposal,
                    rel_pair_tensor=rel_prop_pairing_ind,
                )

            rel_proposals.append(rel_proposal)

        return rel_proposals


class RelMatcher:
    """
    matching between the relationship proposal and gt relationships
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.fg_thres = cfg.MODEL.ROI_RELATION_HEAD.FG_IOU_THRESHOLD
        pass

    def __call__(
        self,
        rel_prop_pair_idx: torch.Tensor,
        entities_instance: Instances,
        rel_targets: Relationships,
        matching_condi_name=None,
    ):
        """
         according to the entities boxes between the relationship proposal and GT relationship
         give the matchting results between the proposals and GT.

        Args:
            cand_matrix (torch.Tensor): M x 2, relationship pairing index indicators
            entities_instance (Instances): N entities
            rel_targets (Relationships): relation ship ground-truth

        Return:
            corrsp_gt_rel_idx: M' the corresponding GT index relationship proposals that match with the GT
            matching_qualities: M' the location overlapping qualities
        """

        prp_lab = entities_instance.pred_classes.long()
        tgt_ent = rel_targets.instances
        tgt_ent_lab = tgt_ent.gt_classes.long()
        device = entities_instance.pred_boxes.device


         # [tgt, prp]
        ious_match_mat = pairwise_iou(tgt_ent.gt_boxes, entities_instance.pred_boxes)

        det_match = (tgt_ent_lab[:, None] == prp_lab[None]) & (
            ious_match_mat > self.fg_thres
        )  # [tgt, prp]
        # one box may match multiple gt boxes here we just mark them as a valid matching if they
        # match any boxes
        locating_match = ious_match_mat > self.fg_thres  # [tgt, prp]

        locating_match_stat = (
            -1 * torch.ones((len(entities_instance)), device=device).long()
        )

        locating_match_idx = torch.nonzero(locating_match)
        locating_match_stat[locating_match_idx[:, 1]] = locating_match_idx[:, 0]
        entities_instance.locating_match = locating_match_stat

        # prepare gt data structure for matching
        tgt_rel_matrix = rel_targets.get_rel_matrix()
        tgt_pair_idxs = rel_targets.rel_pair_tensor
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = (
            tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
        )
        num_tgt_rels = tgt_rel_labs.shape[0]

        # use which entities matching results for relationship matching
        match_condit = {
            "loc": locating_match,
            "det": det_match,
        }

        if matching_condi_name is None:
            matching_condi_name = self.cfg.MODEL.ROI_RELATION_HEAD.PAIR_MATCH_CONDITION

        is_match = match_condit[matching_condi_name]

        # prepare proposal data structure
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[
            tgt_head_idxs
        ]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[
            tgt_tail_idxs
        ]  # num_tgt_rel, num_prp (matched prp tail)
        binary_rel_matrixs = torch.zeros((num_prp, num_prp), device=device).long()

        rel_matching_scores = torch.zeros((num_prp, num_prp), device=device)

        fg_rel_triplets = (
            []
        )  # (N' x 3) the relationship pairing index of entities proposals
        corrsp_gt_rel_idx = []  # N' the relationship proposal matching with the GT

        pred_gt_ious_match_score = torch.transpose(ious_match_mat, 1 ,0).max(dim=1)[0]
        ious_score = (
            (
                pred_gt_ious_match_score[rel_prop_pair_idx[:, 0]]
                * pred_gt_ious_match_score[rel_prop_pair_idx[:, 1]]
            )
            .view(-1)
            .detach()
        )
        rel_matching_scores[
            rel_prop_pair_idx[:, 0], rel_prop_pair_idx[:, 1]
        ] = ious_score

        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = (
                    bi_match_head.view(1, num_bi_head)
                    .expand(num_bi_tail, num_bi_head)
                    .contiguous()
                )
                bi_match_tail = (
                    bi_match_tail.view(num_bi_tail, 1)
                    .expand(num_bi_tail, num_bi_head)
                    .contiguous()
                )
                # binary rel only consider related or not, so its symmetric
                binary_rel_matrixs[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel_matrixs[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs from the boxes pairs matching with the ground truth
            prp_head_idxs = (
                prp_head_idxs.view(-1, 1)
                .expand(num_match_head, num_match_tail)
                .contiguous()
                .view(-1)
            )
            prp_tail_idxs = (
                prp_tail_idxs.view(1, -1)
                .expand(num_match_head, num_match_tail)
                .contiguous()
                .view(-1)
            )

            # remove the self connection
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue

            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]

            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor(
                [tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device
            ).view(-1, 1)

            fg_rel_i = torch.cat(
                (prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels),
                dim=-1,
            ).to(torch.int64)
            # select higher quality proposals as fg if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            # todo: the matching quality can be provide by different function

            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

            corrsp_gt_rel_idx.extend([i,] * fg_rel_i.shape[0])
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3)).long().to(device)
        else:
            fg_rel_triplets = torch.cat(fg_rel_triplets, dim=0).long().to(device)

        corrsp_gt_rel_idx = torch.Tensor(corrsp_gt_rel_idx).long().to(device)

        # matching between the give candidates pairing matrix

        fg_rel_matrixs = -1 * torch.ones((num_prp, num_prp), device=device).long()
        fg_rel_matrixs[fg_rel_triplets[:, 0], fg_rel_triplets[:, 1]] = corrsp_gt_rel_idx

        corrsp_gt_rel_idx = fg_rel_matrixs[
            rel_prop_pair_idx[:, 0], rel_prop_pair_idx[:, 1]
        ]
        matching_qualities = rel_matching_scores[
            rel_prop_pair_idx[:, 0], rel_prop_pair_idx[:, 1]
        ]

        return corrsp_gt_rel_idx, matching_qualities, binary_rel_matrixs


class RelationshipOutput:
    """calculate the loss in training and post processing while testing."""

    def __init__(
        self, cfg, relationship_proposal, rel_pred_logits, entities_logits, add_outputs=None
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.rel_pred_logits = rel_pred_logits
        self.entities_logits = entities_logits
        self.rel_proposals = relationship_proposal

        self.empty_weight = torch.ones(self.rel_pred_logits.shape[-1]).to(self.device)
        self.empty_weight[0] = 0.1


        self.gt_rel_labels = None
        if self.rel_proposals[0].has("rel_label"):
            self.gt_rel_labels = torch.cat(
                [each.rel_label for each in self.rel_proposals]
            )

        self.rel_batch_sizes = [len(i) for i in self.rel_proposals]
        self.ent_batch_sizes = [len(i.instances) for i in self.rel_proposals]


        self.rce_logits = None
        self.rce_pred_mat = None

        self._no_instances = False
        if len(self.rel_proposals) == 0:
            self._no_instances = True

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        todo: for relationship_proposal
        """

        pass

        # num_instances = self.gt_classes.numel()
        # pred_classes = self.pred_class_logits.argmax(dim=1)
        # bg_class_ind = self.pred_class_logits.shape[1] - 1
        #
        # fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        # num_fg = fg_inds.nonzero(as_tuple=False).numel()
        # fg_gt_classes = self.gt_classes[fg_inds]
        # fg_pred_classes = pred_classes[fg_inds]
        #
        # num_false_negative = (fg_pred_classes == bg_class_ind).nonzero(as_tuple=False).numel()
        # num_accurate = (pred_classes == self.gt_classes).nonzero(as_tuple=False).numel()
        # fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero(as_tuple=False).numel()
        #
        # storage = get_event_storage()
        # storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        # if num_fg > 0:
        #     storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
        #     storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def predicate_cls_softmax_ce_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        loss_dict = {}

        if self._no_instances:
            loss_dict["loss_rel_cls"] = 0.0 * self.rel_pred_logits.sum()
        else:
            self._log_accuracy()
            if self.rel_pred_logits is not None:
                valid_idx = self.gt_rel_labels >= 0
                loss_dict["loss_rel_cls"] = F.cross_entropy(
                    self.rel_pred_logits[valid_idx],
                    self.gt_rel_labels[valid_idx],
                    # self.empty_weight,
                    reduction="mean"
                )
        return loss_dict

    def losses(self):
        loss_dict = {}
        loss_dict.update(self.predicate_cls_softmax_ce_loss())
        # todo entities loss
        return loss_dict

    def inference(self, max_proposal_pairs=200):

        rel_pred_logits = torch.split(self.rel_pred_logits, self.rel_batch_sizes)

        rel_proposals_predict = []
        for i, (rel_prop, rel_logit) in enumerate(
            zip(self.rel_proposals, rel_pred_logits)
        ):
            rel_prob = F.softmax(rel_logit, -1)

            rel_scores, rel_class = rel_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            ent_score = rel_prop.instances.scores
            pair_indxs = rel_prop.rel_pair_tensor

            triple_scores = (
                rel_scores * ent_score[pair_indxs[:, 0]] * ent_score[pair_indxs[:, 1]]
            )

            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)

            selected_idx = sorting_idx[:max_proposal_pairs]

            if rel_prop.has("rel_label"):
                rel_label = rel_prop.rel_label[selected_idx]

            rel_pred = Relationships(
                instances=rel_prop.instances,
                rel_pair_tensor=pair_indxs[selected_idx],
                pred_rel_classs=rel_class[selected_idx],
                pred_rel_scores=rel_scores[selected_idx],
                pred_rel_dist=rel_prob[selected_idx],
                # pred_rel_trp_scores= triple_scores[:max_proposal_pairs],
                # pred_init_prop_idx=selected_idx,
            )

            if self.rce_pred_mat is not None:
                rel_pair_idx = pair_indxs[selected_idx]
                relness = self.rce_pred_mat[i, rel_pair_idx[:, 0], rel_pair_idx[:, 1], :]
                rel_pred.pred_rel_confidence = relness

            
            rel_proposals_predict.append(rel_pred)

        return rel_proposals_predict



class RelationshipOutputRCE(RelationshipOutput):
    """calculate the loss in training and post processing while testing."""

    def __init__(
        self, cfg, relationship_proposal, rel_pred_logits, entities_logits, add_outputs
    ):
        super(RelationshipOutputRCE, self).__init__(cfg, relationship_proposal, rel_pred_logits, entities_logits,)
        self.rel_conf_loss = RelAwareLoss(cfg)

        rce_logits=add_outputs['conf_est_logits_each_iter']
        rce_pred_mat=add_outputs['relatedness_each_iters']
        if self.rel_proposals[0].has("rel_label_no_mask"):
            gt_rel_labels_rce = torch.cat(
                [each.rel_label_no_mask for each in self.rel_proposals]
            )
            self.gt_rel_labels_rce = torch.zeros_like(self.gt_rel_labels)
            self.gt_rel_labels_rce[gt_rel_labels_rce > 0] = 1

        self.rce_logits = rce_logits
        self.rce_pred_mat = rce_pred_mat

    def RCELoss(self):
        loss_dict = {
            f"rce_loss_{i}": self.rel_conf_loss(rce_loss, self.gt_rel_labels) * 0.3
            for i, rce_loss in enumerate(self.rce_logits)
        }
        return loss_dict

    def losses(self):
        loss_dict = {}
        loss_dict.update(self.predicate_cls_softmax_ce_loss())
        loss_dict.update(self.RCELoss())
        # todo entities loss
        return loss_dict




class RelationshipPredictor(nn.Module):
    def __init__(
        self, cfg, in_channels, num_cls,
    ):
        super(RelationshipPredictor, self).__init__()
        # frequency biases
        #     need the data distribution
        statistics = get_dataset_statistics(cfg)

        self.freq_bias = None
        if cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BIAS:
            self.freq_bias = FrequencyBias(statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.num_cls = num_cls
        self.classifier = build_rel_classifier(
            cfg, "linear", input_dim=in_channels, num_class=self.num_cls + 1
        )
        self.cfg = cfg

    def add_frequency_bias(self, rel_cls_logits, rel_proposals: List[Relationships]):
        if self.freq_bias is None:
            return  rel_cls_logits

        pair_preds = []
        for rel_prop in rel_proposals:

            pair_preds.append(
                torch.stack((rel_prop.instances.pred_classes[rel_prop.rel_pair_tensor[:, 0]],
                             rel_prop.instances.pred_classes[rel_prop.rel_pair_tensor[:, 1]]), dim=1)
            )
        pair_pred = cat(pair_preds, dim=0)


        with torch.no_grad():
            freq_logits = self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())

        rel_cls_logits = rel_cls_logits + freq_logits

        # return  self.freq_bias.index_with_labels(pair_pred.long())

        return  rel_cls_logits

    def forward(self, rel_proposals: list, rel_feat: list):
        rel_logits = self.classifier(torch.cat(rel_feat))

        # add frequency biases
        rel_logits = self.add_frequency_bias(rel_logits, rel_proposals)

        # TODO: entities_logits predictions
        entities_logits = None

        return rel_logits, entities_logits
