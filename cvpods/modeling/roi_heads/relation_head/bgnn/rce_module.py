import math

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.data.datasets.builtin_meta import get_dataset_statistics
from cvpods.modeling.roi_heads.relation_head.rel_classifier import build_rel_classifier, FrequencyBias
from cvpods.modeling.roi_heads.relation_head import obj_edge_vectors, encode_box_info
from cvpods.evaluation.vg_sgg_eval_tools import boxlist_iou
from cvpods.structures.relationship import squeeze_tensor
from .rce_loss import FocalLoss



def gt_rel_proposal_matching(proposals, targets, fg_thres, require_overlap):
    """

    :param proposals:
    :param targets:
    :param fg_thres:
    :param require_overlap:
    :return:
        fg_pair_matrixs the box pairs that both box are matching with gt ground-truth
        prop_relatedness_matrixs: the box pairs that both boxes are matching with ground-truth relationship
    """
    assert targets is not None
    prop_relatedness_matrixs = []
    fg_pair_matrixs = []
    for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
        device = proposal.bbox.device
        tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]

        # IoU matching for object detection results
        ious = boxlist_iou(target, proposal)  # [tgt, prp]
        is_match = ious > fg_thres  # [tgt, prp]
        # one box may match multiple gt boxes here we just mark them as a valid matching if they
        # match any boxes
        locating_match = (ious > fg_thres).nonzero()  # [tgt, prp]
        locating_match_stat = torch.zeros((len(proposal)), device=device)
        locating_match_stat[locating_match[:, 1]] = 1
        proposal.add_field("locating_match", locating_match_stat)

        # Proposal self IoU to filter non-overlap
        prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
        # store the box pairs whose head and tails bbox are all overlapping with the GT boxes
        # does not requires classification results
        if require_overlap:
            fg_boxpair_mat = (
                (prp_self_iou > 0) & (prp_self_iou < 1)
            ).long()  # not self & intersect
        else:
            num_prp = len(proposal)
            # [prp, prp] mark the affinity relation between the det prediction
            fg_boxpair_mat = (
                torch.ones((num_prp, num_prp), device=device).long()
                - torch.eye(num_prp, device=device).long()
            )
        # only select relations between fg proposals
        fg_boxpair_mat[locating_match_stat == 0] = 0
        fg_boxpair_mat[:, locating_match_stat == 0] = 0

        fg_pair_matrixs.append(fg_boxpair_mat)

        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix != 0)

        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = len(proposal)
        # num_tgt_rel, num_prp (matched prp head)
        binary_prp_head = is_match[tgt_head_idxs]
        # num_tgt_rel, num_prp (matched prp tail)
        binary_prp_tail = is_match[tgt_tail_idxs]
        # mark the box pair who overlaps with the gt relation box pairs
        binary_rel_mat = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
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
                binary_rel_mat[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                # binary_rel_mat[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

        prop_relatedness_matrixs.append(binary_rel_mat)

    return fg_pair_matrixs, prop_relatedness_matrixs



def reverse_sigmoid(x):
    new_x = x.clone()
    new_x[x > 0.999] = x[x > 0.999] - (x[x > 0.999].clone().detach() - 0.999)
    new_x[x < 0.001] = x[x < 0.001] + (-x[x < 0.001].clone().detach() + 0.001)
    return torch.log((new_x) / (1 - (new_x)))



# @registry.RELATION_CONFIDENCE_AWARE_MODULES.register("RelAwareRelFeature")
class RelAwareRelFeature(nn.Module):
    def __init__(
        self,
        cfg,
        input_dim,
    ):
        super(RelAwareRelFeature, self).__init__()
        self.cfg = cfg

        self.input_dim = input_dim

        self.predictor_type = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.REL_AWARE_PREDICTOR_TYPE
        )

        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE_REL_FEATURE.WORD_EMBEDDING_FEATURES_DIM
        self.geometry_feat_dim = 128
        self.hidden_dim = 512

        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics["obj_classes"][1:], statistics["rel_classes"][1:]

        obj_embed_vecs = obj_edge_vectors(
            obj_classes, wv_dir=cfg.EXT_KNOWLEDGE.GLOVE_DIR, wv_dim=self.embed_dim
        )

        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        
        self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)

        with torch.no_grad():
            self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_pos_embed = nn.Sequential(
            nn.Linear(9, self.geometry_feat_dim),
            nn.ReLU(),
            nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
        )

        self.visual_features_on = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.VISUAL_FEATURES_ON
        )

        self.proposal_box_feat_extract = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.embed_dim * 2 + self.geometry_feat_dim * 2,
                self.hidden_dim,
            ),
        )

        if self.visual_features_on:
            self.vis_embed = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.input_dim, self.hidden_dim),
            )

            self.proposal_feat_fusion = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )

        self.out_dim = self.num_rel_classes

        self.proposal_relness_cls_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        if self.predictor_type == "hybrid":
            self.fusion_layer = nn.Linear(self.out_dim, 1)

    def forward(
        self,
        visual_feat,
        entities_proposals,
        rel_pair_inds,
    ):

        relness_matrix = []
        relness_logits_batch = []

        if self.visual_features_on:
            if isinstance(visual_feat, tuple):
                visual_feat = torch.cat(visual_feat)
            visual_feat = self.vis_embed(visual_feat.detach())
            visual_feat_split = torch.split(visual_feat, [len(p) for p in rel_pair_inds], dim=0)
        else:
            visual_feat_split = visual_feat
            if not isinstance(visual_feat, tuple):
                visual_feat_split = torch.split(visual_feat, [len(p) for p in rel_pair_inds], dim=0)

        for img_id, (proposal, vis_feats, pair_idx) in enumerate(
            zip(entities_proposals, visual_feat_split, rel_pair_inds)
        ):
            pred_score_dist = proposal.get_field("pred_score_dist").detach()
            pred_labels = proposal.get_field("pred_labels").detach()
            device = proposal.bbox.device
            pred_rel_matrix = torch.zeros(
                (len(proposal), len(proposal)), device=device, dtype=pred_score_dist.dtype
            )
            pos_embed = self.obj_pos_embed(
                encode_box_info([proposal,])
            )
            obj_sem_embed = self.obj_sem_embed(pred_labels.long())
            rel_pair_symb_repre = torch.cat(
                (
                    pos_embed[pair_idx[:, 0]],
                    obj_sem_embed[pair_idx[:, 0]],
                    pos_embed[pair_idx[:, 1]],
                    obj_sem_embed[pair_idx[:, 1]],
                ),
                dim=1,
            )

            prop_pair_geo_feat = self.proposal_box_feat_extract(rel_pair_symb_repre)

            if self.visual_features_on:
                # visual_relness_feat = self.self_att(vis_feats, vis_feats, vis_feats).squeeze(1)
                visual_relness_feat = vis_feats
                rel_prop_repre = self.proposal_feat_fusion(
                    torch.cat((visual_relness_feat, prop_pair_geo_feat), dim=1)
                )
            else:
                rel_prop_repre = prop_pair_geo_feat

            relness_logits = self.proposal_relness_cls_fc(rel_prop_repre)
            try:
                if self.predictor_type == "hybrid":
                    relness_bin_logits = self.fusion_layer(relness_logits)

                    relness_scores = squeeze_tensor(torch.sigmoid(relness_bin_logits))
                    pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores

                    relness_logits = torch.cat((relness_logits, relness_bin_logits), dim=1)
                elif self.predictor_type == "single":
                    relness_scores = squeeze_tensor(torch.sigmoid(relness_logits))
                    if len(relness_scores.shape) == 1:
                        relness_scores = relness_scores.unsqueeze(0)
                    pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores.max(dim=1)[0]
            except (RuntimeError, IndexError):
                print(relness_scores)
                print(relness_logits.shape)

            relness_logits_batch.append(relness_logits)

            relness_matrix.append(pred_rel_matrix)

        return (
            torch.cat(relness_logits_batch),
            relness_matrix,
        )


def make_relation_confidence_aware_module(in_channels):
    return RelAwareRelFeature(in_channels)



def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to 
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1,1], dim=-1)
        x, y = xy.split([1,1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                          w*h/(wid*hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        def attention(q, k, v, d_k, mask=None, dropout=None):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e9)

            scores = F.softmax(scores, dim=-1)

            if dropout is not None:
                scores = dropout(scores)

            output = torch.matmul(scores, v)
            return output

        # calculate attention using function we will define next
        att_result = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = att_result.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output
