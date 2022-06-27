import pickle
from cvpods.modeling.meta_arch.detr import MLP
import math

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.data.datasets.builtin_meta import get_dataset_statistics
from cvpods.modeling.roi_heads.relation_head import obj_edge_vectors, encode_box_info
from cvpods.structures import boxes as box_ops
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_inference import (
    get_matching_scores_entities_aware,
)



class EntitiesIndexingHead(nn.Module):
    def __init__(self, cfg):
        super(EntitiesIndexingHead, self).__init__()

        self.cfg = cfg

        self.vis_feat_input_dim = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL
        self.hidden_dim = self.vis_feat_input_dim * 2

        self.ent_input_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.vis_feat_input_dim, self.hidden_dim)
        )

        self.rel_input_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.vis_feat_input_dim, self.hidden_dim)
        )

    def forward(self, entities_features, rel_features):
        rel_feat2cmp = self.rel_input_fc(rel_features)

        ent_feat2cmp = self.ent_input_fc(entities_features)

        scaling = float(self.hidden_dim) ** -0.5
        attn_output_weights = rel_feat2cmp @ ent_feat2cmp.permute(0, 2, 1) * scaling

        # entities_features = entities_features.transpose(1, 0)
        # rel_features = rel_features.transpose(1, 0)
        # _, attn_output_weights = self.matching_att(
        #     query=rel_feat2cmp, key=ent_feat2cmp, value=ent_feat2cmp
        # )

        return attn_output_weights


class EntitiesIndexingHeadHOTR(nn.Module):
    def __init__(self, cfg):
        super(EntitiesIndexingHead, self).__init__()

        self.cfg = cfg

        self.tau = 0.05

        self.hidden_dim = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL

        self.H_Pointer_embed   = MLP(self.hidden_dim , self.hidden_dim , self.hidden_dim , 3)
        self.O_Pointer_embed   = MLP(self.hidden_dim , self.hidden_dim , self.hidden_dim , 3)



    def forward(self, entities_features, rel_features):

        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(rel_features), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(rel_features), p=2, dim=-1)
        outputs_hidx = [(torch.bmm(H_Pointer_repr, entities_features.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, entities_features.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]

        return outputs_hidx, outputs_oidx


class EntitiesIndexingHeadRuleBased(nn.Module):
    def __init__(self, cfg):
        super(EntitiesIndexingHeadRuleBased, self).__init__()
        self.cfg = cfg
        self.num_ent_class = cfg.MODEL.DETR.NUM_CLASSES

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        ent_box_all = outputs["pred_boxes"]
        if outputs.get("pred_probs") is None:
            ent_prob_all = F.softmax(outputs["pred_logits"], dim=-1)

        obj_dist_mat = []
        sub_dist_mat = []

        for ind in range(len(outputs["pred_rel_obj_logits"])):
            ent_box = (
                box_ops.box_cxcywh_to_xyxy(ent_box_all[ind]) * scale_fct[ind, None, :]
            )
            ent_box_cnter = box_ops.box_xyxy_to_cxcywh(ent_box)[..., :2]

            ent_box_normed = ent_box_all[ind]
            ent_box_cnter_normed = ent_box_normed[..., :2]

            if self.num_ent_class == outputs["pred_rel_obj_logits"][ind].shape[-1]:
                pred_rel_obj_dist = torch.sigmoid(outputs["pred_rel_obj_logits"][ind])
                pred_rel_sub_dist = torch.sigmoid(outputs["pred_rel_sub_logits"][ind])
            else:
                pred_rel_obj_dist = F.softmax(
                    outputs["pred_rel_obj_logits"][ind], dim=-1
                )[..., :-1]
                pred_rel_sub_dist = F.softmax(
                    outputs["pred_rel_sub_logits"][ind], dim=-1
                )[..., :-1]

            pred_rel_obj_box = box_ops.box_cxcywh_to_xyxy(
                outputs["pred_rel_obj_box"][ind]
            )
            pred_rel_obj_box = torch.squeeze(pred_rel_obj_box * scale_fct[ind, None, :])

            pred_rel_sub_box = box_ops.box_cxcywh_to_xyxy(
                outputs["pred_rel_sub_box"][ind]
            )
            pred_rel_sub_box = torch.squeeze(pred_rel_sub_box * scale_fct[ind, None, :])

            # print((pred_rel_sub_box[:, 2:] < pred_rel_sub_box[:, :2]).sum())
            # print((pred_rel_obj_box[:, 2:] < pred_rel_obj_box[:, :2]).sum())
            # print((ent_box[:, 2:] <= ent_box[:, :2]).sum())
            if not (pred_rel_sub_box[:, 2:] >= pred_rel_sub_box[:, :2]).all():
                with open("box_tmp.pkl", 'wb') as f:
                    pickle.dump((pred_rel_sub_box, pred_rel_obj_box, ent_box), )
                

            rel_vec_flat_normed = outputs["pred_rel_vec"][ind]
            rel_vec_flat = rel_vec_flat_normed * scale_fct[ind, None, :]

            ent_prob = ent_prob_all[ind]

            if self.num_ent_class != ent_prob.shape[-1]:
                ent_prob = ent_prob[..., :-1]
            ent_score = ent_prob.max(-1)[0]

            (dist_s, dist_o, match_cost_details) = get_matching_scores_entities_aware(
                s_cetr=ent_box_cnter,
                o_cetr=ent_box_cnter,
                s_scores=ent_score,
                o_scores=ent_score,
                rel_vec=rel_vec_flat,
                s_cetr_normed=ent_box_cnter_normed,
                o_cetr_normed=ent_box_cnter_normed,
                rel_vec_normed=rel_vec_flat_normed,
                ent_box=ent_box,
                ent_box_normed=box_ops.box_cxcywh_to_xyxy(ent_box_normed),
                s_dist=ent_prob,
                o_dist=ent_prob,
                rel_ent_s_box=pred_rel_sub_box,
                rel_ent_o_box=pred_rel_obj_box,
                rel_s_dist=pred_rel_sub_dist,
                rel_o_dist=pred_rel_obj_dist,
                normed_rel_vec_dist=self.cfg.MODEL.REL_DETR.NORMED_REL_VEC_DIST,
            )

            for k,v in match_cost_details.items():
                if torch.isnan(v).any():
                    print(k)
                if torch.isinf(v).any():
                    print(k)
            # if self.training:
            #     # suppress the low quality matching
            #     dist_s[match_cost_details["match_sub_giou"] < 0.7] *= 0.1
            #     dist_o[match_cost_details["match_obj_giou"] < 0.7] *= 0.1

            obj_dist_mat.append(dist_o)
            sub_dist_mat.append(dist_s)

        return torch.stack(sub_dist_mat).detach(), torch.stack(obj_dist_mat).detach()


class EntitiesIndexingHeadPredAtt(nn.Module):
    def __init__(self, cfg):
        super(EntitiesIndexingHeadPredAtt, self).__init__()
        self.cfg = cfg

        self.hidden_dim = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL

        self.cls_num = cfg.MODEL.DETR.NUM_CLASSES

        self.rel_geo_info_encode = nn.Sequential(nn.Linear(4 + 4, self.hidden_dim))

        self.ent_geo_info_encode = nn.Sequential(nn.Linear(4, self.hidden_dim))

        self.cls_info_encode_fc = nn.Sequential(
            nn.BatchNorm1d(self.cls_num),
            nn.Linear(self.cls_num, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.ent_sub_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.ent_obj_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.rel_obj_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.rel_sub_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def cls_info_encode(self, pred_dist):
        bs, num_q, cls_num = pred_dist.shape
        return self.cls_info_encode_fc(pred_dist.reshape(-1, cls_num)).reshape(
            bs, num_q, self.hidden_dim
        )

    def forward(self, outputs):
        pred_rel_vec = outputs["pred_rel_vec"]
        rel_obj_ent = []
        rel_sub_ent = []
        ent = []

        for ind, ent_det in enumerate(pred_rel_vec):
            ent_box = outputs["pred_boxes"][ind]
            num_ent = len(ent_box)
            ent_box_cnter = ent_box[..., :2]

            pred_rel_obj_dist = F.softmax(outputs["pred_rel_obj_logits"][ind], dim=-1)[
                ..., :-1
            ]
            pred_rel_obj_box = outputs["pred_rel_obj_box"][ind]
            pred_rel_obj_box = torch.squeeze(pred_rel_obj_box)

            pred_rel_sub_dist = F.softmax(outputs["pred_rel_sub_logits"][ind], dim=-1)[
                ..., :-1
            ]
            pred_rel_sub_box = outputs["pred_rel_sub_box"][ind]
            pred_rel_sub_box = torch.squeeze(pred_rel_sub_box)
            ent_prob = F.softmax(outputs["pred_logits"][ind], dim=-1)[..., :-1]

            rel_vec_flat = outputs["pred_rel_vec"][ind]

            rel_obj_ent.append(
                torch.cat((pred_rel_obj_dist, pred_rel_obj_box, rel_vec_flat), -1)
            )
            rel_sub_ent.append(
                torch.cat((pred_rel_sub_dist, pred_rel_sub_box, rel_vec_flat), -1)
            )
            ent.append(torch.cat((ent_prob, ent_box), -1))

        # todo word embedding

        rel_obj_ent_input = torch.stack(rel_obj_ent)
        rel_sub_ent_input = torch.stack(rel_sub_ent)
        ent_input = torch.stack(ent)

        rel_feat2cmp_obj = self.rel_obj_input_fc(
            torch.cat(
                (
                    self.rel_geo_info_encode(rel_obj_ent_input[:, :, -8:]),
                    self.cls_info_encode(rel_obj_ent_input[:, :, :-8]),
                ),
                dim=-1,
            )
        )

        rel_feat2cmp_sub = self.rel_sub_input_fc(
            torch.cat(
                (
                    self.rel_geo_info_encode(rel_sub_ent_input[:, :, -8:]),
                    self.cls_info_encode(rel_sub_ent_input[:, :, :-8]),
                ),
                dim=-1,
            )
        )

        ent_feat2cmp_obj = self.ent_obj_input_fc(
            torch.cat(
                (
                    self.ent_geo_info_encode(ent_input[:, :, -4:]),
                    self.cls_info_encode(ent_input[:, :, :-4]),
                ),
                dim=-1,
            )
        )
        ent_feat2cmp_sub = self.ent_sub_input_fc(
            torch.cat(
                (
                    self.ent_geo_info_encode(ent_input[:, :, -4:]),
                    self.cls_info_encode(ent_input[:, :, :-4]),
                ),
                dim=-1,
            )
        )

        scaling = float(self.hidden_dim) ** -0.5
        obj_attn_output = rel_feat2cmp_obj @ ent_feat2cmp_obj.permute(0, 2, 1) * scaling
        sub_attn_output = rel_feat2cmp_sub @ ent_feat2cmp_sub.permute(0, 2, 1) * scaling

        return obj_attn_output, sub_attn_output




def minmax_norm(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 0.02)


def gen_sineembed_for_position(pos_tensor, h_dim=256):
    """[summary]

    Args:
        pos_tensor ([Tensor]):  [num_queries, batch_size, 2]

    Returns:
        [type]: [num_queries, batch_size, hidden_dim]
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    h_dim /= 2
    dim_t = torch.arange(h_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / h_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos
