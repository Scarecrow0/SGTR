import time
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou

from cvpods.structures import boxes as box_ops
from cvpods.structures.boxes import box_cxcywh_to_xyxy
from cvpods.utils.metrics.cosine_sim import cosine_similarity


def get_matching_scores_entities_aware(
        s_cetr,
        o_cetr,
        s_scores,
        o_scores,
        rel_vec,
        s_cetr_normed,
        o_cetr_normed,
        rel_vec_normed,
        ent_box,
        ent_box_normed,
        s_dist,
        o_dist,
        rel_ent_s_box,
        rel_ent_o_box,
        rel_s_dist,
        rel_o_dist,
        normed_rel_vec_dist=False,
):
    """

    Args:
        s_cetr: image size normed
        o_cetr: image size normed
        s_scores:
        o_scores:
        rel_vec: image size normed
        ent_box:
        s_dist:
        o_dist:
        rel_ent_s_box:
        rel_ent_o_box:
        rel_s_dist:
        rel_o_dist:

    Returns:

    """

    def rev_vec_abs_dist(rel_vec, s_cetr, o_cetr):
        rel_s_centr = rel_vec[..., :2].unsqueeze(-1).repeat(1, 1, s_cetr.shape[0])
        rel_o_centr = rel_vec[..., 2:].unsqueeze(-1).repeat(1, 1, o_cetr.shape[0])
        s_cetr = s_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        o_cetr = o_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)

        dist_s_x = abs(rel_s_centr[..., 0, :] - s_cetr[..., 0])
        dist_s_y = abs(rel_s_centr[..., 1, :] - s_cetr[..., 1])
        dist_o_x = abs(rel_o_centr[..., 0, :] - o_cetr[..., 0])
        dist_o_y = abs(rel_o_centr[..., 1, :] - o_cetr[..., 1])
        return dist_s_x, dist_s_y, dist_o_x, dist_o_y

    if not normed_rel_vec_dist:
        (dist_s_x, dist_s_y, dist_o_x, dist_o_y) = rev_vec_abs_dist(
            rel_vec, s_cetr, o_cetr
        )
    else:
        (dist_s_x, dist_s_y, dist_o_x, dist_o_y) = rev_vec_abs_dist(
            rel_vec_normed, s_cetr_normed, o_cetr_normed
        )
    match_rel_vec_sub = 1 / (dist_s_x + dist_s_y + 1)
    match_rel_vec_obj = 1 / (dist_o_x + dist_o_y + 1)

    s_scores = s_scores ** 0.6
    o_scores = o_scores ** 0.6
    s_scores = s_scores.repeat(rel_vec.shape[0], 1)
    o_scores = o_scores.repeat(rel_vec.shape[0], 1)
    
    # match_vec_n_conf_sub = s_scores
    # match_vec_n_conf_obj = o_scores
    match_vec_n_conf_sub = s_scores * match_rel_vec_sub
    match_vec_n_conf_obj = o_scores * match_rel_vec_obj

    match_cost_details = {
        "match_rel_vec_sub": match_rel_vec_sub,
        "match_rel_vec_obj": match_rel_vec_obj,
        "match_sub_conf": s_scores,
        "match_obj_conf": o_scores,
        "match_vec_n_conf_sub": match_vec_n_conf_sub,
        "match_vec_n_conf_obj": match_vec_n_conf_obj,
    }

    match_rel_sub_cls = cosine_similarity(rel_s_dist, s_dist)
    match_rel_obj_cls = cosine_similarity(rel_o_dist, o_dist)
    
    # match_rel_sub_cls = torch.squeeze(torch.cdist(rel_s_dist.unsqueeze(0), s_dist.unsqueeze(0), p=2)) / s_dist.shape[-1]
    # match_rel_obj_cls = torch.squeeze(torch.cdist(rel_o_dist.unsqueeze(0), o_dist.unsqueeze(0), p=2)) / s_dist.shape[-1]

    match_rel_sub_cls = match_rel_sub_cls ** 0.6
    match_rel_obj_cls = match_rel_obj_cls ** 0.6
    match_cost_details["match_rel_sub_cls"] = match_rel_sub_cls
    match_cost_details["match_rel_obj_cls"] = match_rel_obj_cls


    match_sub_giou = torch.clip(generalized_box_iou(rel_ent_s_box, ent_box), 0)
    match_obj_giou = torch.clip(generalized_box_iou(rel_ent_o_box, ent_box), 0)

    match_cost_details["match_sub_giou"] = match_sub_giou
    match_cost_details["match_obj_giou"] = match_obj_giou

    match_scr_sub = match_rel_sub_cls * match_sub_giou * match_vec_n_conf_sub
    match_scr_obj = match_rel_obj_cls * match_obj_giou * match_vec_n_conf_obj

    # match_scr_sub = minmax_norm(match_scr_sub)
    # match_scr_obj = minmax_norm(match_scr_obj)

    match_cost_details["match_scr_sub"] = match_scr_sub
    match_cost_details["match_scr_obj"] = match_scr_obj

    return match_scr_sub, match_scr_obj, match_cost_details


def rel_vec_matching_scores(
        s_cetr, o_cetr, s_scores, o_scores, rel_vec,
):
    rel_s_centr = rel_vec[..., :2].unsqueeze(-1).repeat(1, 1, s_cetr.shape[0])
    rel_o_centr = rel_vec[..., 2:].unsqueeze(-1).repeat(1, 1, o_cetr.shape[0])
    s_cetr = s_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
    s_scores = s_scores.repeat(rel_vec.shape[0], 1)
    o_cetr = o_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
    o_scores = o_scores.repeat(rel_vec.shape[0], 1)
    dist_s_x = abs(rel_s_centr[..., 0, :] - s_cetr[..., 0])
    dist_s_y = abs(rel_s_centr[..., 1, :] - s_cetr[..., 1])
    dist_o_x = abs(rel_o_centr[..., 0, :] - o_cetr[..., 0])
    dist_o_y = abs(rel_o_centr[..., 1, :] - o_cetr[..., 1])
    dist_s = (1.0 / (dist_s_x + 1.0)) * (1.0 / (dist_s_y + 1.0))
    dist_o = (1.0 / (dist_o_x + 1.0)) * (1.0 / (dist_o_y + 1.0))
    dist_s = dist_s * s_scores
    dist_o = dist_o * o_scores
    return dist_s, dist_o


class RelPostProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_ent_class = cfg.MODEL.DETR.NUM_CLASSES
        self.overlap_thres = cfg.MODEL.REL_DETR.OVERLAP_THRES
        self.ent_ranking = False

    @torch.no_grad()
    def forward(
            self,
            outputs,
            det_res,
            target_sizes,
            max_proposal_pairs=300,
            init_max_match_ent=None,
            post_proc_filtering=True,
    ):

        pred_rel_logits = outputs["pred_rel_logits"]
        pred_rel_vec = outputs["pred_rel_vec"]

        device = pred_rel_vec.device

        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            pred_rel_probs = torch.sigmoid(pred_rel_logits)
        else:
            pred_rel_probs = torch.softmax(pred_rel_logits, -1)

        # [batch_size, query_num, ]

        pred_rel_conf_score = None

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_rel_vec = pred_rel_vec * scale_fct[:, None, :]

        rel_proposals_predict = []
        init_rel_proposals_predict = []

        for batch_ind, ent_det in enumerate(det_res):

            ent_score = ent_det["scores"]
            ent_label = ent_det["labels"]
            ent_box = ent_det["boxes"]
            ent_box_normed = ent_det["boxes_norm"]

            ent_box_cnter_normed = ent_box_normed[..., :2]
            rel_vec_flat_normed = outputs["pred_rel_vec"][batch_ind]

            ent_box_cnter = box_ops.box_xyxy_to_cxcywh(ent_box)[..., :2]

            rel_vec_flat = pred_rel_vec[batch_ind]
            if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
                pred_rel_obj_box = (
                        box_ops.box_cxcywh_to_xyxy(outputs["pred_rel_obj_box"][batch_ind])
                        * scale_fct[batch_ind, None, :]
                )
                pred_rel_obj_box = torch.squeeze(pred_rel_obj_box)

                pred_rel_sub_box = (
                        box_ops.box_cxcywh_to_xyxy(outputs["pred_rel_sub_box"][batch_ind])
                        * scale_fct[batch_ind, None, :]
                )
                pred_rel_sub_box = torch.squeeze(pred_rel_sub_box)

                pred_rel_sub_dist = F.softmax(
                    outputs["pred_rel_sub_logits"][batch_ind], dim=-1
                )[..., :-1]
                pred_rel_sub_score, pred_rel_sub_label = torch.max(
                    pred_rel_sub_dist, dim=-1
                )

                pred_rel_obj_dist = F.softmax(
                    outputs["pred_rel_obj_logits"][batch_ind], dim=-1
                )[..., :-1]
                pred_rel_obj_score, pred_rel_obj_label = torch.max(
                    pred_rel_obj_dist, dim=-1
                )

                if self.num_ent_class == ent_det["prob"].shape[-1]:
                    # in case done remove the bg class
                    ent_prob = ent_det["prob"]
                else:
                    ent_prob = ent_det["prob"][..., :-1]

                match_cost_details = {}
                ent_num = len(ent_prob)
                rel_num = len(pred_rel_sub_box)

                match_scr_sub = torch.zeros((rel_num, ent_num), device=device)
                match_scr_obj = torch.zeros((rel_num, ent_num), device=device)
                match_cost_details = {}

                if ("rule_base" in self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_TYPE_INFERENCE):
                    (
                        match_scr_sub_r,
                        match_scr_obj_r,
                        match_cost_details_r,
                    ) = get_matching_scores_entities_aware(
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

                    match_scr_sub = match_scr_sub + match_scr_sub_r
                    match_scr_obj = match_scr_obj + match_scr_obj_r
                    match_cost_details.update(match_cost_details_r)
                if ("feat_att" in self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_TYPE_INFERENCE):
                    match_scr_sub_f = torch.sigmoid(
                        outputs["sub_entities_indexing"][batch_ind]
                    )
                    match_scr_obj_f = torch.sigmoid(
                        outputs["obj_entities_indexing"][batch_ind]
                    )

                    match_cost_details.update(
                        {
                            "sub_entities_indexing": match_scr_sub_f,
                            "obj_entities_indexing": match_scr_obj_f,
                        }
                    )
                    match_scr_sub = match_scr_sub + match_scr_sub_f
                    match_scr_obj = match_scr_obj + match_scr_obj_f

                if ("rel_vec" in self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_TYPE_INFERENCE):
                    match_scr_sub_v, match_scr_obj_v = self.get_matching_scores(
                        ent_box_cnter, ent_box_cnter, ent_score, ent_score, rel_vec_flat
                    )
                    match_scr_sub = match_scr_sub + match_scr_sub_v
                    match_scr_obj = match_scr_obj + match_scr_obj_v

            else:

                match_scr_sub, match_scr_obj = rel_vec_matching_scores(
                    ent_box_cnter, ent_box_cnter, ent_score, ent_score, rel_vec_flat
                )
            # num_rel_queries, num_ent_queries
            # one relationship prediction may matching with multiple entities
            if init_max_match_ent is None:
                init_max_match_ent = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING

            max_match_ent = (
                init_max_match_ent
                if match_scr_sub.shape[-1] > init_max_match_ent
                else match_scr_sub.shape[-1]
            )
            rel_match_sub_scores, rel_match_sub_ids = torch.topk(
                match_scr_sub, max_match_ent, dim=-1
            )

            # num_rel_queries; num_rel_queries
            max_match_ent = (
                init_max_match_ent
                if match_scr_obj.shape[-1] > init_max_match_ent
                else match_scr_obj.shape[-1]
            )
            rel_match_obj_scores, rel_match_obj_ids = torch.topk(
                match_scr_obj, max_match_ent, dim=-1
            )

            # the category for each prob in predictions
            # pred_num_per_edge = pred_rel_prob.shape[-1]  # (num_categories)
            if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                pred_rel_prob = pred_rel_probs[batch_ind]
                num_q, cls_num = pred_rel_prob.shape

                pred_num_per_edge = self.cfg.MODEL.REL_DETR.NUM_PRED_EDGES

                topk = num_q * pred_num_per_edge  # num of query * pred_num_per_edge

                topk_values_all, topk_indexes_all = torch.sort(
                    pred_rel_prob.reshape(-1), dim=-1, descending=True
                )  # num_query * num_cls

                pred_rel_prob = topk_values_all[
                                :topk
                                ]  # scores for each relationship predictions
                # (num of query * pred_num_per_edge)
                total_pred_idx = torch.div(topk_indexes_all[:topk], cls_num, rounding_mode='trunc')
                pred_rel_labels = topk_indexes_all[:topk] % cls_num
                pred_rel_labels += 1

                # =>  (num_queries * num_pred_rel,  num_group_entities)
                rel_match_sub_ids = rel_match_sub_ids[total_pred_idx]
                rel_match_obj_ids = rel_match_obj_ids[total_pred_idx]

                total_pred_idx = (
                    total_pred_idx.contiguous()
                        .unsqueeze(1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_prob = (
                    pred_rel_prob.reshape(-1, 1).repeat(1, max_match_ent).view(-1)
                )
                pred_rel_labels = (
                    pred_rel_labels.reshape(-1, 1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

            else:
                pred_rel_prob = pred_rel_probs[batch_ind][:, 1:]

                num_rel_queries = pred_rel_prob.shape[0]

                pred_num_per_edge = 1

                pred_rel_prob, pred_rel_labels = pred_rel_prob.sort(-1, descending=True)
                pred_rel_labels += 1

                pred_rel_labels = pred_rel_labels[:, :pred_num_per_edge]
                pred_rel_prob = pred_rel_prob[:, :pred_num_per_edge]

                # (num_queries * num_categories)
                # => (num_queries * num_categories, 1)
                # =>  (num_queries * num_pred_rel * num_group_entities)
                pred_rel_prob = pred_rel_prob.reshape(-1, 1)
                pred_rel_prob = pred_rel_prob.repeat(1, max_match_ent).view(-1)

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_labels = pred_rel_labels.reshape(-1, 1)
                pred_rel_labels = (
                    pred_rel_labels.repeat(1, max_match_ent).view(-1).unsqueeze(1)
                )

                total_pred_idx = (
                    torch.arange(num_rel_queries)
                        .unsqueeze(1)
                        .repeat(1, pred_num_per_edge)
                )
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                total_pred_idx = total_pred_idx.reshape(-1, 1)
                total_pred_idx = (
                    total_pred_idx.repeat(1, max_match_ent)
                        .view(-1)
                        .contiguous()
                        .unsqueeze(1)
                )

            rel_match_sub_ids_flat = rel_match_sub_ids.view(-1).contiguous()
            rel_match_obj_ids_flat = rel_match_obj_ids.view(-1).contiguous()

            rel_trp_scores = (
                    pred_rel_prob
                    * ent_score[rel_match_sub_ids_flat]
                    * ent_score[rel_match_obj_ids_flat]
            )  # (num_queries,  1)

            if (
                    self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_ENTITIES_INDEXING_RANKING
            ):
                rel_match_sub_scores_flat = rel_match_sub_scores.view(-1).contiguous()
                rel_match_obj_scores_flat = rel_match_obj_scores.view(-1).contiguous()
                matching_score = (
                        rel_match_sub_scores_flat * rel_match_obj_scores_flat
                ).view(-1)
                rel_trp_scores = rel_trp_scores * matching_score  # (num_queries,  1)


            # (num_queries, num_categories)
            # =>  (num_queries * num_pred_rel * num_group_entities, 1)
            pred_rel_pred_score = pred_rel_prob.contiguous().unsqueeze(1)
            rel_trp_scores = rel_trp_scores.unsqueeze(1)

            rel_match_sub_ids2cat = rel_match_sub_ids_flat.unsqueeze(1)
            rel_match_obj_ids2cat = rel_match_obj_ids_flat.unsqueeze(1)

            SUB_IDX = 0
            OBJ_IDX = 1
            REL_LABEL = 2
            REL_TRP_SCR = 3
            REL_PRED_SCR = 4
            INIT_PROP_IDX = 5

            pred_rel_triplet = torch.cat(
                (
                    rel_match_sub_ids2cat.long().to(device),
                    rel_match_obj_ids2cat.long().to(device),
                    pred_rel_labels.long().to(device),
                    rel_trp_scores.to(device),
                    pred_rel_pred_score.to(device),
                    total_pred_idx.to(device),
                ),
                1,
            )

            if pred_rel_conf_score is not None:
                rel_conf_score_rept = (
                    pred_rel_conf_score[batch_ind]
                        .unsqueeze(-1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

                pred_rel_triplet = torch.cat(
                    (pred_rel_triplet, rel_conf_score_rept,), dim=1
                )

            pred_rel_triplet = pred_rel_triplet.to("cpu")
            init_pred_rel_triplet = pred_rel_triplet.to("cpu")
            # init_pred_rel_triplet = pred_rel_triplet.to(device)

            # removed the self connection
            self_iou = generalized_box_iou(
                box_cxcywh_to_xyxy(ent_box), box_cxcywh_to_xyxy(ent_box)
            )
            non_self_conn_idx = (rel_match_obj_ids_flat - rel_match_sub_ids_flat) != 0
            sub_obj_iou_check = self_iou[pred_rel_triplet[:, SUB_IDX].long(), pred_rel_triplet[:, OBJ_IDX].long()] < 0.95
            non_self_conn_idx = torch.logical_and(non_self_conn_idx, sub_obj_iou_check)

            # first stage filtering
            if post_proc_filtering:
                pred_rel_triplet = init_pred_rel_triplet[non_self_conn_idx]

                _, top_rel_idx = torch.sort(
                    pred_rel_triplet[:, REL_TRP_SCR], descending=True
                )
                top_rel_idx = top_rel_idx[: self.cfg.MODEL.REL_DETR.NUM_MAX_REL_PRED]

                pred_rel_triplet = pred_rel_triplet[top_rel_idx]

            ent_label = ent_det["labels"].detach().cpu()
            ent_box = ent_det["boxes"]
            ent_score = ent_det["scores"].detach().cpu()

            self_iou = self_iou.detach().cpu()

            sub_idx = pred_rel_triplet[:, SUB_IDX].long().detach().cpu()
            obj_idx = pred_rel_triplet[:, OBJ_IDX].long().detach().cpu()

            rel_pred_label = (
                pred_rel_triplet[:, REL_LABEL].long().detach().cpu()
            )
            rel_pred_score = pred_rel_triplet[:, REL_TRP_SCR].detach().cpu()

            def rel_prediction_filtering(pred_rel_triplet):
                """

                Args:
                    pred_idx_set:
                    new_come_pred_idx:

                Returns:

                """
                pred_idx_set = []
                for new_come_pred_idx in range(len(pred_rel_triplet)):

                    new_come_sub_idx = sub_idx[new_come_pred_idx]
                    new_come_obj_idx = obj_idx[new_come_pred_idx]

                    new_come_sub_label = ent_label[new_come_sub_idx]
                    new_come_obj_label = ent_label[new_come_obj_idx]

                    new_come_pred_label = rel_pred_label[new_come_pred_idx]
                    new_come_pred_score = rel_pred_score[new_come_pred_idx] * ent_score[new_come_sub_idx] * ent_score[new_come_obj_idx]

                    pred_idx = torch.Tensor(pred_idx_set).long()
                    curr_sub_idx = sub_idx[pred_idx]
                    curr_obj_idx = obj_idx[pred_idx]

                    curr_sub_label = ent_label[curr_sub_idx]
                    curr_obj_label = ent_label[curr_obj_idx]

                    curr_pred_label = rel_pred_label[pred_idx]
                    curr_pred_score = rel_pred_score[pred_idx] * ent_score[curr_sub_idx] * ent_score[curr_obj_idx]

                    entities_indx_match = torch.logical_and(
                        curr_sub_idx == new_come_sub_idx,
                        curr_obj_idx == new_come_obj_idx
                    )

                    new_come_sub_idx = (torch.ones(len(pred_idx)) * new_come_sub_idx).long()
                    new_come_obj_idx = (torch.ones(len(pred_idx)) * new_come_obj_idx).long()

                    sub_iou = self_iou[new_come_sub_idx, curr_sub_idx]
                    obj_iou = self_iou[new_come_obj_idx, curr_obj_idx]

                    entities_pred_match = torch.logical_and(
                            torch.logical_and(sub_iou > self.overlap_thres, obj_iou > self.overlap_thres),
                            torch.logical_and(curr_sub_label == new_come_sub_label, curr_obj_label == new_come_obj_label)
                    )
                    entity_match = torch.logical_or(entities_pred_match, entities_indx_match)

                    if entity_match.any():
                        pred_match = curr_pred_label == new_come_pred_label
                        rel_match = torch.logical_and(entity_match, pred_match)

                        if rel_match.any():
                            is_existed = new_come_pred_score < curr_pred_score[rel_match]
                            if not is_existed.any():
                                pred_idx_set.append(new_come_pred_idx)
                        else:
                            pred_idx_set.append(new_come_pred_idx)
                        
                    else:
                        pred_idx_set.append(new_come_pred_idx)

                pred_idx_set = torch.Tensor(pred_idx_set).long().to(device)
                bin_mask = torch.zeros((pred_rel_triplet.shape[0]), dtype=torch.bool).to(
                    device
                )
                bin_mask[pred_idx_set] = True
                pred_rel_triplet_selected = pred_rel_triplet[bin_mask]

                return pred_rel_triplet_selected

            # start = time.perf_counter()
            if post_proc_filtering and self.overlap_thres > 0:

                pred_rel_triplet_selected = rel_prediction_filtering(
                    pred_rel_triplet
                )
                # lower score thres
                # low_score_filter_idx = pred_rel_triplet_selected[:, REL_TRP_SCR] > 0.001
                # pred_rel_triplet_selected = pred_rel_triplet_selected[low_score_filter_idx]
                # print(time.perf_counter() - start)
            else:
                pred_rel_triplet_selected = pred_rel_triplet
                non_max_suppressed_idx = None


            # top K selection
            _, top_rel_idx = torch.sort(
                pred_rel_triplet_selected[:, REL_TRP_SCR], descending=True
            )
            pred_rel_triplet_selected = pred_rel_triplet_selected[
                top_rel_idx[:max_proposal_pairs]
            ]

            def res2dict(pred_rel_triplet):
                ret = {
                    "rel_trp": pred_rel_triplet[:, :3].long(),
                    "rel_pred_label": pred_rel_triplet[:, REL_LABEL].long(),
                    "rel_score": pred_rel_triplet[:, REL_PRED_SCR],
                    "rel_trp_score": pred_rel_triplet[:, REL_TRP_SCR],
                    "pred_prob_dist": pred_rel_probs[batch_ind][
                        pred_rel_triplet[:, INIT_PROP_IDX].long()
                    ],
                    "rel_vec": rel_vec_flat[pred_rel_triplet[:, INIT_PROP_IDX].long()],
                    "init_prop_indx": pred_rel_triplet[:, INIT_PROP_IDX].long(),
                }

                return ret

            init_pred_dict = res2dict(init_pred_rel_triplet)

            init_rel_proposals_predict.append(init_pred_dict)

            rel_proposals_predict.append(res2dict(pred_rel_triplet_selected))

        return rel_proposals_predict, init_rel_proposals_predict


class RelPostProcessSingleBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.OVERLAP_THRES = cfg.MODEL.REL_DETR.OVERLAP_THRES
        assert self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED
        self.num_ent_class = cfg.MODEL.DETR.NUM_CLASSES

    @torch.no_grad()
    def forward(
            self,
            outputs,
            det_res,
            target_sizes,
            max_proposal_pairs=300,
            init_max_match_ent=None,
            post_proc_filtering=True,
    ):

        pred_rel_logits = outputs["pred_rel_logits"]
        pred_rel_vec = outputs["pred_rel_vec"]

        device = pred_rel_vec.device

        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            pred_rel_probs = torch.sigmoid(pred_rel_logits)
        else:
            pred_rel_probs = torch.softmax(pred_rel_logits, -1)

        # [batch_size, query_num, ]

        pred_rel_conf_score = None

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_rel_vec = pred_rel_vec * scale_fct[:, None, :]

        rel_proposals_predict = []
        init_rel_proposals_predict = []

        for batch_ind, ent_det in enumerate(det_res):

            # the category for each prob in predictions
            # pred_num_per_edge = pred_rel_prob.shape[-1]  # (num_categories)
            pred_num_per_edge = self.cfg.MODEL.REL_DETR.NUM_PRED_EDGES
            max_match_ent = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING
            # max_match_ent = 1
            # pred_num_per_edge = 1

            if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                pred_rel_prob = pred_rel_probs[batch_ind]
                num_q, cls_num = pred_rel_prob.shape

                topk = num_q * pred_num_per_edge  # num of query * pred_num_per_edge

                topk_values_all, topk_indexes_all = torch.sort(
                    pred_rel_prob.reshape(-1), dim=-1, descending=True
                )  # num_query * num_cls

                pred_rel_prob = topk_values_all[
                                :topk
                                ]  # scores for each relationship predictions
                # (num of query * pred_num_per_edge)
                total_pred_idx = topk_indexes_all[:topk] // cls_num
                pred_rel_labels = topk_indexes_all[:topk] % cls_num
                pred_rel_labels += 1

                total_pred_idx = total_pred_idx.contiguous()

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_prob = (
                    pred_rel_prob.reshape(-1, 1).repeat(1, max_match_ent).view(-1)
                )
                pred_rel_labels = (
                    pred_rel_labels.reshape(-1, 1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

            else:
                pred_rel_prob = pred_rel_probs[batch_ind][:, 1:]

                num_rel_queries = pred_rel_prob.shape[0]

                pred_rel_prob, pred_rel_labels = pred_rel_prob.sort(-1, descending=True)
                pred_rel_labels += 1

                pred_rel_labels = pred_rel_labels[:, :pred_num_per_edge]
                pred_rel_prob = pred_rel_prob[:, :pred_num_per_edge]

                # (num_queries * num_categories)
                # => (num_queries * num_categories, 1)
                # =>  (num_queries * num_pred_rel * num_group_entities)
                pred_rel_prob = pred_rel_prob.reshape(-1, 1)
                pred_rel_prob = pred_rel_prob.repeat(1, max_match_ent).view(-1)

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_labels = pred_rel_labels.reshape(-1, 1)
                pred_rel_labels = (
                    pred_rel_labels.repeat(1, max_match_ent).view(-1).unsqueeze(1)
                )

                total_pred_idx = (
                    torch.arange(num_rel_queries)
                        .unsqueeze(1)
                        .repeat(1, pred_num_per_edge)
                )
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                total_pred_idx = total_pred_idx.reshape(-1, 1)
                total_pred_idx = (
                    total_pred_idx.repeat(1, max_match_ent).view(-1).contiguous()
                )

            # rel branch entities outputs
            # in case to remove the bg class
            if (
                    self.num_ent_class
                    == outputs["pred_rel_obj_logits"][batch_ind].shape[-1]
            ):
                pred_rel_obj_dist = torch.sigmoid(
                    outputs["pred_rel_obj_logits"][batch_ind]
                )
                pred_rel_sub_dist = torch.sigmoid(
                    outputs["pred_rel_sub_logits"][batch_ind]
                )
            else:
                pred_rel_obj_dist = F.softmax(
                    outputs["pred_rel_obj_logits"][batch_ind], dim=-1
                )[..., :-1]
                pred_rel_sub_dist = F.softmax(
                    outputs["pred_rel_sub_logits"][batch_ind], dim=-1
                )[..., :-1]

            def extract_rel_branch_pred(
                    pred_rel_box,
                    batch_ind,
                    max_match_ent,
                    total_pred_idx,
                    pred_rel_ent_dist,
            ):
                pred_rel_box = (
                        box_ops.box_cxcywh_to_xyxy(pred_rel_box)
                        * scale_fct[batch_ind, None, :]
                )
                pred_rel_box = torch.squeeze(pred_rel_box)[total_pred_idx]

                # num_rel_queries; num_rel_queries
                rel_ent_scores, rel_ent_label = torch.topk(
                    pred_rel_ent_dist[total_pred_idx], max_match_ent, dim=-1
                )

                pred_rel_ent_box_expand = pred_rel_box.unsqueeze(1)
                pred_rel_ent_box_expand = pred_rel_ent_box_expand.repeat(
                    1, max_match_ent, 1
                ).flatten(0, 1)

                pred_rel_ent_dist_expand = pred_rel_ent_dist.unsqueeze(1)
                pred_rel_ent_dist_expand = pred_rel_ent_dist_expand.repeat(
                    1, max_match_ent, 1
                ).flatten(0, 1)

                return (
                    pred_rel_box,
                    pred_rel_ent_box_expand,
                    pred_rel_ent_dist_expand,
                    rel_ent_label.view(-1).contiguous(),
                    rel_ent_scores.view(-1).contiguous(),
                )

            (
                pred_rel_obj_box,
                pred_rel_obj_box_expand,
                pred_rel_obj_dist_expand,
                rel_obj_ent_label_flat,
                rel_obj_ent_scores_flat,
            ) = extract_rel_branch_pred(
                outputs["pred_rel_obj_box"][batch_ind],
                batch_ind, max_match_ent, total_pred_idx, pred_rel_obj_dist,
            )

            (
                pred_rel_sub_box,
                pred_rel_sub_box_expand,
                pred_rel_sub_dist_expand,
                rel_sub_ent_label_flat,
                rel_sub_ent_scores_flat,
            ) = extract_rel_branch_pred(
                outputs["pred_rel_sub_box"][batch_ind],
                batch_ind, max_match_ent, total_pred_idx, pred_rel_sub_dist,
            )

            # multiple the matching cost for ranking
            rel_trp_scores = (
                    pred_rel_prob * rel_sub_ent_scores_flat * rel_obj_ent_scores_flat
            )  # (num_queries,  1)

            # (num_queries, num_categories)
            # =>  (num_queries * num_pred_rel * num_group_entities, 1)
            pred_rel_pred_score = pred_rel_prob.contiguous().unsqueeze(1)
            rel_trp_scores = rel_trp_scores.unsqueeze(1)

            ent_score = torch.cat((rel_sub_ent_scores_flat, rel_obj_ent_scores_flat), dim=0)
            ent_label = torch.cat((rel_sub_ent_label_flat, rel_obj_ent_label_flat), dim=0)
            ent_box = torch.cat((pred_rel_sub_box_expand, pred_rel_obj_box_expand), dim=0)
            ent_dist = torch.cat((pred_rel_sub_dist_expand, pred_rel_obj_dist_expand), dim=0)

            rel_match_sub_ids2cat = torch.arange(len(rel_sub_ent_scores_flat)).to(ent_score.device)
            rel_match_obj_ids2cat = torch.arange(len(rel_sub_ent_scores_flat)).to(ent_score.device) + len(
                rel_sub_ent_scores_flat)

            total_pred_idx = (
                total_pred_idx.reshape(-1, 1).repeat(1, max_match_ent).view(-1).contiguous().unsqueeze(1)
            )

            SUB_IDX = 0
            OBJ_IDX = 1
            REL_LABEL = 2
            REL_TRP_SCR = 3
            REL_PRED_SCR = 4
            INIT_PROP_IDX = 5

            pred_rel_triplet = torch.cat(
                (
                    rel_match_sub_ids2cat.unsqueeze(1).long().to(device),
                    rel_match_obj_ids2cat.unsqueeze(1).long().to(device),
                    pred_rel_labels.long().to(device),
                    rel_trp_scores.to(device),
                    pred_rel_pred_score.to(device),
                    total_pred_idx.to(device),
                ), 1,
            )

            REL_CONFIDENCE = 6
            if pred_rel_conf_score is not None:
                rel_conf_score_rept = (
                    pred_rel_conf_score[batch_ind]
                        .unsqueeze(-1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

                pred_rel_triplet = torch.cat(
                    (pred_rel_triplet, rel_conf_score_rept,), dim=1
                )

            pred_rel_triplet = pred_rel_triplet.to(device)
            self_iou = generalized_box_iou(
                box_cxcywh_to_xyxy(ent_box), box_cxcywh_to_xyxy(ent_box)
            )
            sub_idx = pred_rel_triplet[:, SUB_IDX].long()
            obj_idx = pred_rel_triplet[:, OBJ_IDX].long()



            # first stage filtering
            if post_proc_filtering:
                boxes_pair_iou = self_iou[sub_idx, obj_idx]
                non_self_conn_idx = torch.logical_and(boxes_pair_iou < 0.95, 
                                                      rel_sub_ent_label_flat != rel_obj_ent_label_flat)
                pred_rel_triplet = pred_rel_triplet[non_self_conn_idx]

                
                _, top_rel_idx = torch.sort(
                    pred_rel_triplet[:, REL_TRP_SCR], descending=True
                )
                top_rel_idx = top_rel_idx[: self.cfg.MODEL.REL_DETR.NUM_MAX_REL_PRED]
                pred_rel_triplet = pred_rel_triplet[top_rel_idx]


            self_iou = self_iou.detach().cpu()

            sub_idx = pred_rel_triplet[:, SUB_IDX].long().detach().cpu()
            obj_idx = pred_rel_triplet[:, OBJ_IDX].long().detach().cpu()

            rel_pred_label = (
                pred_rel_triplet[:, REL_LABEL].long().detach().cpu()
            )
            rel_pred_score = pred_rel_triplet[:, REL_TRP_SCR].detach().cpu()

            def rel_prediction_filtering(pred_rel_triplet):
                """

                Args:
                    pred_idx_set:
                    new_come_pred_idx:

                Returns:

                """
                pred_idx_set = []
                for new_come_pred_idx in range(len(pred_rel_triplet)):

                    new_come_sub_idx = sub_idx[new_come_pred_idx]
                    new_come_obj_idx = obj_idx[new_come_pred_idx]

                    new_come_sub_label = ent_label[new_come_sub_idx]
                    new_come_obj_label = ent_label[new_come_obj_idx]

                    new_come_pred_label = rel_pred_label[new_come_pred_idx]
                    new_come_pred_score = rel_pred_score[new_come_pred_idx] * ent_score[new_come_sub_idx] * ent_score[new_come_obj_idx]

                    pred_idx = torch.Tensor(pred_idx_set).long()
                    curr_sub_idx = sub_idx[pred_idx]
                    curr_obj_idx = obj_idx[pred_idx]

                    curr_sub_label = ent_label[curr_sub_idx]
                    curr_obj_label = ent_label[curr_obj_idx]

                    curr_pred_label = rel_pred_label[pred_idx]
                    curr_pred_score = rel_pred_score[pred_idx] * ent_score[curr_sub_idx] * ent_score[curr_obj_idx]

                    entities_indx_match = torch.logical_and(
                        curr_sub_idx == new_come_sub_idx,
                        curr_obj_idx == new_come_obj_idx
                    )

                    new_come_sub_idx = (torch.ones(len(pred_idx)) * new_come_sub_idx).long()
                    new_come_obj_idx = (torch.ones(len(pred_idx)) * new_come_obj_idx).long()

                    sub_iou = self_iou[new_come_sub_idx, curr_sub_idx]
                    obj_iou = self_iou[new_come_obj_idx, curr_obj_idx]

                    entities_pred_match = torch.logical_and(
                            torch.logical_and(sub_iou > self.overlap_thres, obj_iou > self.overlap_thres),
                            torch.logical_and(curr_sub_label == new_come_sub_label, curr_obj_label == new_come_obj_label)
                    )
                    entity_match = torch.logical_or(entities_pred_match, entities_indx_match)

                    if entity_match.any():
                        pred_match = curr_pred_label == new_come_pred_label
                        rel_match = torch.logical_and(entity_match, pred_match)

                        if rel_match.any():
                            is_existed = new_come_pred_score < curr_pred_score[rel_match]
                            if not is_existed.any():
                                pred_idx_set.append(new_come_pred_idx)
                        else:
                            pred_idx_set.append(new_come_pred_idx)
                        
                    else:
                        pred_idx_set.append(new_come_pred_idx)

                pred_idx_set = torch.Tensor(pred_idx_set).long().to(device)
                bin_mask = torch.zeros((pred_rel_triplet.shape[0]), dtype=torch.bool).to(
                    device
                )
                bin_mask[pred_idx_set] = True
                pred_rel_triplet_selected = pred_rel_triplet[bin_mask]

                return pred_rel_triplet_selected

            # start = time.perf_counter()
            if post_proc_filtering and self.overlap_thres > 0:

                pred_rel_triplet_selected = rel_prediction_filtering(
                    pred_rel_triplet
                )
            else:
                pred_rel_triplet_selected = pred_rel_triplet
                non_max_suppressed_idx = None

            # top K selection
            _, top_rel_idx = torch.sort(
                pred_rel_triplet_selected[:, REL_TRP_SCR], descending=True
            )
            pred_rel_triplet_selected = pred_rel_triplet_selected[top_rel_idx[:max_proposal_pairs]]

            def res2dict(pred_rel_triplet):
                ret = {
                    "rel_trp": pred_rel_triplet[:, :3].long(),
                    "rel_pred_label": pred_rel_triplet[:, REL_LABEL].long(),
                    "rel_score": pred_rel_triplet[:, REL_PRED_SCR],
                    "rel_trp_score": pred_rel_triplet[:, REL_TRP_SCR],
                    "pred_prob_dist": pred_rel_probs[batch_ind][
                        pred_rel_triplet[:, INIT_PROP_IDX].long()
                    ],
                    "init_prop_indx": pred_rel_triplet[:, INIT_PROP_IDX].long(),
                    "rel_branch_score": ent_score,
                    "rel_branch_label": ent_label.long(),
                    "rel_branch_dist": ent_dist,
                }

                if pred_rel_conf_score is not None:
                    ret["pred_rel_confidence"] = pred_rel_triplet[:, REL_CONFIDENCE]
                if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_ENTITIES_PRED:
                    ret["rel_branch_box"] = ent_box

                ret["pred_rel_ent_obj_box"] = pred_rel_obj_box[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]
                ret["pred_rel_ent_sub_box"] = pred_rel_sub_box[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]

                ret["pred_rel_ent_obj_label"] = rel_sub_ent_label_flat[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]
                ret["pred_rel_ent_sub_label"] = rel_obj_ent_label_flat[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]

                ret["pred_rel_ent_obj_score"] = rel_sub_ent_scores_flat[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]
                ret["pred_rel_ent_sub_score"] = rel_obj_ent_scores_flat[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]

                ret["pred_rel_ent_obj_dist"] = pred_rel_obj_dist[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]
                ret["pred_rel_ent_sub_dist"] = pred_rel_sub_dist[
                    pred_rel_triplet[:, INIT_PROP_IDX].long()
                ]

                return ret

            init_pred_dict = res2dict(pred_rel_triplet)
            init_rel_proposals_predict.append(init_pred_dict)

            rel_proposals_predict.append(res2dict(pred_rel_triplet_selected))

        return rel_proposals_predict, init_rel_proposals_predict


