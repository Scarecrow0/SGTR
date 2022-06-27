import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch import nn
from torchvision.ops import generalized_box_iou

from cvpods.structures import boxes as box_ops
from cvpods.structures.boxes import box_cxcywh_to_xyxy
from cvpods.utils import comm
from cvpods.utils.metrics import accuracy


class RelHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            cfg,
            cost_rel_class: float = 1.5,
            cost_rel_vec: float = 1.0,
            cost_class: float = 1.5,
            cost_bbox: float = 0.8,
            cost_giou: float = 1,
            cost_indexing: float = 0.2,
            cost_foreground_ent: float = 0.3
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
        """
        super().__init__()

        self.cost_rel_class = cost_rel_class
        self.cost_rel_vec = cost_rel_vec
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_indexing = cost_indexing
        self.cost_foreground_ent = cost_foreground_ent

        self.cfg = cfg

        self.det_match_res = None

        self.entities_aware_matching = False
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            self.entities_aware_matching = (
                self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_AWARE_MATCHING
            )

        assert cost_rel_class != 0 or cost_rel_vec != 0, "all costs cant be 0"

    def inter_vec_cost_calculation(
            self, outputs, targets,
    ):
        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        pred_rel_prob = (
            outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        pred_rel_vec = outputs["pred_rel_vec"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])
        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            # Compute the classification cost.
            pred_rel_prob = torch.sigmoid(
                outputs["pred_rel_logits"].flatten(0, 1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = -pred_rel_prob[:, tgt_rel_labels - 1]
        else:
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = -pred_rel_prob[:, tgt_rel_labels]

        # Compute the L1 cost between relationship vector
        cost_rel_vec = torch.cdist(pred_rel_vec, tgt_rel_vec, p=1)

        # Final cost matrix
        # calculate the distance matrix across all gt in batch with prediction
        C = self.cost_rel_vec * cost_rel_vec + self.cost_rel_class * cost_class

        C = C.view(bs, num_rel_queries, -1).cpu()  # bs, num_queries, all_label_in_batch

        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        detailed_cost_dict = {
            "cost_rel_vec": cost_rel_vec.view(bs, num_rel_queries, -1)
                .cpu()
                .split(sizes, -1),
            "cost_class": cost_class.view(bs, num_rel_queries, -1)
                .cpu()
                .split(sizes, -1),
        }

        return match_cost, detailed_cost_dict

    def indexing_entities_cost_calculation(
            self, outputs, targets,
    ):

        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]
        _, num_ent_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch

        pred_rel_vec = outputs["pred_rel_vec"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # batch_size, num_rel_queries, num_ent_queries
        pred_sub_index_scr = outputs["sub_entities_indexing"]
        pred_obj_index_scr = outputs["obj_entities_indexing"]

        # batch_size, num_rel_queries, num_ent_queries
        sub_idxing_rule = outputs["sub_ent_indexing_rule"]
        obj_idxing_rule = outputs["obj_ent_indexing_rule"]

        if self.training:
            num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING_TRAIN
        else:
            num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING

        # batch_size, num_rel_queries, num_ent_pairs
        num_ent_pairs = num_ent_pairs if sub_idxing_rule.shape[-1] > num_ent_pairs else sub_idxing_rule.shape[-1]
        # todo only accumulate the foreground entities that match with the GTs
        rel_match_sub_scores, rel_match_sub_ids = torch.topk(sub_idxing_rule, num_ent_pairs, dim=-1)
        rel_match_obj_scores, rel_match_obj_ids = torch.topk(obj_idxing_rule, num_ent_pairs, dim=-1)

        # rel_match_sub_scores = torch.gather(pred_sub_index_scr, -1, rel_match_sub_ids)
        # rel_match_obj_scores = torch.gather(pred_obj_index_scr, -1, rel_match_obj_ids)

        pred_ent_probs = outputs["pred_logits"].softmax(-1)
        pred_ent_boxes = outputs["pred_boxes"]

        def minmax_norm(data):
            return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 0.02)

        # batch_size, num_rel_queries, num_pairs ->  batch_size, num_rel_queries * num_pairs
        # sub_match_scores_flat = minmax_norm(rel_match_sub_scores).flatten(1, 2)
        # obj_match_scores_flat = minmax_norm(rel_match_obj_scores).flatten(1, 2)

        sub_match_scores_flat = rel_match_sub_scores.flatten(1, 2)
        obj_match_scores_flat = rel_match_obj_scores.flatten(1, 2)
        sub_idx_flat = rel_match_sub_ids.flatten(1, 2)
        obj_idx_flat = rel_match_obj_ids.flatten(1, 2)

        # batch_size, num_rel_queries * num_pairs -> # batch_size * num_rel_queries,  num_pairs
        pred_rel_sub_prob = torch.stack(
            [pred_ent_probs[i, sub_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1).contiguous()
        pred_rel_obj_prob = torch.stack(
            [pred_ent_probs[i, obj_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1).contiguous()

        pred_rel_sub_bbox = torch.stack(
            [pred_ent_boxes[i, sub_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1)
        pred_rel_obj_bbox = torch.stack(
            [pred_ent_boxes[i, obj_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1)

        # prepare targets
        # Also concat the target labels and boxes
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])
        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])

        # Also concat the target labels and boxes
        tgt_ent_labels = torch.cat([v["labels"] for v in targets]).contiguous()

        if targets[0].get('labels_non_masked') is not None:
            tgt_ent_labels = torch.cat([v["labels_non_masked"] for v in targets]).contiguous()

        tgt_ent_bbox = torch.cat([v["boxes"] for v in targets]).contiguous()

        num_total_gt_rel = len(tgt_rel_labels)

        # batch concate the pair index tensor with the start index padding
        tgt_rel_pair_idx = []
        start_idx = 0
        for v in targets:
            tgt_rel_pair_idx.append(v["gt_rel_pair_tensor"] + start_idx)
            start_idx += len(v["boxes"])
        tgt_rel_pair_idx = torch.cat(tgt_rel_pair_idx)

        # Compute cost of relationship vector
        # project the prediction probability vector to the GT probability vector

        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            # Compute the classification cost.

            pred_rel_prob = torch.sigmoid(
                outputs["pred_rel_logits"].flatten(0, 1)
            )  # [batch_size * num_queries, num_classes]
            # alpha = 0.25
            # gamma = 2.0
            # neg_cost_class = (
            #         (1 - alpha) * (pred_rel_prob ** gamma) * (-(1 - pred_rel_prob + 1e-8).log())
            # )
            # pos_cost_class = (
            #         alpha * ((1 - pred_rel_prob) ** gamma) * (-(pred_rel_prob + 1e-8).log())
            # )
            # cost_class = pos_cost_class[:, tgt_rel_labels - 1] - neg_cost_class[:, tgt_rel_labels - 1]

            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels - 1])
        else:

            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels])

        cost_rel_vec = torch.cdist(pred_rel_vec, tgt_rel_vec, p=1)

        cost_sub_class = pred_rel_sub_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 0]]]
        cost_obj_class = pred_rel_obj_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 1]]]
        # and operation, the both giou should large, one small giou will suppress the pair matching score

        cost_ent_pair_class = 32 ** (-1 * (cost_sub_class + cost_obj_class))

        # cost_sub_class = torch.cdist(pred_rel_sub_prob[:, :-1], tgt_sub_cls[:, :-1], p=2)
        # cost_obj_class = torch.cdist(pred_rel_obj_prob[:, :-1], tgt_obj_cls[:, :-1], p=2)

        cost_sub_box_l1 = torch.cdist(
            pred_rel_sub_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 0]], p=1
        )
        cost_obj_box_l1 = torch.cdist(
            pred_rel_obj_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 1]], p=1
        )

        cost_ent_pair_box_l1 = (cost_obj_box_l1 + cost_sub_box_l1) / 2

        # Compute the giou cost betwen boxes
        cost_sub_giou = torch.clip(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred_rel_sub_bbox),
                box_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 0]]),
            ),
            0,
        )
        cost_obj_giou = torch.clip(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred_rel_obj_bbox),
                box_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 1]]),
            ),
            0,
        )

        # batch_size * pair_num x gt_num

        # and operation, the both giou should large, one small giou will suppress the pair matching score
        # cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou))
        cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou))

        cost_foreground_ent = None
        if self.det_match_res is not None and self.cost_foreground_ent > 0:
            gt_ent_idxs = [v["gt_rel_pair_tensor"] for v in targets]

            def build_foreground_cost(ent_role_idx_flat, role_id):
                all_cost_foreground_ent = []
                gt_num = torch.cat(gt_ent_idxs).shape[0]
                for batch_idxs, ent_match_idx in enumerate(self.det_match_res):
                    ent_role_idx = ent_role_idx_flat[batch_idxs]
                    selected_gt_ent_idx = gt_ent_idxs[batch_idxs][:, role_id]
                    match_dict = {}
                    for pred_i, gt_i in zip(ent_match_idx[0], ent_match_idx[1]):
                        match_dict[gt_i.item()] = pred_i.item()

                    for gt_id in selected_gt_ent_idx:
                        cost_foreground_ent = torch.zeros_like(ent_role_idx).float()
                        matched_pred_i = match_dict[gt_id.item()]
                        cost_foreground_ent[torch.nonzero(ent_role_idx == matched_pred_i)] = -1
                        all_cost_foreground_ent.append(cost_foreground_ent)

                    for _ in range(gt_num - len(selected_gt_ent_idx)):
                        cost_foreground_ent = torch.zeros_like(ent_role_idx).float()
                        all_cost_foreground_ent.append(cost_foreground_ent)
                bz = len(self.det_match_res)
                cost_foreground_ent = torch.stack(all_cost_foreground_ent).reshape(bz, gt_num, -1)
                cost_foreground_ent = cost_foreground_ent.permute(0, 2, 1).reshape(-1, gt_num)
                return cost_foreground_ent

            cost_foreground_ent = build_foreground_cost(sub_idx_flat, 0) + build_foreground_cost(sub_idx_flat, 0)
            cost_foreground_ent[cost_foreground_ent > -1.5] = 0  # both role entities should matching with the GT
            cost_foreground_ent /= 2

        # batch_size * num_rel_queries, num_total_gt_rel 
        # -> batch_size * num_rel_queries * num_ent_pairs, num_total_gt_rel
        cost_rel_vec = (
            cost_rel_vec.unsqueeze(1)
                .repeat(1, num_ent_pairs, 1)
                .reshape(-1, num_total_gt_rel)
        )
        cost_class = (
            cost_class.unsqueeze(1)
                .repeat(1, num_ent_pairs, 1)
                .reshape(-1, num_total_gt_rel)
        )

        # scatter the ent_rel matching score for each gt
        # this value respect the quality of entity-rel matchin quality the triplets
        sub_match_scores_flat_to_cost = (
            sub_match_scores_flat.reshape(-1)
                .unsqueeze(1)
                .repeat(1, num_total_gt_rel)
        )

        obj_match_scores_flat_to_cost = (
            obj_match_scores_flat.reshape(-1)
                .unsqueeze(1)
                .repeat(1, num_total_gt_rel)
        )

        ent_pair_match_score = (1 - (sub_match_scores_flat_to_cost + obj_match_scores_flat_to_cost) / 2)

        # Final cost matrix
        # calculate the distance matrix across all gt in batch with prediction

        detailed_cost_dict = {
            "cost_rel_vec": (self.cost_rel_vec * cost_rel_vec),
            "cost_class": (self.cost_rel_class * cost_class),
            "cost_ent_cls": (self.cost_class * cost_ent_pair_class),
            "cost_ent_box_l1": (self.cost_bbox * cost_ent_pair_box_l1),
            "cost_ent_box_giou": (self.cost_giou * cost_ent_pair_giou),
            "cost_regrouping": self.cost_indexing * ent_pair_match_score,
        }

        if cost_foreground_ent is not None:
            detailed_cost_dict['cost_foreground_ent'] = cost_foreground_ent * self.cost_foreground_ent

        # batch_size * num_rel_queries * num_ent_pairs, num_total_gt_rel
        C = torch.zeros_like(cost_rel_vec).to(cost_rel_vec.device)
        for k, v in detailed_cost_dict.items():
            if torch.isnan(v).any():
                print(k)
            if torch.isinf(v).any():
                print(k)
            C += v
        C = C.view(bs, num_rel_queries * num_ent_pairs, -1).cpu()  # bs, num_queries, num_total_gt_rel
        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        # add the non sum cost detail for further analysis
        detailed_cost_dict["cost_sub_giou"] = 1 - cost_sub_giou
        detailed_cost_dict["cost_obj_giou"] = 1 - cost_obj_giou

        detailed_cost_dict["cost_sub_box_l1"] = cost_sub_box_l1
        detailed_cost_dict["cost_obj_box_l1"] = cost_obj_box_l1

        detailed_cost_dict["cost_sub_class"] = 1 - cost_sub_class
        detailed_cost_dict["cost_obj_class"] = 1 - cost_obj_class

        # split in to batch-wise
        for k in detailed_cost_dict.keys():
            detailed_cost_dict[k] = (
                detailed_cost_dict[k]
                    .view(bs, num_rel_queries * num_ent_pairs, -1)
                    .cpu()
                    .split(sizes, -1)
            )

        detailed_cost_dict.update({
            'sub_idx': rel_match_sub_ids,
            "obj_idx": rel_match_obj_ids
        })

        return match_cost, detailed_cost_dict

    def get_ent_pred_prob(self, pred_rel_ent_logits):
        pred_rel_obj_prob = (
            pred_rel_ent_logits.flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        return pred_rel_obj_prob

    def inter_vec_entities_cost_calculation(
            self, outputs, targets,
    ):
        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        pred_rel_prob = (
            outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        pred_rel_vec = outputs["pred_rel_vec"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        pred_rel_obj_prob = self.get_ent_pred_prob(outputs["pred_rel_obj_logits"])
        # [batch_size * num_queries, num_classes]

        pred_rel_obj_bbox = outputs["pred_rel_obj_box"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        pred_rel_sub_prob = self.get_ent_pred_prob(outputs["pred_rel_sub_logits"])
        # [batch_size * num_queries, num_classes]

        pred_rel_sub_bbox = outputs["pred_rel_sub_box"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])

        # Also concat the target labels and boxes
        tgt_ent_labels = torch.cat([v["labels"] for v in targets]).contiguous()
        if targets[0].get('labels_non_masked') is not None:
            tgt_ent_labels = torch.cat([v["labels_non_masked"] for v in targets]).contiguous()
        tgt_ent_bbox = torch.cat([v["boxes"] for v in targets])

        # batch concate the pair index tensor with the start index padding
        tgt_rel_pair_idx = []
        start_idx = 0
        for v in targets:
            tgt_rel_pair_idx.append(v["gt_rel_pair_tensor"] + start_idx)
            start_idx += len(v["boxes"])
        tgt_rel_pair_idx = torch.cat(tgt_rel_pair_idx)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            # Compute the classification cost.
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels - 1])
        else:
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels])

        # Compute the L1 cost between relationship vector
        cost_rel_vec = torch.cdist(pred_rel_vec, tgt_rel_vec, p=1)

        # Compute cost of relationship vector
        cost_sub_class = pred_rel_sub_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 0]]]
        cost_obj_class = pred_rel_obj_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 1]]]

        cost_ent_pair_class = 32 ** (-1 * (cost_sub_class + cost_obj_class))

        cost_sub_box_l1 = torch.cdist(
            pred_rel_sub_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 0]], p=1
        )
        cost_obj_box_l1 = torch.cdist(
            pred_rel_obj_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 1]], p=1
        )
        cost_ent_pair_box_l1 = (cost_obj_box_l1 + cost_sub_box_l1) / 2

        cost_sub_giou = torch.clip(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred_rel_sub_bbox),
                box_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 0]]),
            ),
            0,
        )
        cost_obj_giou = torch.clip(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred_rel_obj_bbox),
                box_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 1]]),
            ),
            0,
        )
        cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou))

        # Final cost matrix
        # calculate the distance matrix across all gt in batch with prediction

        detailed_cost_dict = {
            "cost_rel_vec": (self.cost_rel_vec * cost_rel_vec),
            "cost_class": (self.cost_rel_class * cost_class),
            "cost_ent_cls": self.cost_class * cost_ent_pair_class,
            "cost_ent_box_l1": (self.cost_bbox * cost_ent_pair_box_l1),
            "cost_ent_box_giou": (self.cost_giou * cost_ent_pair_giou),
        }

        C = torch.zeros_like(cost_rel_vec).to(cost_rel_vec.device)
        for v in detailed_cost_dict.values():
            C += v
        C = C.view(bs, num_rel_queries, -1).cpu()  # bs, num_queries, all_label_in_batch

        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        # add the non sum cost detail for further analysis
        detailed_cost_dict["cost_sub_giou"] = 1 - cost_sub_giou
        detailed_cost_dict["cost_obj_giou"] = 1 - cost_obj_giou

        detailed_cost_dict["cost_sub_box_l1"] = cost_sub_box_l1
        detailed_cost_dict["cost_obj_box_l1"] = cost_obj_box_l1

        detailed_cost_dict["cost_sub_class"] = 1 - cost_sub_class
        detailed_cost_dict["cost_obj_class"] = 1 - cost_obj_class

        for k in detailed_cost_dict.keys():
            detailed_cost_dict[k] = (
                detailed_cost_dict[k]
                    .view(bs, num_rel_queries, -1)
                    .cpu()
                    .split(sizes, -1)
            )

        return match_cost, detailed_cost_dict

    def top_score_match(self, match_cost, return_init_idx=False):
        indices_all = []
        for cost in [c[i] for i, c in enumerate(match_cost)]:
            cost_inplace = copy.deepcopy(cost)
            topk = self.cfg.MODEL.REL_DETR.NUM_MATCHING_PER_GT
            indice_multi = []
            for _ in range(topk):
                # selective matching:
                # We observe the the macthing is only happend in the 
                # small set of predictions that have top K cost value,
                # to this end, we optimize the matching pool by: instead 
                # matching with all possible prediction, we use the
                # top K times of GT num predictions for matching
                min_cost = cost_inplace.min(-1)[0]
                selected_range = self.cfg.MODEL.REL_DETR.MATCHING_RANGE
                selected_range = (
                    selected_range
                    if selected_range < cost_inplace.shape[0]
                    else cost_inplace.shape[0]
                )
                _, idx = min_cost.topk(selected_range, largest=False)
                indices = linear_sum_assignment(cost_inplace[idx, :])
                indices = (idx[indices[0]], indices[1])
                # if one pred match with the gt, we exclude it
                cost_inplace[indices[0], :] = 1e10
                indice_multi.append(indices)

            if self.training:
                # filtering that the prediction from one query is matched with the multiple GT
                init_pred_idx = np.concatenate([each[0] for each in indice_multi])
                if (
                        self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED
                        and self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_INDEXING
                ):
                    num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING_TRAIN
                    pred_idx = init_pred_idx // num_ent_pairs
                    # transform into the indices along the query num
                else:
                    pred_idx = init_pred_idx

                # check the matching relationship between the query id and GT id
                gt_idx = np.concatenate([each[1] for each in indice_multi])
                dup_match_dict = dict()
                for init_idx, (p_i, g_i) in enumerate(zip(pred_idx, gt_idx)):
                    if dup_match_dict.get(p_i) is not None:
                        if cost[p_i][dup_match_dict[p_i][1]] > cost[p_i][g_i]:
                            # print(cost[p_i][dup_match_dict[p_i]], cost[p_i][g_i])
                            # print(p_i, dup_match_dict[p_i], g_i)
                            dup_match_dict[p_i] = (init_idx, g_i)
                    else:
                        dup_match_dict[p_i] = (init_idx, g_i)

                init_pred_idx_sort = []
                pred_idx = []
                gt_idx = []
                for p_i, (init_idx, g_i) in dup_match_dict.items():
                    pred_idx.append(p_i)
                    gt_idx.append(g_i)
                    init_pred_idx_sort.append(init_pred_idx[init_idx])

                if return_init_idx:
                    indices_all.append((np.array(init_pred_idx_sort), np.array(gt_idx)))
                else:
                    indices_all.append((np.array(pred_idx), np.array(gt_idx)))
            else:
                indices_all.append(
                    (
                        np.concatenate([each[0] for each in indice_multi]),
                        np.concatenate([each[1] for each in indice_multi]),
                    )
                )
            # match_idx = torch.topk(-1 * cost.transpose(1, 0), topk, dim=-1)
            # pred_idx = match_idx[1].reshape(-1)
            # gt_idx = torch.arange(cost.shape[1]).unsqueeze(1).repeat(1, topk).reshape(-1)
            # indices.append((pred_idx, gt_idx))

        return indices_all

    @torch.no_grad()
    def forward(self, outputs, targets, ent_match=True, return_init_idx=False, det_match_res=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                            classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                            box coordinates
                'pred_rel_logits': [batch_size, num_rel_queries, num_classes]
                "pred_rel_vec": [batch_size, num_rel_queries, 4]

            targets: This is a list of targets (len(targets) = batch_size), where each target
                            is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                            of ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        self.det_match_res = det_match_res
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED and ent_match:
            if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_INDEXING:
                if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_REL_VEC_MATCH_ONLY:
                    match_cost, detailed_cost_dict = self.inter_vec_cost_calculation(
                        outputs, targets
                    )
                else:
                    (match_cost,
                     detailed_cost_dict) = self.indexing_entities_cost_calculation(outputs, targets)
            else:
                match_cost, detailed_cost_dict = self.inter_vec_entities_cost_calculation(
                    outputs, targets
                )
        else:
            match_cost, detailed_cost_dict = self.inter_vec_cost_calculation(
                outputs, targets
            )

        indices = self.top_score_match(match_cost, return_init_idx)

        match_cost_each_img = [c[i] for i, c in enumerate(match_cost)]

        for k in detailed_cost_dict.keys():
            if "cost" in k:
                detailed_cost_dict[k] = [c[i] for i, c in enumerate(detailed_cost_dict[k])]

        match_idx = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

        return match_idx, match_cost_each_img, detailed_cost_dict


class RelSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                        relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.rel_num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.cfg = cfg

        self.det_match_res = None

        self.loss_map = {
            "rel_labels": self.loss_labels,
            "rel_labels_fl": self.loss_labels_fl,
            "relation_vec": self.loss_relation_vec,
            "cardinality": self.loss_cardinality,
            "rel_entities_aware": self.loss_aux_entities_awareness,
            "rel_entities_indexing": self.loss_aux_entities_indexing,
            'rel_dynamic_query': self.loss_dynamic_query,
            "disalign_loss": self.disalign_loss
        }

        self.entities_num_classes = cfg.MODEL.DETR.NUM_CLASSES

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef

        self.register_buffer("empty_weight", empty_weight)

        empty_weight = torch.ones(self.entities_num_classes + 1)
        empty_weight[-1] = cfg.MODEL.DETR.EOS_COEFF
        self.register_buffer("entities_empty_weight", empty_weight)

        self.entities_match_cache = {}

        self.ce_loss_exp_reweight = None
        if "disalign_loss" in losses:
            self.ce_loss_exp_reweight = self._init_disalign_weights(exp_scale=self.cfg.MODEL.DISALIGN.EXP_SCALE)

    def _init_disalign_weights(self, exp_scale=0.7):
        import numpy as np
        import os

        save_file = os.path.join(self.cfg.OUTPUT_DIR, "vgs_statistics.cache")
        stat_result = torch.load(save_file, map_location=torch.device("cpu"))

        num_samples_list = stat_result['inst_cnt']
        num_foreground = len(num_samples_list)
        assert num_foreground > 0, "num_samples_list is empty"
        # assert exp_scale <= 1.0 and exp_scale >= 0, "exp_scale must less than 1.0 and "

        num_shots = num_samples_list
        ratio_list = num_shots / np.sum(num_shots)

        exp_reweight = 1 / (ratio_list ** exp_scale)

        exp_reweight = exp_reweight / np.sum(exp_reweight) * num_foreground
        exp_reweight = torch.tensor(exp_reweight).float()
        final_reweight = exp_reweight
        # final_reweight = torch.ones(num_foreground)
        # final_reweight[1:] = exp_reweight

        return final_reweight

    def disalign_loss(self, outputs, targets, indices, num_boxes, log=True):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        """Classification loss (CE loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rel_logits" in outputs
        del num_boxes

        src_logits = outputs["pred_rel_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["rel_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes_no_mask = torch.cat(
            [t["rel_label_no_mask"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o
        valid_indx = target_classes > 0

        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss_ce = F.cross_entropy(
            src_logits[valid_indx], target_classes[valid_indx] - 1,
            self.ce_loss_exp_reweight.to(src_logits.device)
        )
        losses = {"loss_rel_ce(disalign)": loss_ce}

        return losses

    def loss_dynamic_query(self, outputs, targets, indices, num_boxes, log=True):
        out = outputs['dynamic_query_pred']
        losses_box = self.loss_aux_entities_boxes(out, targets, indices, num_boxes)
        losses_cls = self.loss_aux_entities_labels(out, targets, indices, num_boxes, log=True)

        losses = {}
        for loss_dict in (losses_box, losses_cls):
            for k, v in loss_dict.items():
                losses[k + '(dynamic_query)'] = v
                if v.requires_grad:
                    losses[k + '(dynamic_query)'] *= self.cfg.MODEL.REL_DETR.DYNAMIC_QUERY_AUX_LOSS_WEIGHT

        return losses

    def loss_labels_fl(self, outputs, targets, indices, num_gt_rel, log=True):
        """Classification loss (Focal loss NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rel_logits" in outputs

        del num_gt_rel

        src_logits = outputs["pred_rel_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["rel_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes_o_init = torch.cat(
            [t["rel_label_no_mask"][J] for t, (_, J) in zip(targets, indices)]
        )

        # get the valid label indices
        target_classes_ref = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )
        target_classes_ref[idx] = target_classes_o
        valid_indx = target_classes_ref >= 0

        # build onehot labels
        target_classes_ref_init = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )
        target_classes_ref_init[idx] = target_classes_o_init

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes_ref_init.unsqueeze(-1), 1)

        # target_classes_onehot[idx[0], idx[1], target_classes_o] = 1

        # extract the non resamplind index
        valid_prd_logits = src_logits[valid_indx, :, ]
        valid_label_onehot = target_classes_onehot[valid_indx][:, 1:]
        # label ids start from 0

        # focal loss
        alpha = self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ALPHA
        gamma = self.cfg.MODEL.REL_DETR.FOCAL_LOSS.GAMMA

        num_gt_rel = sum(len(t["rel_labels"]) for t in targets)
        num_gt_rel = torch.as_tensor(
            [num_gt_rel], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        loss_rel_focal = (
                sigmoid_focal_loss(
                    valid_prd_logits,
                    valid_label_onehot,
                    num_gt_rel,
                    alpha=alpha,
                    gamma=gamma,
                )
                * src_logits.shape[1]
        )
        losses = {"loss_rel_focal": loss_rel_focal}

        if log:
            # minus 1, labels id start from 0
            target_classes_o = torch.cat(
                [t["rel_label_no_mask"][J] - 1 for t, (_, J) in zip(targets, indices)]
            )
            losses["rel_class_error"] = (
                    100 - accuracy(src_logits[idx], target_classes_o)[0]
            )

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (CE loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rel_logits" in outputs
        del num_boxes

        src_logits = outputs["pred_rel_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["rel_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes_no_mask = torch.cat(
            [t["rel_label_no_mask"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )


        target_classes[idx] = target_classes_o
        valid_indx = target_classes >= 0

        # print(len((target_classes > 0).nonzero()))
        loss_ce = F.cross_entropy(
            src_logits[valid_indx], target_classes[valid_indx], self.empty_weight
        )
        losses = {"loss_rel_ce": loss_ce}

        if log:
            losses["rel_class_error"] = (
                    100
                    - accuracy(
                src_logits[idx][target_classes_no_mask >= 0],
                target_classes_no_mask[target_classes_no_mask >= 0],
            )[0]
            )

        return losses

    def loss_aux_entities_awareness(self, outputs, targets, indices, num_boxes):
        loss_ent_label = self.loss_aux_entities_labels(
            outputs, targets, indices, num_boxes
        )
        loss_ent_box = self.loss_aux_entities_boxes(
            outputs, targets, indices, num_boxes
        )

        loss_ent_box.update(loss_ent_label)
        return loss_ent_box

    def get_fg_pred_idx(self, outputs, targets, indices):

        src_logits = outputs["pred_rel_logits"]

        target_classes_no_mask = torch.cat(
            [t["rel_label_no_mask"][J] for t, (_, J) in zip(targets, indices)]
        )

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_no_mask

        fg_idx = target_classes > 0

        return fg_idx

    def loss_aux_entities_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rel_obj_logits" in outputs

        idx = self._get_src_permutation_idx(indices)

        def cal_entities_class_loss(src_logits, role_name):
            role_id = 0 if role_name == "sub" else 1

            target_classes_o = torch.cat(
                [
                    t["labels"][t["gt_rel_pair_tensor"][:, role_id]][J]
                    for t, (_, J) in zip(targets, indices)
                ]
            )  # label cate id from 0 to num-classes-1

            target_classes_cmp = target_classes_o

            if targets[0].get('labels_non_masked') is not None:
                target_classes_o = torch.cat(
                    [
                        t["labels_non_masked"][t["gt_rel_pair_tensor"][:, role_id]][J]
                        for t, (_, J) in zip(targets, indices)
                    ]
                )  # label cate id from 0 to num-classes-1

            target_classes = torch.full(
                src_logits.shape[:2],
                src_logits.shape[-1] - 1,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o

            valid_indx = target_classes >= 0

            loss_ce = F.cross_entropy(
                src_logits[valid_indx],
                target_classes[valid_indx],
                self.entities_empty_weight
            )

            losses = {f"loss_aux_{role_name}_entities_labels_ce": loss_ce}

            if log:
                # only evaluate the foreground class entities
                losses[f"aux_{role_name}_entities_class_error"] = (
                        100 - accuracy(src_logits[idx][:, :-1], target_classes_cmp)[0]
                )
            return losses

        src_obj_logits = outputs["pred_rel_obj_logits"]
        src_sub_logits = outputs["pred_rel_sub_logits"]

        losses = cal_entities_class_loss(src_obj_logits, "obj")
        losses.update(cal_entities_class_loss(src_sub_logits, "sub"))

        return losses

    def _cal_entities_box_loss(self, targets, indices, src_boxes, role_name, num_boxes):
        role_id = 0 if role_name == "sub" else 1
        target_boxes = torch.cat(
            [
                t["boxes"][t["gt_rel_pair_tensor"][:, role_id]][i]
                for t, (_, i) in zip(targets, indices)
            ],
            dim=0,
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )

        losses = {}
        losses[f"box_l1"] = loss_bbox.sum() / num_boxes
        losses[f"box_giou"] = (
                loss_giou.sum() / num_boxes
        )
        return losses

    def _cal_entities_class_loss(self, targets, indices, src_logits, role_name, num_boxes):
        role_id = 0 if role_name == "sub" else 1
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [
                t["labels"][t["gt_rel_pair_tensor"][:, role_id]][J]
                for t, (_, J) in zip(targets, indices)
            ]
        )  # label cate id from 0 to num-classes-1

        target_classes_cmp = target_classes_o

        if targets[0].get('labels_non_masked') is not None:
            target_classes_o = torch.cat(
                [
                    t["labels_non_masked"][t["gt_rel_pair_tensor"][:, role_id]][J]
                    for t, (_, J) in zip(targets, indices)
                ]
            )  # label cate id from 0 to num-classes-1

        target_classes = torch.full(
            src_logits.shape[:2],
            src_logits.shape[-1] - 1,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        valid_indx = target_classes >= 0

        loss_ce = F.cross_entropy(
            src_logits[valid_indx],
            target_classes[valid_indx],
            self.entities_empty_weight
        )

        losses = {f"labels_ce": loss_ce}
        # only evaluate the foreground class entities
        losses["class_error"] = (
                100 - accuracy(src_logits[idx][:, :-1], target_classes_cmp)[0]
        )
        return losses

    def loss_aux_entities_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the
        image size.
        """
        assert "pred_rel_obj_box" in outputs
        idx = self._get_src_permutation_idx(indices)

        def cal_entities_cox_loss(src_boxes, role_name):
            role_id = 0 if role_name == "sub" else 1
            target_boxes = torch.cat(
                [
                    t["boxes"][t["gt_rel_pair_tensor"][:, role_id]][i]
                    for t, (_, i) in zip(targets, indices)
                ],
                dim=0,
            )

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

            loss_giou = 1 - torch.diag(
                generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )
            )

            losses = {}
            losses[f"loss_aux_{role_name}_entities_boxes"] = loss_bbox.sum() / num_boxes
            losses[f"loss_aux_{role_name}_entities_boxes_giou"] = (
                    loss_giou.sum() / num_boxes
            )
            return losses

        losses = cal_entities_cox_loss(outputs["pred_rel_obj_box"][idx], "obj")
        losses.update(cal_entities_cox_loss(outputs["pred_rel_sub_box"][idx], "sub"))

        return losses

    def loss_aux_entities_indexing(self, outputs, targets, indices, num_boxes):
        assert "sub_entities_indexing" in outputs

        num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING_TRAIN

        topk_pred = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING
        topk_all = num_ent_pairs

        def cal_entities_index_loss(matching_matrix, role_name):
            role_id = 0 if role_name == "sub" else 1
            num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING_TRAIN
            matching_ref_mat_collect = []
            num_ent_pairs = num_ent_pairs if num_ent_pairs < matching_matrix.shape[-1] else matching_matrix.shape[-1]
            _, rel_match_pred_ent_ids = torch.topk(
                matching_matrix, num_ent_pairs, dim=-1
            )

            valid_rel_idx = (
                []
            )  # the index of relationship prediction match with the and also has the fg indexing matching
            batch_offset = 0  # the offset for concatenated all indices in batch from different images

            entities_match_cache = []
            if self.entities_match_cache is None:
                self.entities_match_cache = dict()

            for batch_idx in range(len(indices)):
                if self.entities_match_cache.get(role_name) is None:
                    # check the prediction box quality
                    pred_boxes = box_ops.box_cxcywh_to_xyxy(
                        outputs["pred_boxes"][batch_idx]
                    )
                    tgt_boxes = box_ops.box_cxcywh_to_xyxy(targets[batch_idx]["boxes"])

                    box_giou = generalized_box_iou(pred_boxes, tgt_boxes).detach()
                    box_match_idx = box_giou >= 0.7
                    inst_loc_hit_idx = torch.nonzero(box_match_idx)
                    pred_box_loc_hit_idx = inst_loc_hit_idx[:, 0]
                    gt_box_loc_hit_idx = inst_loc_hit_idx[:, 1]

                    loc_box_matching_results = defaultdict(set)
                    for idx in range(len(gt_box_loc_hit_idx)):
                        loc_box_matching_results[gt_box_loc_hit_idx[idx].item()].add(
                            pred_box_loc_hit_idx[idx].item()
                        )

                    pred_labels = outputs["pred_logits"][batch_idx, :, :-1].max(-1)[1]
                    tgt_labels = targets[batch_idx]["labels"]
                    gt_det_label_to_cmp = pred_labels[pred_box_loc_hit_idx]
                    pred_det_label_to_cmp = tgt_labels[gt_box_loc_hit_idx]

                    pred_det_hit_stat = pred_det_label_to_cmp == gt_det_label_to_cmp

                    pred_box_det_hit_idx = pred_box_loc_hit_idx[pred_det_hit_stat]
                    gt_box_det_hit_idx = gt_box_loc_hit_idx[pred_det_hit_stat]

                    det_box_matching_results = defaultdict(set)
                    for idx in range(len(gt_box_det_hit_idx)):
                        det_box_matching_results[gt_box_det_hit_idx[idx].item()].add(
                            pred_box_det_hit_idx[idx].item()
                        )

                    # merge the entities set matching results
                    # det_box_matching_results = defaultdict(set)
                    if self.det_match_res is not None:
                        gt_ent_idxs = self.det_match_res[batch_idx][1]
                        pred_ent_idxs = self.det_match_res[batch_idx][0]
                        for idx in range(len(gt_ent_idxs)):
                            gt_ent_idx = gt_ent_idxs[idx].item()
                            pred_ent_idx = pred_ent_idxs[idx].item()
                            det_box_matching_results[gt_ent_idx].add(
                                pred_ent_idx
                            )
                    # loc_box_matching_results = det_box_matching_results

                    entities_match_cache.append({
                        'loc_box_matching_results': loc_box_matching_results,
                        'det_box_matching_results': det_box_matching_results
                    })

                else:
                    loc_box_matching_results = self.entities_match_cache[role_name][batch_idx][
                        'loc_box_matching_results']
                    det_box_matching_results = self.entities_match_cache[role_name][batch_idx][
                        'det_box_matching_results']

                # todo: 2nd entities loss

                # indexing loss
                pred_rel_idx, gt_rel_idx = indices[batch_idx]
                ent_pred_idx = rel_match_pred_ent_ids[batch_idx, pred_rel_idx, :]

                rel_pred_num = outputs["pred_rel_obj_logits"][batch_idx].shape[0]
                ent_pred_num = outputs["pred_logits"][batch_idx].shape[0]
                matching_ref_mat = torch.zeros(
                    (rel_pred_num, ent_pred_num), device=outputs["pred_boxes"].device
                ).long()

                matching_ref_mat_loc = torch.zeros(
                    (rel_pred_num, ent_pred_num), device=outputs["pred_boxes"].device
                ).long()

                # assign matching results as foreground if set matching result
                # is also foreground entities

                gt_rel_pair_tensor = targets[batch_idx]["gt_rel_pair_tensor"]
                for idx, (rel_idx, gt_rel_idx) in enumerate(
                        zip(pred_rel_idx.cpu().numpy(), gt_rel_idx.cpu().numpy())
                ):
                    gt_box_idx = gt_rel_pair_tensor[gt_rel_idx, role_id].item()
                    # for ent_idx in ent_pred_idx[idx].cpu().numpy():
                    for ent_idx in ent_pred_idx[idx].cpu().numpy():
                        if ent_idx in loc_box_matching_results[gt_box_idx]:
                            matching_ref_mat_loc[rel_idx, ent_idx] = 1
                        if ent_idx in det_box_matching_results[gt_box_idx]:
                            matching_ref_mat[rel_idx, ent_idx] = 1

                matching_ref_mat_collect.append(matching_ref_mat.long())

                valid_rel_idx.append(pred_rel_idx.unique() + batch_offset)
                batch_offset += rel_pred_num

            if self.entities_match_cache.get(role_name) is None:
                self.entities_match_cache[role_name] = entities_match_cache

            matching_ref_mat = torch.cat(matching_ref_mat_collect, dim=0).float()
            valid_idx = matching_ref_mat.sum(-1) > 0
            # print(valid_idx.sum()/ valid_idx.shape[0])
            # bs*rel_num_pred, ent_num_pred
            matching_matrix = matching_matrix.view(-1, matching_matrix.shape[-1])
            # print(matching_matrix[valid_idx].max(-1))
            # print(matching_matrix_log[valid_idx][matching_ref_mat[valid_idx] !=0].sigmoid())
            # match_loss = nn.BCEWithLogitsLoss()(matching_matrix[valid_idx], matching_ref_mat[valid_idx])

            alpha = self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_FOCAL_LOSS.ALPHA
            gamma = self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_FOCAL_LOSS.GAMMA
            match_loss = focal_loss(
                matching_matrix[valid_idx],
                matching_ref_mat[valid_idx],
                alpha,
                gamma,
                reduction="none",
            )
            # only take the fg matching results
            # average by instances number
            match_loss = match_loss.sum() / match_loss.shape[0]
            valid_pred_rel_idx = torch.cat(valid_rel_idx)
            # the index that prediction rel match with the GT

            if (
                    len(torch.nonzero(matching_matrix)) > 0
                    and len(torch.nonzero(matching_ref_mat)) != len(matching_ref_mat)
                    and len(torch.nonzero(valid_pred_rel_idx)) > 0
            ):

                # y = matching_ref_mat[valid_idx].view(-1).detach().long().cpu().numpy()
                # pred = matching_matrix[valid_idx].view(-1).detach().cpu().numpy()
                # fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
                # auc = metrics.auc(fpr, tpr)
                # auc = torch.Tensor([auc])

                def topk_matching_acc(topk):
                    topk = topk if topk < matching_matrix[valid_pred_rel_idx].shape[-1] else \
                        matching_matrix[valid_pred_rel_idx].shape[-1]
                    pred_idx = matching_matrix[valid_pred_rel_idx].topk(topk, dim=-1)[1]
                    matching_matrix_bin = torch.zeros_like(
                        matching_matrix[valid_pred_rel_idx]
                    )
                    for i in range(len(pred_idx)):
                        matching_matrix_bin[i, pred_idx[i]] = 1
                    hit_res = (
                            matching_matrix_bin == matching_ref_mat[valid_pred_rel_idx]
                    )
                    hit_res[matching_matrix_bin == 0] = False
                    hit_res = hit_res.any(-1)
                    return hit_res

                # top1 accuracy
                top1_hit_res = topk_matching_acc(1)
                # top5 accuracy
                top5_hit_res = topk_matching_acc(topk_pred)
                all_hit_res = topk_matching_acc(topk_all)

            else:
                top1_hit_res = None
                top5_hit_res = None
                all_hit_res = None

            if torch.isnan(match_loss):
                match_loss = torch.Tensor([0.0])
            auc = torch.Tensor([float("nan")])

            return match_loss, auc, top1_hit_res, top5_hit_res, all_hit_res

        def acc_calculate(sub_hit, obj_hit):
            if sub_hit is not None and obj_hit is not None:
                return (
                        torch.all(
                            torch.cat((sub_hit.unsqueeze(1), obj_hit.unsqueeze(1)), dim=-1),
                            dim=-1,
                        )
                        .long()
                        .sum()
                        / sub_hit.shape[0]
                )
            else:
                return torch.Tensor([float("nan")])

        ret_dict = {}
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_TYPE != 'rule_base':
            (
                sub_losses,
                _,
                sub_top1_hit_res,
                sub_top5_hit_res,
                sub_all_hit_res,
            ) = cal_entities_index_loss(outputs["sub_entities_indexing"], "sub")
            (
                obj_losses,
                _,
                obj_top1_hit_res,
                obj_top5_hit_res,
                obj_all_hit_res,
            ) = cal_entities_index_loss(outputs["obj_entities_indexing"], "obj")

            ret_dict.update({
                "loss_sub_entities_indexin": sub_losses,
                "loss_obj_entities_indexin": obj_losses,
                "ent_idx_acc_top1": acc_calculate(sub_top1_hit_res, obj_top1_hit_res),
                f"ent_idx_acc_top{topk_pred}": acc_calculate(
                    sub_top5_hit_res, obj_top5_hit_res
                ),
                f"ent_idx_acc_top{topk_all}": acc_calculate(
                    sub_all_hit_res, obj_all_hit_res
                ),
            })

        (
            _,
            _,
            sub_top1_hit_res,
            sub_top5_hit_res,
            sub_all_hit_res,
        ) = cal_entities_index_loss(outputs["sub_ent_indexing_rule"], "sub")
        (
            _,
            _,
            obj_top1_hit_res,
            obj_top5_hit_res,
            obj_all_hit_res,
        ) = cal_entities_index_loss(outputs["obj_ent_indexing_rule"], "obj")

        ret_dict.update(
            {
                "ent_idx_acc_top1-r": acc_calculate(sub_top1_hit_res, obj_top1_hit_res),
                f"ent_idx_acc_top{topk_pred}-r": acc_calculate(
                    sub_top5_hit_res, obj_top5_hit_res
                ),
                f"ent_idx_acc_top{topk_all}-r": acc_calculate(
                    sub_all_hit_res, obj_all_hit_res
                ),
            }
        )

        return ret_dict

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty
        boxes. This is not really a loss, it is intended for logging purposes only. It doesn't
        propagate gradients
        This error can indicate how the model learn about the NMS strategy
        """
        del indices
        del num_boxes
        pred_logits = outputs["pred_rel_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["rel_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != 0).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"rel_cardinality_error": card_err}
        return losses

    def loss_relation_vec(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the
        image size.
        """
        assert "pred_rel_vec" in outputs
        idx = self._get_src_permutation_idx(indices)
        # here only take the fg relationship vector for supervision
        src_boxes = outputs["pred_rel_vec"][idx]
        target_boxes = torch.cat(
            [t["rel_vector"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {"loss_rel_vector": loss_bbox.sum() / num_boxes}

        return losses


    def _get_src_permutation_idx(self, indices, get_init_index=False):
        """
        extract the src_idx of matching result with the batch indicator array,
        and concate them into whole tensor array
        Args:
            indices: Tuple[Tuple[Tensor, Tensor]]

        Returns:
            batch_idx : Tensor,
            tgt_idx: Tensor

        """
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

        assert loss in self.loss_map, f"do you really want to compute {loss} loss?"
        return self.loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, match_res=None, det_match_res=None, entities_match_cache=None):
        """
        This performs the loss computation.

        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each
                      loss' doc
        """

        self.det_match_res = det_match_res
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "ref_pts_pred"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_gt_rel = sum(len(t["rel_labels"]) for t in targets)
        num_gt_rel = torch.as_tensor(
            [num_gt_rel], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(num_gt_rel)
        num_gt_rel = torch.clamp(num_gt_rel / comm.get_world_size(), min=1).item()

        losses = {}
        self.entities_match_cache = entities_match_cache
        # the matching and loss calculation is done for the last layer prediction previously,
        # here just skip the matching opteration for saving time
        if match_res is None:
            # Retrieve the matching between the outputs of the last layer and the targets
            (indices, match_cost, detailed_cost_dict) = self.matcher(
                outputs_without_aux, targets, det_match_res=det_match_res
            )

            match_res = (indices, match_cost, detailed_cost_dict)
            # Compute all the requested losses
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_gt_rel))
        else:
            (indices, match_cost, detailed_cost_dict) = match_res

        # # In case of auxiliary losses, we repeat this process with the output of
        # # each intermediate layer.
        if "aux_outputs" in outputs:
            aux_match_indices = []
            aux_match_cost = []
            aux_cost_dict = []

            if outputs["aux_outputs"][0].get('pred_rel_logits') is not None:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    if not self.cfg.MODEL.REL_DETR.USE_FINAL_MATCH:
                        (indices, match_cost, detailed_cost_dict) = self.matcher(aux_outputs, targets)
                        aux_match_indices.append(indices)
                        aux_match_cost.append(match_cost)
                        aux_cost_dict.append(detailed_cost_dict)

                    for loss in self.losses:
                        kwargs = {}
                        if loss == "rel_labels":
                            # Logging is enabled only for the last layer
                            kwargs = {"log": False}

                        l_dict = self.get_loss(
                            loss, aux_outputs, targets, indices, num_gt_rel, **kwargs
                        )

                        l_dict = {k + f"/layer{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
            # collect aux loss
            if len(aux_match_indices) > 0:
                match_res = (aux_match_indices, aux_match_cost, aux_cost_dict)
            else:
                match_res = None

        if "ref_pts_pred" in outputs:
            aux_outputs = outputs["ref_pts_pred"]
            (indices, match_cost, _) = self.matcher(
                aux_outputs, targets, ent_match=False
            )  # disable entities matching
            for loss in ["relation_vec", "rel_labels"]:
                kwargs = {}
                # if loss == "rel_labels":
                #     # Logging is enabled only for the last layer
                #     kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices, num_gt_rel, **kwargs
                )
                l_dict = {"ref_pts_pred-" + k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, match_res


def focal_loss(valid_prd_logits, valid_label, alpha, gamma, reduction):
    # extract the non resamplind index
    valid_prd_score = torch.sigmoid(valid_prd_logits).detach()
    valid_label = valid_label.long()

    # focal loss
    pt = (1 - valid_prd_score) * valid_label + valid_prd_score * (1 - valid_label)
    focal_weight = (alpha * valid_label + (1 - alpha) * (1 - valid_label)) * pt.pow(
        gamma
    )
    rel_loss = focal_weight * F.binary_cross_entropy_with_logits(
        valid_prd_logits,
        valid_label.type_as(valid_prd_score),
        weight=focal_weight.detach(),
        reduction=reduction,
    )
    return rel_loss


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
