# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr_matcher.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

import copy
from cvpods.structures.boxes import box_cxcywh_to_xyxy
from torchvision.ops import generalized_box_iou
import numpy as np
from cvpods.utils.metrics import accuracy

from cvpods.utils import comm

class HungarianPairMatcher(nn.Module):
    def __init__(self, set_cost_act, set_cost_idx, set_cost_tgt):
        """Creates the matcher
        Params:
            cost_action: This is the relative weight of the multi-label action classification error in the matching cost
            cost_hbox: This is the relative weight of the classification error for human idx in the matching cost
            cost_obox: This is the relative weight of the classification error for object idx in the matching cost
        """
        super().__init__()
        self.cost_action = set_cost_act
        self.cost_hbox = self.cost_obox = set_cost_idx
        self.cost_target = set_cost_tgt
        assert self.cost_action != 0 or self.cost_hbox != 0 or self.cost_obox != 0, "all costs cant be 0"

    def reduce_redundant_gt_box(self, tgt_bbox, indices):
        """Filters redundant Ground-Truth Bounding Boxes
        Due to random crop augmentation, there exists cases where there exists
        multiple redundant labels for the exact same bounding box and object class.
        This function deals with the redundant labels for smoother HOTR training.
        """
        tgt_bbox_unique, map_idx, idx_cnt = torch.unique(tgt_bbox, dim=0, return_inverse=True, return_counts=True)

        k_idx, bbox_idx = indices
        triggered = False
        if (len(tgt_bbox) != len(tgt_bbox_unique)):
            map_dict = {k: v for k, v in enumerate(map_idx)}
            map_bbox2kidx = {int(bbox_id): k_id for bbox_id, k_id in zip(bbox_idx, k_idx)}

            bbox_lst, k_lst = [], []
            for bbox_id in bbox_idx:
                if map_dict[int(bbox_id)] not in bbox_lst:
                    bbox_lst.append(map_dict[int(bbox_id)])
                    k_lst.append(map_bbox2kidx[int(bbox_id)])
            bbox_idx = torch.tensor(bbox_lst)
            k_idx = torch.tensor(k_lst)
            tgt_bbox_res = tgt_bbox_unique
        else:
            tgt_bbox_res = tgt_bbox
        bbox_idx = bbox_idx.to(tgt_bbox.device)

        return tgt_bbox_res, k_idx, bbox_idx

    @torch.no_grad()
    def forward(self, outputs, targets, ent_match_indices, log=False):
        bs, num_queries = outputs["pred_rel_logits"].shape[:2]

        return_list = []

        for batch_idx in range(bs):

            tgt_act = targets[batch_idx]["rel_label_no_mask"] # (num_pair_boxes, 29)
            tgt_sum = (tgt_act.sum(dim=-1)).unsqueeze(0)

            # Concat target label
            tgt_hids = targets[batch_idx]["gt_rel_pair_tensor"][:, 0]
            tgt_oids = targets[batch_idx]["gt_rel_pair_tensor"][:, 1]

            ent_match_idx = ent_match_indices[batch_idx]

            match_dict = {}
            for pred_i, gt_i in zip(ent_match_idx[0], ent_match_idx[1]):
                match_dict[gt_i.item()] = pred_i.item()

            tgt_hids_pred_ent_id = torch.zeros_like(tgt_hids)
            tgt_oids_pred_ent_id = torch.zeros_like(tgt_hids)
            for i, _ in enumerate(tgt_hids):
                tgt_hids_pred_ent_id[i] = match_dict[tgt_hids[i].item()]
                tgt_oids_pred_ent_id[i] = match_dict[tgt_oids[i].item()]

            tgt_hids = tgt_hids_pred_ent_id
            tgt_oids = tgt_oids_pred_ent_id

            targets[batch_idx]['ent_id_idxs'] = (tgt_hids, tgt_oids)

            out_hprob = outputs["pred_hidx"][batch_idx].softmax(-1)
            out_oprob = outputs["pred_oidx"][batch_idx].softmax(-1)

            cost_hclass = 1 - out_hprob[:, tgt_hids] # [batch_size * num_queries, detr.num_queries+1]
            cost_oclass = 1 - out_oprob[:, tgt_oids] # [batch_size * num_queries, detr.num_queries+1]

            # tgt_act = torch.cat([tgt_act, torch.zeros(tgt_act.shape[0]).unsqueeze(-1).to(tgt_act.device)], dim=-1)
            # cost_pos_act = (-torch.matmul(out_act, tgt_act.t().float())) / tgt_sum
            # cost_neg_act = (torch.matmul(out_act, (~tgt_act.bool()).type(torch.int64).t().float())) / (~tgt_act.bool()).type(torch.int64).sum(dim=-1).unsqueeze(0)
            # cost_action = cost_pos_act + cost_neg_act

            tgt_rel_labels = tgt_act
            pred_rel_prob = torch.sigmoid(
                outputs["pred_rel_logits"][batch_idx]
            )  # [batch_size * num_queries, num_classes]


            pos_cost_class = 1 - pred_rel_prob[:, tgt_rel_labels - 1]
            pred_rel_prob_tmp = pred_rel_prob.clone()
            pred_rel_prob_tmp[:, tgt_rel_labels - 1] = 0
            neg_cost_class = pred_rel_prob_tmp.mean(1).unsqueeze(1)

            cost_class = pos_cost_class + neg_cost_class

            h_cost = self.cost_hbox * cost_hclass
            o_cost = self.cost_obox * cost_oclass
            act_cost = self.cost_action * cost_class

            C = h_cost + o_cost + act_cost
            C = C.view(num_queries, -1).cpu()

            return_list.append(linear_sum_assignment(C))


        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in return_list], targets


class HOTRRelSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, eos_coef, HOI_losses, HOI_matcher):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.entities_num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.rel_num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.eos_coef=eos_coef

        self.HOI_losses = HOI_losses
        self.HOI_matcher = HOI_matcher

        empty_weight = torch.ones(self.rel_num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)



    #######################################################################################################################
    # * HOTR Losses
    #######################################################################################################################
    # >>> HOI Losses 1 : HO Pointer
    def loss_pair_labels(self, outputs, targets, hoi_indices, num_boxes, log=False):
        assert ('pred_hidx' in outputs and 'pred_oidx' in outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        src_hidx = outputs['pred_hidx']
        src_oidx = outputs['pred_oidx']

        idx = self._get_src_permutation_idx(hoi_indices)

        target_hidx_classes = torch.full(src_hidx.shape[:2], -1, dtype=torch.int64, device=src_hidx.device)
        target_oidx_classes = torch.full(src_oidx.shape[:2], -1, dtype=torch.int64, device=src_oidx.device)

        # O Pointer loss
        # H Pointer loss 
        for batch_idx, (tgt, (pred_idx, gt_idx)) in enumerate(zip(targets, hoi_indices)):
            target_hidx = tgt["ent_id_idxs"][0][gt_idx]
            target_hidx_classes[batch_idx, pred_idx] = target_hidx
            target_oidx = tgt["ent_id_idxs"][1][gt_idx]
            target_oidx_classes[batch_idx, pred_idx] = target_oidx

        # N, C, WH -> N, WH
        loss_h = F.cross_entropy(src_hidx.transpose(1, 2), target_hidx_classes, ignore_index=-1)
        loss_o = F.cross_entropy(src_oidx.transpose(1, 2), target_oidx_classes, ignore_index=-1)

        losses = {'loss_hidx': loss_h, 'loss_oidx': loss_o}

        return losses

    # >>> HOI Losses 2 : pair actions
    def loss_pair_actions(self, outputs, targets, indices, num_gt_rel, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rel_logits" in outputs
        del num_gt_rel

        src_logits = outputs["pred_rel_logits"]

        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["rel_labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes_ref = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )
        target_classes_ref[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.rel_num_classes + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )

        valid_indx = target_classes_ref >= 0

        # build onehot labels
        target_classes_onehot.scatter_(2, target_classes_ref.unsqueeze(-1), 1)

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

    # HOI Losses 3 : action targets
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
            target_classes = torch.full(
                src_logits.shape[:2],
                src_logits.shape[-1] - 1,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o

            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.entities_empty_weight,
            )

            losses = {f"loss_aux_{role_name}_entities_labels_ce": loss_ce}

            if log:
                # only evaluate the foreground class entities
                losses[f"aux_{role_name}_entities_class_error"] = (
                        100 - accuracy(src_logits[idx][:, :-1], target_classes_o)[0]
                )
            return losses

        src_obj_logits = outputs["pred_rel_obj_logits"]
        src_sub_logits = outputs["pred_rel_sub_logits"]

        losses = cal_entities_class_loss(src_obj_logits, "obj")
        losses.update(cal_entities_class_loss(src_sub_logits, "sub"))

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # *****************************************************************************
    # >>> DETR Losses
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # >>> HOTR Losses
    def get_HOI_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'pair_labels': self.loss_pair_labels,
            'pair_actions': self.loss_pair_actions,
            'pair_targets': self.loss_aux_entities_labels
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    # *****************************************************************************

    def forward(self, outputs, targets, ent_index, log=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}

        # Retrieve the matching between the outputs of the last layer and the targets
        input_targets = [copy.deepcopy(target) for target in targets]
        hoi_indices, hoi_targets = self.HOI_matcher(outputs_without_aux, input_targets, ent_index, log)


        num_gt_rel = sum(len(t["rel_labels"]) for t in targets)
        num_gt_rel = torch.as_tensor(
            [num_gt_rel], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(num_gt_rel)
        num_gt_rel = torch.clamp(num_gt_rel / comm.get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        losses = {}
        # HOI detection losses
        if self.HOI_losses is not None:
            for loss in self.HOI_losses:
                losses.update(self.get_HOI_loss(loss, outputs, hoi_targets, hoi_indices, num_gt_rel))

            if 'hoi_aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                    input_targets = [copy.deepcopy(target) for target in targets]
                    hoi_indices, targets_for_aux = self.HOI_matcher(aux_outputs, input_targets, ent_index, log)
                    for loss in self.HOI_losses:
                        kwargs = {}
                        if loss == 'pair_targets': kwargs = {'log': False} # Logging is enabled only for the last layer
                        l_dict = self.get_HOI_loss(loss, aux_outputs, hoi_targets, hoi_indices, num_gt_rel, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses


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


def build_hoi_matcher(args):
    return HungarianPairMatcher(args)
