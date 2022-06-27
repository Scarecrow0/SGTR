import torch
import torch.nn.functional as F

from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_losses import RelHungarianMatcher
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_losses import RelSetCriterion


class AuxRelHungarianMatcher(RelHungarianMatcher):
    def __init__(
            self,
            cfg,
            cost_rel_class: float = 1.5,
            cost_rel_vec: float = 1.0,
            cost_class: float = 1.5,
            cost_bbox: float = 0.8,
            cost_giou: float = 1,
            cost_indexing: float = 0.2,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
        """
        super().__init__(cfg, cost_rel_class, cost_rel_vec,
                         cost_class, cost_bbox, cost_giou, cost_indexing)

    @torch.no_grad()
    def forward(self, outputs, targets, ent_match=True, return_init_idx=False):
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
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED and ent_match:

            (match_cost,
             detailed_cost_dict) = self.inter_vec_entities_cost_calculation(outputs, targets)
        else:
            (match_cost,
             detailed_cost_dict) = self.inter_vec_cost_calculation(outputs, targets)

        indices = self.top_score_match(match_cost, return_init_idx)

        match_cost_each_img = [c[i] for i, c in enumerate(match_cost)]

        for k in detailed_cost_dict.keys():
            detailed_cost_dict[k] = [c[i] for i, c in enumerate(detailed_cost_dict[k])]

        match_idx = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

        return match_idx, match_cost_each_img, detailed_cost_dict


class AuxRelSetCriterion(RelSetCriterion):
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
        super().__init__(cfg, num_classes, matcher, weight_dict, eos_coef, losses)



    def forward(self, outputs, targets, match_res, aux_only=False, ent_match_idx=None, entities_match_cache=None):
        """
        This performs the loss computation.

        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each
                      loss' doc
        """ 
        losses, match_res = super().forward(outputs, targets, match_res, det_match_res=ent_match_idx, 
                                                                         entities_match_cache=entities_match_cache)

        l_dict = {}
        for k, v in losses.items():
            if '/layer' not in k and aux_only:
                continue
            if self.weight_dict.get(k) is not None:
                v *= self.weight_dict[k]
            l_dict['aux_' + k] = v
        return l_dict, match_res


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
