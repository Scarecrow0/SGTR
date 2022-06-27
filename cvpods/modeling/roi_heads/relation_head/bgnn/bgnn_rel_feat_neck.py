import copy

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.data.datasets.builtin_meta import get_dataset_statistics
from cvpods.structures.relationship import Relationships
from .rce_module import RelAwareRelFeature
from cvpods.evaluation.boxlist import BoxList

from cvpods.modeling.roi_heads.relation_head.rel_feat_neck import (
    PairwiseFeatureExtractor,
    RelationshipFeatureNeck,
    cat,
)


def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor


def make_fc(dim_in, hidden_dim, use_gn=False):
    """
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    """
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )


class MessagePassingUnit_v2(nn.Module):
    def __init__(self, input_dim, filter_dim=128):
        super(MessagePassingUnit_v2, self).__init__()
        self.w = nn.Linear(input_dim, filter_dim, bias=True)
        self.fea_size = input_dim
        self.filter_size = filter_dim

    def forward(self, unary_term, pair_term):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        # print '[unary_term, pair_term]', [unary_term, pair_term]
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = torch.sigmoid(gate.sum(1))
        # print 'gate', gate
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])

        return output, gate


def reverse_sigmoid(x):
    new_x = x.clone()
    new_x[x > 0.999] = x[x > 0.999] - (x[x > 0.999].clone().detach() - 0.999)
    new_x[x < 0.001] = x[x < 0.001] + (-x[x < 0.001].clone().detach() + 0.001)
    return torch.log((new_x) / (1 - (new_x)))


class MessagePassingUnit_v1(nn.Module):
    def __init__(self, input_dim, filter_dim=64):
        """

        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnit_v1, self).__init__()
        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )

        self.fea_size = input_dim
        self.filter_size = filter_dim

        self.gate_weight = nn.Parameter(torch.Tensor([0.5,]), requires_grad=True,)
        self.aux_gate_weight = nn.Parameter(torch.Tensor([0.5,]), requires_grad=True,)

    def forward(self, unary_term, pair_term, aux_gate=None):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        paired_feats = torch.cat([unary_term, pair_term], 1)

        gate = torch.sigmoid(self.w(paired_feats))
        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes
        if aux_gate is not None:
            # sigmoid_reverse_aux_gate = reverse_sigmoid(aux_gate)
            # sigmoid_reverse_gate = reverse_sigmoid(gate)
            # gate = (
            #     self.gate_weight * sigmoid_reverse_gate
            #     + self.aux_gate_weight * sigmoid_reverse_aux_gate
            # )
            # gate = torch.sigmoid(gate)

            gate = gate * aux_gate
        # print 'gate', gate
        output = pair_term * gate.view(-1, 1).expand(
            gate.size()[0], pair_term.size()[1]
        )

        return output, gate


class MessagePassingUnitGatingWithRelnessLogits(nn.Module):
    def __init__(
        self, input_dim, auxiliary_dim, use_auxiliary_gate_weight=False, filter_dim=64
    ):
        """

        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnitGatingWithRelnessLogits, self).__init__()
        self.auxiliary_dim = auxiliary_dim

        self.w_aux = nn.Sequential(
            nn.LayerNorm(self.auxiliary_dim),
            nn.ReLU(),
            nn.Linear(self.auxiliary_dim, 8, bias=True),
        )

        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )
        if use_auxiliary_gate_weight:
            self.aux_gate_weight = nn.Parameter(
                torch.Tensor([0.33,]), requires_grad=True,
            )
            self.gate_weight = nn.Parameter(torch.Tensor([0.33,]), requires_grad=True,)
            self.aux_score_weight = nn.Parameter(
                torch.Tensor([0.33,]), requires_grad=True,
            )
        else:
            self.aux_gate_weight = nn.Parameter(
                torch.Tensor([0.5,]), requires_grad=True,
            )
            self.gate_weight = nn.Parameter(torch.Tensor([0.5,]), requires_grad=True,)
        self.use_auxiliary_gate_weight = use_auxiliary_gate_weight

        self.fea_size = input_dim
        self.filter_size = filter_dim

    def forward(
        self, unary_term, pair_term, auxiliary_term, auxiliary_gating_weight=None
    ):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        paired_feats = F.relu(torch.cat([unary_term, pair_term,], 1,))

        gate = torch.sigmoid(self.w(paired_feats))
        aux_gate = torch.sigmoid(self.w_aux(auxiliary_term))

        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes
        if aux_gate.shape[1] > 1:
            aux_gate = aux_gate.mean(1)
        aux_gate = squeeze_tensor(aux_gate)
        gate = squeeze_tensor(gate)

        gate = self.gate_weight * reverse_sigmoid(
            gate
        ) + self.aux_gate_weight * reverse_sigmoid(aux_gate)

        if self.use_auxiliary_gate_weight:
            assert auxiliary_gating_weight is not None
            # sigmoid_reverse_gate = reverse_sigmoid(auxiliary_gating_weight)
            # gate += self.aux_score_weight * sigmoid_reverse_gate

            gate = torch.sigmoid(gate)
            gate = gate * auxiliary_gating_weight
        else:
            gate = torch.sigmoid(gate)
        # print 'gate', gate
        output = pair_term * gate.view(-1, 1).expand(gate.shape[0], pair_term.shape[1])

        return output, gate


class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MessageFusion, self).__init__()
        self.wih = nn.Linear(input_dim, input_dim, bias=True)
        self.whh = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


class LearnableRelatednessGating(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(LearnableRelatednessGating, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=False)

    def forward(self, relness):
        relness = torch.clamp(
            self.alpha * relness - self.alpha * self.beta, min=0, max=1.0
        )
        return relness


class BGNNContext(nn.Module):
    def __init__(
        self,
        cfg,
        rel_in_channels,
        ent_in_channels,
        hidden_dim=1024,
        num_iter=2,
        dropout=False,
        gate_width=128,
        use_kernel_function=False,
    ):
        super(BGNNContext, self).__init__()

        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.rel_pooling_dim = rel_in_channels
        self.ent_pooling_dim = ent_in_channels

        self.update_step = num_iter

        if self.update_step < 1:
            print(
                "WARNING: the update_step should be greater than 0, current: ",
                +self.update_step,
            )

        self.rel_aware_on = (
            self.cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE
        )

        self.num_rel_cls = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.relness_weighting_mp = False
        self.gating_with_relness_logits = False
        self.filter_the_mp_instance = False
        self.relation_conf_aware_models = None
        self.apply_gt_for_rel_conf = False

        self.mp_pair_refine_iter = 1

        self.vail_pair_num = cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_VALID_PAIRS_NUM

        if self.rel_aware_on:
            self.init_conf_module(cfg)

        #######################
        # decrease the dimension before mp
        self.obj_downdim_fc = nn.Sequential(
            make_fc(self.ent_pooling_dim, self.hidden_dim), nn.ReLU(True),
        )
        self.rel_downdim_fc = nn.Sequential(
            make_fc(self.rel_pooling_dim, self.hidden_dim), nn.ReLU(True),
        )

        self.obj_pair2rel_fuse = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            make_fc(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )

        self.padding_feature = nn.Parameter(
            torch.zeros((self.hidden_dim)), requires_grad=False
        )

        self.init_graph_module(cfg, num_iter, dropout, gate_width, use_kernel_function)

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val

    def normalize(self, each_img_relness, selected_rel_prop_pairs_idx):

        if len(squeeze_tensor(torch.nonzero(each_img_relness != 1.0))) > 10:
            select_relness_for_minmax = each_img_relness[selected_rel_prop_pairs_idx]
            curr_relness_max = select_relness_for_minmax.detach()[
                int(len(select_relness_for_minmax) * 0.05) :
            ].max()
            curr_relness_min = select_relness_for_minmax.detach().min()

            min_val = self.min_relness.data * 0.7 + curr_relness_min * 0.3
            max_val = self.max_relness.data * 0.7 + curr_relness_max * 0.3

            if self.training:
                # moving average for the relness scores normalization
                self.min_relness.data = (
                    self.min_relness.data * 0.9 + curr_relness_min * 0.1
                )
                self.max_relness.data = (
                    self.max_relness.data * 0.9 + curr_relness_max * 0.1
                )

        else:
            min_val = self.min_relness
            max_val = self.max_relness

        def minmax_norm(data, min, max):
            return (data - min) / (max - min + 1e-5)

        # apply on all non 1.0 relness scores
        each_img_relness[each_img_relness != 1.0] = torch.clamp(
            minmax_norm(each_img_relness[each_img_relness != 1.0], min_val, max_val),
            max=1.0,
            min=0.0,
        )

        return each_img_relness

    def ranking_minmax_recalibration(
        self, each_img_relness, selected_rel_prop_pairs_idx
    ):

        # normalize the relness score
        each_img_relness = self.normalize(each_img_relness, selected_rel_prop_pairs_idx)

        # take the top 10% pairs set as the must keep relationship by set it relness into 1.0
        total_rel_num = len(selected_rel_prop_pairs_idx)
        each_img_relness[selected_rel_prop_pairs_idx[: int(total_rel_num * 0.1)]] += (
            1.0
            - each_img_relness[selected_rel_prop_pairs_idx[: int(total_rel_num * 0.1)]]
        )

        return each_img_relness

    def relness_score_recalibration(
        self, each_img_relness, selected_rel_prop_pairs_idx
    ):
        if self.relness_score_recalibration_method == "minmax":
            each_img_relness = self.ranking_minmax_recalibration(
                each_img_relness, selected_rel_prop_pairs_idx
            )
        elif self.relness_score_recalibration_method == "learnable_scaling":

            each_img_relness = self.learnable_relness_score_gating_recalibration(
                each_img_relness
            )
        return each_img_relness

    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, relatedness):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:
            ALL RETURN THINGS ARE BATCH-WISE CONCATENATED

            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            selected_relness,
                the relatness score for selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
            selected_rel_prop_pairs_idx
                the relationship proposal id that selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
        """
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        rel_prop_pairs_relness_batch = []

        for idx, (prop, rel_ind_i) in enumerate(zip(proposals, rel_pair_idxs,)):
            if self.filter_the_mp_instance:
                assert relatedness is not None
                related_matrix = relatedness[idx]
                rel_prop_pairs_relness = related_matrix[
                    rel_ind_i[:, 0], rel_ind_i[:, 1]
                ]

                det_score = prop.get_field("pred_scores")

                rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)

        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        # only message passing on valid pairs

        if len(rel_prop_pairs_relness_batch) != 0:

            if self.rel_aware_on:
                offset = 0
                rel_prop_pairs_relness_sorted_idx = []
                rel_prop_pairs_relness_batch_update = []
                for idx, each_img_relness in enumerate(rel_prop_pairs_relness_batch):

                    (
                        selected_rel_prop_pairs_relness,
                        selected_rel_prop_pairs_idx,
                    ) = torch.sort(each_img_relness, descending=True)

                    if self.apply_gt_for_rel_conf:
                        # add the non-GT rel pair dynamically according to the GT rel num
                        gt_rel_idx = squeeze_tensor(
                            torch.nonzero(selected_rel_prop_pairs_relness == 1.0)
                        )
                        pred_rel_idx = squeeze_tensor(
                            torch.nonzero(selected_rel_prop_pairs_relness < 1.0)
                        )
                        pred_rel_num = int(len(gt_rel_idx) * 0.2)
                        pred_rel_num = (
                            pred_rel_num
                            if pred_rel_num < len(pred_rel_idx)
                            else len(pred_rel_idx)
                        )
                        pred_rel_num = pred_rel_num if pred_rel_num > 0 else 5
                        selected_rel_prop_pairs_idx = torch.cat(
                            (
                                selected_rel_prop_pairs_idx[gt_rel_idx],
                                selected_rel_prop_pairs_idx[
                                    pred_rel_idx[:pred_rel_num]
                                ],
                            )
                        )
                    else:
                        # recaliberating the relationship confidence for weighting
                        selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[
                            : self.vail_pair_num
                        ]

                        if (
                            self.relness_weighting_mp
                            and not self.pretrain_pre_clser_mode
                        ):
                            each_img_relness = self.relness_score_recalibration(
                                each_img_relness, selected_rel_prop_pairs_idx
                            )

                            selected_rel_prop_pairs_idx = squeeze_tensor(
                                torch.nonzero(each_img_relness > 0.0001)
                            )

                    rel_prop_pairs_relness_batch_update.append(each_img_relness)

                    rel_prop_pairs_relness_sorted_idx.append(
                        selected_rel_prop_pairs_idx + offset
                    )
                    offset += len(each_img_relness)

                selected_rel_prop_pairs_idx = torch.cat(
                    rel_prop_pairs_relness_sorted_idx, 0
                )
                rel_prop_pairs_relness_batch_cat = torch.cat(
                    rel_prop_pairs_relness_batch_update, 0
                )

            subj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 0],
                selected_rel_prop_pairs_idx,
            ] = 1
            obj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 1],
                selected_rel_prop_pairs_idx,
            ] = 1
            selected_relness = rel_prop_pairs_relness_batch_cat
        else:
            # or all relationship pairs
            selected_rel_prop_pairs_idx = torch.arange(
                len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device
            )
            selected_relness = None
            subj_pred_map.scatter_(
                0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1
            )
            obj_pred_map.scatter_(
                0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1
            )
        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map,
            selected_relness,
            selected_rel_prop_pairs_idx,
        )

    # Here, we do all the operations out of loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    def prepare_message(
        self,
        target_features,
        source_features,
        select_mat,
        gate_module,
        relness_scores=None,
        relness_logits=None,
    ):
        """
        generate the message from the source nodes for the following merge operations.

        Then the message passing process can be
        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_inst, rel_pair_num)
        :param gate_module:
        :param relness_scores: (num_rel, )
        :param relness_logit (num_rel, num_rel_category)

        :return: messages representation: (num_inst, dim)
        """
        feature_data = []

        if select_mat.sum() == 0:
            temp = torch.zeros(
                (target_features.size()[1:]),
                requires_grad=True,
                dtype=target_features.dtype,
                device=target_features.dtype,
            )
            feature_data = torch.stack(temp, 0)
        else:
            transfer_list = torch.nonzero(select_mat > 0)
            source_indices = transfer_list[:, 1]
            target_indices = transfer_list[:, 0]
            source_f = torch.index_select(source_features, 0, source_indices)
            target_f = torch.index_select(target_features, 0, target_indices)

            select_relness = relness_scores[source_indices]

            if self.gating_with_relness_logits:
                assert relness_logits is not None

                # relness_dist =  relness_logits
                select_relness_dist = torch.sigmoid(relness_logits[source_indices])

                if self.relness_weighting_mp:
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness_dist, select_relness
                    )
                else:
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness_dist
                    )
            else:
                if self.relness_weighting_mp:
                    select_relness = relness_scores[transfer_list[:, 1]]
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness
                    )
                else:
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f
                    )
            aggregator_matrix = torch.zeros(
                (target_features.shape[0], transferred_features.shape[0]),
                dtype=weighting_gate.dtype,
                device=weighting_gate.device,
            )

            for f_id in range(target_features.shape[0]):
                if select_mat[f_id, :].data.sum() > 0:
                    # average from the multiple sources
                    feature_indices = squeeze_tensor(
                        torch.nonzero(transfer_list[:, 0] == f_id)
                    )  # obtain source_relevant_idx
                    # (target, source_relevant_idx)
                    aggregator_matrix[f_id, feature_indices] = 1
            # (target, source_relevant_idx) @ (source_relevant_idx, feat-dim) => (target, feat-dim)
            aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
            avg_factor = aggregator_matrix.sum(dim=1)
            vaild_aggregate_idx = avg_factor != 0
            avg_factor = avg_factor.unsqueeze(1).expand(
                avg_factor.shape[0], aggregate_feat.shape[1]
            )
            aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

            feature_data = aggregate_feat
        return feature_data

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs):
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.hidden_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)

        obj_pair_feat4rel_rep = torch.cat(
            (head_rep[rel_pair_idxs[:, 0]], tail_rep[rel_pair_idxs[:, 1]]), dim=-1
        )

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(
            obj_pair_feat4rel_rep
        )  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(
        self,
        augment_obj_feat,
        rel_feats,
        ent_proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None,
    ):
        """

        :param inst_features: instance_num, pooling_dim
        :param rel_union_features:  rel_num, pooling_dim
        :param proposals: instance proposals
        :param rel_pair_inds: relaion pair indices list(tensor)
        :param rel_binarys: [num_prop, num_prop] the relatedness of each pair of boxes
        :return:
        """

        num_inst_proposals = [len(b) for b in ent_proposals]

        relatedness_each_iters = []
        refine_rel_feats_each_iters = [rel_feats]
        refine_ent_feats_each_iters = [augment_obj_feat]
        conf_est_logits_each_iter = []
        self.forward_time += 1

        if self.forward_time > 1000 or not self.training:
            if self.pretrain_pre_clser_mode:
                print("stop RCE pretraining")
                self.pretrain_pre_clser_mode = False
        else:
            self.pretrain_pre_clser_mode = True


        for refine_iter in range(self.mp_pair_refine_iter):
            pre_cls_logits = None

            pre_cls_logits, relatedness_scores = self.rel_confence_estimation(
                ent_proposals,
                rel_pair_inds,
                rel_gt_binarys,
                refine_rel_feats_each_iters,
                refine_iter,
            )

            conf_est_logits_each_iter.append(pre_cls_logits)
            relatedness_each_iters.append(relatedness_scores)

            # build up list for massage passing process
            inst_feature4iter = [
                self.obj_downdim_fc(augment_obj_feat),
            ]
            rel_feature4iter = [
                self.rel_downdim_fc(rel_feats),
            ]

            valid_inst_idx = []
            if self.filter_the_mp_instance:
                for p in ent_proposals:
                    valid_inst_idx.append(p.get_field("pred_scores") > 0.03)

            if len(valid_inst_idx) > 0:
                valid_inst_idx = torch.cat(valid_inst_idx, 0)
            else:
                valid_inst_idx = torch.zeros(0)


            if self.pretrain_pre_clser_mode:
                #  directly return without graph building
                refined_inst_features = inst_feature4iter[-1]
                refined_rel_features = rel_feature4iter[-1]

                refine_ent_feats_each_iters.append(refined_inst_features)
                refine_rel_feats_each_iters.append(refined_rel_features)
            
                continue

            else:

                (
                    batchwise_rel_pair_inds,
                    subj_pred_map,
                    obj_pred_map,
                    relness_scores,
                    selected_rel_prop_pairs_idx,
                ) = self._prepare_adjacency_matrix(
                    ent_proposals, rel_pair_inds, relatedness_each_iters[-1]
                )

                if (
                    len(squeeze_tensor(torch.nonzero(valid_inst_idx))) < 1
                    or len(squeeze_tensor(torch.nonzero(batchwise_rel_pair_inds))) < 1
                    or len(squeeze_tensor(torch.nonzero(subj_pred_map))) < 1
                    or len(squeeze_tensor(torch.nonzero(obj_pred_map))) < 1
                    or self.pretrain_pre_clser_mode
                ):  # directly return, no mp process

                    # print("valid_inst_idx", valid_inst_idx.nonzero())
                    # print("batchwise_rel_pair_inds", batchwise_rel_pair_inds.nonzero())
                    # print("subj_pred_map", subj_pred_map.nonzero())
                    # print("obj_pred_map", obj_pred_map.nonzero())
                    # print("WARNING: all graph nodes has been filtered out. ")

                    refined_inst_features = inst_feature4iter[-1]
                    refined_rel_features = rel_feature4iter[-1]

                    refine_ent_feats_each_iters.append(refined_inst_features)
                    refine_rel_feats_each_iters.append(refined_rel_features)

                    continue

            inst_feature4iter, rel_feature4iter = self.graph_msp(
                pre_cls_logits,
                inst_feature4iter,
                rel_feature4iter,
                valid_inst_idx,
                batchwise_rel_pair_inds,
                subj_pred_map,
                obj_pred_map,
                relness_scores,
            )

            refined_inst_features = inst_feature4iter[-1]
            refined_rel_features = rel_feature4iter[-1]

            refine_ent_feats_each_iters.append(refined_inst_features)
            # fuse the entities features to the relationship feature for classification
            # paired_inst_feats = self.pairwise_rel_features(refined_inst_features, batchwise_rel_pair_inds)
            # refine_rel_feats_each_iters.append(paired_inst_feats + refined_rel_features)

            # directly use the msp features
            refine_rel_feats_each_iters.append(refined_rel_features)

        if (
            len(relatedness_each_iters) > 0 and not self.training
        ):  # todo why disabled in training??
            relatedness_each_iters = torch.stack(
                [torch.stack(each) for each in relatedness_each_iters]
            )
            # bsz, num_obj, num_obj, iter_num
            relatedness_each_iters = relatedness_each_iters.permute(1, 2, 3, 0)
        else:
            relatedness_each_iters = None

        if len(conf_est_logits_each_iter) == 0:
            conf_est_logits_each_iter = None

        return (
            refine_ent_feats_each_iters[-1],
            refine_rel_feats_each_iters[-1],
            conf_est_logits_each_iter,
            relatedness_each_iters,
        )

    def graph_msp(
        self,
        pre_cls_logits,
        inst_feature4iter,
        rel_feature4iter,
        valid_inst_idx,
        batchwise_rel_pair_inds,
        subj_pred_map,
        obj_pred_map,
        relness_scores,
    ):
        # graph module
        for t in range(self.update_step):
            param_idx = 0
            if not self.share_parameters_each_iter:
                param_idx = t
            """update object features pass message from the predicates to instances"""
            object_sub = self.prepare_message(
                inst_feature4iter[t],
                rel_feature4iter[t],
                subj_pred_map,
                self.gate_pred2sub[param_idx],
                relness_scores=relness_scores,
                relness_logits=pre_cls_logits,
            )
            object_obj = self.prepare_message(
                inst_feature4iter[t],
                rel_feature4iter[t],
                obj_pred_map,
                self.gate_pred2obj[param_idx],
                relness_scores=relness_scores,
                relness_logits=pre_cls_logits,
            )

            GRU_input_feature_object = (object_sub + object_obj) / 2.0
            inst_feature4iter.append(
                inst_feature4iter[t]
                + self.object_msg_fusion[param_idx](
                    GRU_input_feature_object, inst_feature4iter[t]
                )
            )

            """update predicate features from entities features"""
            indices_sub = batchwise_rel_pair_inds[:, 0]
            indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1

            if self.filter_the_mp_instance:
                indices_sub, indices_obj = self.rel_conf_gated_msp(
                    inst_feature4iter,
                    rel_feature4iter,
                    valid_inst_idx,
                    t,
                    param_idx,
                    indices_sub,
                    indices_obj,
                )
            else:
                # obj to pred on all pairs
                feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
                feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
                phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
                    rel_feature4iter[t], feat_sub2pred
                )
                phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
                    rel_feature4iter[t], feat_obj2pred
                )
                GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
                rel_feature4iter.append(
                    rel_feature4iter[t]
                    + self.pred_msg_fusion[param_idx](
                        GRU_input_feature_phrase, rel_feature4iter[t]
                    )
                )

        return inst_feature4iter, rel_feature4iter

    def rel_confence_estimation(
        self,
        proposals,
        rel_pair_inds,
        rel_gt_binarys,
        refine_rel_feats_each_iters,
        refine_iter,
    ):
        pred_relatedness_scores = None
        relatedness_scores = None
        if self.rel_aware_on:
            # input_features = refine_ent_feats_each_iters[-1]
            input_features = refine_rel_feats_each_iters[-1]
            if not self.shared_pre_rel_classifier:
                (
                    pre_cls_logits,
                    pred_relatedness_scores,
                ) = self.relation_conf_aware_models[refine_iter](
                    input_features, proposals, rel_pair_inds
                )
            else:
                (
                    pre_cls_logits,
                    pred_relatedness_scores,
                ) = self.relation_conf_aware_models(
                    input_features, proposals, rel_pair_inds
                )

        # apply GT
        relatedness_scores = pred_relatedness_scores
        if self.apply_gt_for_rel_conf and rel_gt_binarys is not None:
            ref_relatedness = rel_gt_binarys.clone()

            if pred_relatedness_scores is None:
                relatedness_scores = ref_relatedness
            else:
                relatedness_scores = pred_relatedness_scores
                for idx, ref_rel in enumerate(ref_relatedness):
                    gt_rel_idx = torch.nonzero(ref_rel)
                    relatedness_scores[idx][gt_rel_idx[:, 0], gt_rel_idx[:, 1]] = 1.0
        return pre_cls_logits, relatedness_scores

    def rel_conf_gated_msp(
        self,
        inst_feature4iter,
        rel_feature4iter,
        valid_inst_idx,
        t,
        param_idx,
        indices_sub,
        indices_obj,
    ):
        # here we only pass massage from the fg boxes to the predicates
        valid_sub_inst_in_pairs = valid_inst_idx[indices_sub]
        valid_obj_inst_in_pairs = valid_inst_idx[indices_obj]
        valid_inst_pair_inds = (valid_sub_inst_in_pairs) & (valid_obj_inst_in_pairs)
        # num_rel(valid sub inst), 1 Boolean tensor
        # num_rel(valid sub inst), 1
        indices_sub = indices_sub[valid_inst_pair_inds]
        # num_rel(valid obj inst), 1
        indices_obj = indices_obj[valid_inst_pair_inds]

        feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
        feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
        # num_rel(valid obj inst), hidden_dim
        valid_pairs_rel_feats = torch.index_select(
            rel_feature4iter[t], 0, squeeze_tensor(torch.nonzero(valid_inst_pair_inds)),
        )
        phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
            valid_pairs_rel_feats, feat_sub2pred
        )
        phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
            valid_pairs_rel_feats, feat_obj2pred
        )
        GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
        next_stp_rel_feature4iter = self.pred_msg_fusion[param_idx](
            GRU_input_feature_phrase, valid_pairs_rel_feats
        )

        # only update valid pairs feature, others remain as initial value
        padded_next_stp_rel_feats = rel_feature4iter[t].clone()
        padded_next_stp_rel_feats[valid_inst_pair_inds] += next_stp_rel_feature4iter

        rel_feature4iter.append(padded_next_stp_rel_feats)
        return indices_sub, indices_obj

    def init_conf_module(self, cfg):
        #####  build up the relationship aware modules
        self.mp_pair_refine_iter = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.ITERATE_MP_PAIR_REFINE
        )
        assert self.mp_pair_refine_iter > 0

        self.shared_pre_rel_classifier = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SHARE_RELATED_MODEL_ACROSS_REFINE_ITER
        )

        if self.mp_pair_refine_iter <= 1:
            self.shared_pre_rel_classifier = False

        if not self.shared_pre_rel_classifier:
            self.relation_conf_aware_models = nn.ModuleList()
            for ii in range(self.mp_pair_refine_iter):
                if ii == 0:
                    input_dim = self.rel_pooling_dim
                else:
                    input_dim = self.hidden_dim
                self.relation_conf_aware_models.append(RelAwareRelFeature(cfg, input_dim,))
        else:
            input_dim = self.rel_pooling_dim
            self.relation_conf_aware_models = RelAwareRelFeature(cfg, input_dim,)
        self.pretrain_pre_clser_mode = False

        ######  relationship confidence recalibration

        self.apply_gt_for_rel_conf = (
            self.cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT
        )

        self.gating_with_relness_logits = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GATING_WITH_RELNESS_LOGITS
        )
        self.relness_weighting_mp = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELNESS_MP_WEIGHTING
        )
        # 'minmax',  'learnable_scaling'
        self.relness_score_recalibration_method = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELNESS_MP_WEIGHTING_SCORE_RECALIBRATION_METHOD
        )

        if self.relness_score_recalibration_method == "learnable_scaling":
            self.learnable_relness_score_gating_recalibration = (
                LearnableRelatednessGating()
            )
        elif self.relness_score_recalibration_method == "minmax":
            self.min_relness = nn.Parameter(torch.Tensor([1e-5,]), requires_grad=False,)
            self.max_relness = nn.Parameter(torch.Tensor([0.5,]), requires_grad=False,)
        else:
            raise ValueError(
                "Invalid relness_score_recalibration_method "
                + self.relness_score_recalibration_method
            )

        self.filter_the_mp_instance = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_ON_VALID_PAIRS
        )

    def init_graph_module(
        self, cfg, num_iter, dropout, gate_width, use_kernel_function
    ):
        if use_kernel_function:
            MessagePassingUnit = MessagePassingUnit_v2
        else:
            MessagePassingUnit = MessagePassingUnit_v1
        self.share_parameters_each_iter = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SHARE_PARAMETERS_EACH_ITER
        )

        param_set_num = num_iter
        if self.share_parameters_each_iter:
            param_set_num = 1
        self.gate_sub2pred = nn.Sequential(
            *[
                MessagePassingUnit(self.hidden_dim, gate_width)
                for _ in range(param_set_num)
            ]
        )
        self.gate_obj2pred = nn.Sequential(
            *[
                MessagePassingUnit(self.hidden_dim, gate_width)
                for _ in range(param_set_num)
            ]
        )
        self.gate_pred2sub = nn.Sequential(
            *[
                MessagePassingUnit(self.hidden_dim, gate_width)
                for _ in range(param_set_num)
            ]
        )
        self.gate_pred2obj = nn.Sequential(
            *[
                MessagePassingUnit(self.hidden_dim, gate_width)
                for _ in range(param_set_num)
            ]
        )

        if self.gating_with_relness_logits:
            MessagePassingUnit = MessagePassingUnitGatingWithRelnessLogits
            self.gate_pred2sub = nn.Sequential(
                *[
                    MessagePassingUnit(
                        self.hidden_dim, self.num_rel_cls, self.relness_weighting_mp
                    )
                    for _ in range(param_set_num)
                ]
            )
            self.gate_pred2obj = nn.Sequential(
                *[
                    MessagePassingUnit(
                        self.hidden_dim, self.num_rel_cls, self.relness_weighting_mp
                    )
                    for _ in range(param_set_num)
                ]
            )
        self.object_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )  #
        self.pred_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )

        self.forward_time = 0


# warpper class for BGNN feature,
class BGNNFeatureNeck(RelationshipFeatureNeck):
    def __init__(self, cfg, rel_feat_input_shape, ent_feat_input_shape):
        super(BGNNFeatureNeck, self).__init__(
            cfg, rel_feat_input_shape, ent_feat_input_shape
        )

        hidden_dim = 512
        self.bgnn_context = BGNNContext(
            cfg,
            rel_feat_input_shape,
            ent_feat_input_shape,
            hidden_dim=hidden_dim,
            num_iter=2,
        )
        self.rel_out_fc = make_fc(hidden_dim, rel_feat_input_shape)
        self.ent_out_fc = make_fc(hidden_dim, ent_feat_input_shape)
        self.output_skip_connection = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SKIP_CONNECTION_ON_OUTPUT
        )

        self._output_size = rel_feat_input_shape

    def forward(
        self, rel_proposals: Relationships, rel_features: list, ent_features: list
    ):
        """augment the relationship features with additional structure

        Args:
            rel_proposals (Relationships): [description]
            rel_features (list): [description]
            ent_features (list): [description]

        Returns:
            rel_features (list): [description]
            ent_features (list): [description]
            outputs (dict): additional intermidate output
        """

        #   pair-wise ent_features
        #   word embedding
        #   visual features
        rel_features, ent_features = self.pairwise_rel_feat_extractor(
            rel_proposals, rel_features, ent_features
        )  # batch, inst_num, dim

        #   pair wise feature attention(optional)
        rel_gt_binarys = None
        if rel_proposals[0].has_meta_info('fg_pair_matrixs'):
            rel_gt_binarys=[prop.get_meta_info("fg_pair_matrixs") for prop in rel_proposals]
        
        
        ent_features = torch.cat(ent_features)
        rel_features = torch.cat(rel_features)
        (
            ent_features_bgnn,
            rel_features_bgnn,
            conf_est_logits_each_iter,
            relatedness_each_iters,
        ) = self.bgnn_context(
            ent_features, rel_features,
            ent_proposals=[
                ent_format_transform(prop.get("instances")) for prop in rel_proposals
            ],
            rel_pair_inds=[prop.get("rel_pair_tensor") for prop in rel_proposals],
            rel_gt_binarys=rel_gt_binarys
        )
        ent_len = [len(prop.get("instances")) for prop in rel_proposals]
        rel_len = [len(prop.get("rel_pair_tensor")) for prop in rel_proposals]
        
        if self.output_skip_connection:
            ent_features = self.ent_out_fc(ent_features_bgnn) + ent_features
            rel_features = self.rel_out_fc(rel_features_bgnn) + rel_features
        else:
            ent_features = self.ent_out_fc(ent_features_bgnn)
            rel_features = self.rel_out_fc(rel_features_bgnn)

        ent_features = torch.split(ent_features, ent_len)
        rel_features = torch.split(rel_features, rel_len)
        outputs = {
            "conf_est_logits_each_iter": conf_est_logits_each_iter,
            "relatedness_each_iters": relatedness_each_iters,
        }
        return rel_features, ent_features, outputs


def ent_format_transform(pred_instances):
    bbox = pred_instances.pred_boxes.tensor
    image_height, image_width = pred_instances.image_size
    image_size = (image_width, image_height)
    prediction = BoxList(bbox, image_size, mode="xyxy")
    prediction.add_field("pred_labels", pred_instances.pred_classes)
    prediction.add_field("pred_scores", pred_instances.scores)
    prediction.add_field("pred_score_dist", pred_instances.pred_score_dist)
    return prediction


def ent_gt_format_transform(gt_instances):
    bbox = gt_instances.gt_boxes.tensor
    image_height, image_width = gt_instances.image_size
    image_size = (image_width, image_height)
    groundtruths = BoxList(bbox, image_size, mode="xyxy")
    groundtruths.add_field("labels", gt_instances.gt_classes)
    return groundtruths
