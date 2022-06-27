# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from numpy.core.numeric import False_
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from typing import Optional, List
import datetime

from torch import Tensor

from cvpods.structures.boxes import box_cxcywh_to_xyxy
from torchvision.ops import generalized_box_iou
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.modeling.meta_arch.detr import SetCriterion, PostProcess, MLP
from cvpods.modeling.backbone.transformer import (
    Transformer,
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)


from cvpods.layers import ShapeSpec, position_encoding_dict

from cvpods.structures import ImageList, Instances, Boxes
from cvpods.structures import boxes as box_ops
from cvpods.structures.relationship import Relationships

from .loss_matcher import HungarianPairMatcher, HOTRRelSetCriterion

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes



def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)



class HOTR(nn.Module):
    def __init__(self, cfg,
                 share_enc=True,
                 pretrained_dec=True,
                 temperature=0.05,
                 hoi_aux_loss=True,
                 return_obj_class=None):
        super().__init__()

        # * Instance Transformer ---------------
        self.cfg = cfg

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        # Build Transformer
        self.ent_aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS

        self.num_ent_queries = cfg.MODEL.DETR.NUM_QUERIES

        self.num_ent_classes = cfg.MODEL.DETR.NUM_CLASSES

        self.transformer = Transformer(cfg)  # initial detr

        hidden_dim = self.transformer.d_model
        self.hidden_dim = hidden_dim

        # pre encoder modules
        backbone_out_shapes = self.backbone.output_shape()["res5"]
        self.input_proj = nn.Conv2d(
            backbone_out_shapes.channels, hidden_dim, kernel_size=1
        )
        self.position_embedding = position_encoding_dict[
            cfg.MODEL.DETR.POSITION_EMBEDDING
        ](
            num_pos_feats=hidden_dim // 2,
            temperature=cfg.MODEL.DETR.TEMPERATURE,
            normalize=True if cfg.MODEL.DETR.POSITION_EMBEDDING == "sine" else False,
            scale=None,
        )

        # post entities decoder
        # Build FFN
        self.class_embed = nn.Linear(hidden_dim, self.num_ent_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(self.num_ent_queries, hidden_dim)

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }

        losses = ["labels", "boxes", "cardinality"]
        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DETR.COST_CLASS,
            cost_bbox=cfg.MODEL.DETR.COST_BBOX,
            cost_giou=cfg.MODEL.DETR.COST_GIOU,
        )

        self.criterion = SetCriterion(
            self.num_ent_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.MODEL.DETR.EOS_COEFF,
            losses=losses,
        )


        self.post_processors = {
            "bbox": PostProcess(),
            "rel": RelPostProcess(cfg),
        }  # relationship PostProcess


        # preprocessing
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        if not cfg.MODEL.RESNETS.STRIDE_IN_1X1:
            # Custom or torch pretrain weights
            self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        else:
            # MSRA pretrain weights
            self.normalizer = lambda x: (x - pixel_mean) / pixel_std


        # --------------------------------------
        self.num_rel_queries = cfg.MODEL.REL_DETR.NUM_QUERIES
        self.num_rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # * Interaction Transformer -----------------------------------------
        self.rel_query_embed = nn.Embedding(self.num_rel_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            self.action_embed = nn.Linear(hidden_dim, self.num_rel_classes)
        else:
            self.action_embed = nn.Linear(hidden_dim, self.num_rel_classes + 1)

        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.obj_class_embed = nn.Linear(hidden_dim, self.num_ent_classes + 1)
        self.sub_class_embed = nn.Linear(hidden_dim, self.num_ent_classes + 1)
        # ------------------------------------------------------------------

        # * Transformer Options ---------------------------------------------

        # Build Object Queries
        self.rel_query_embed = nn.Embedding(self.num_rel_queries, hidden_dim)

        d_model = 256
        self.num_encoder_layers = 12
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead=8, dim_feedforward=2048, dropout=0.1, normalize_before=False
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(
            decoder_layer,
            self.num_encoder_layers,
            decoder_norm,
            return_intermediate=True,
        )

        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss

        matcher = HungarianPairMatcher(set_cost_act=1, set_cost_idx=10, set_cost_tgt=1)
        self.rel_loss = HOTRRelSetCriterion(cfg, eos_coef=0.1, 
                                            HOI_matcher=matcher, 
                                            HOI_losses=['pair_labels','pair_actions'])

        # ----------------------------------

        # visualization
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.to(self.device)


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        image_sizes = torch.stack(
            [torch.tensor(
                    [bi.get("height", img_size[0]), bi.get("width", img_size[1]),],
                    device=self.device,)
                for bi, img_size in zip(batched_inputs, images.image_sizes) ]
        )

        B, C, H, W = images.tensor.shape
        device = images.tensor.device

        mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
        spatial_shapes = []
        for img_shape, m in zip(images.image_sizes, mask):
            m[: img_shape[0], : img_shape[1]] = False
            spatial_shape = (img_shape[0], img_shape[1])
            spatial_shapes.append(spatial_shape)

        src = self.backbone(images.tensor)["res5"]
        mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).bool()[0]
        pos = self.position_embedding(src, mask)

        precompute_prop = None
        if batched_inputs[0]['relationships'].has_meta_info('precompute_prop'):
            precompute_prop = [each['relationships'].get_meta_info('precompute_prop')
                                for each in batched_inputs] # list(Instances)

        input_mem = self.input_proj(src)
        ent_hs, mem_enc = self.transformer(
            input_mem, mask, self.query_embed.weight, pos,
            enc_return_lvl=self.cfg.MODEL.REL_DETR.TRANSFORMER.SHARE_ENC_FEAT_LAYERS,
            precompute_prop=precompute_prop
            # return  memory for selected encoder layer
        )

        outputs_class = self.class_embed(ent_hs)  # B, N, Dim
        outputs_coord = self.bbox_embed(ent_hs).sigmoid()

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        
        query_embed = self.rel_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        mask = mask.flatten(1)
        pos_embed = pos.flatten(2).permute(2, 0, 1)
        mem_enc_flat = mem_enc.flatten(2).permute(2, 0, 1)
        interaction_hs = self.interaction_decoder(tgt, mem_enc_flat, memory_key_padding_mask=mask, 
                                                    pos=pos_embed, query_pos=query_embed)
        interaction_hs = interaction_hs.transpose(2, 1)

        # [HO Pointers]
        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)
        ints_reprs = F.normalize(ent_hs[-1], p=2, dim=2) # take the last layer

        # the cosine distance between the two representation
        outputs_hidx = [(torch.bmm(H_Pointer_repr, ints_reprs.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, ints_reprs.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]
        
        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        # --------------------------------------------------

        # [Target Classification]

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_rel_logits": outputs_action[-1],
        }


        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = \
                self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

        if self.training:
            targets = self.convert_anno_format(batched_inputs)
            ent_loss, ent_idx = self.criterion(out, targets, with_match_idx=True)
            rel_loss = self.rel_loss(out, targets, ent_idx)
            ent_loss.update(rel_loss)
            out = ent_loss
        else: 
            ent_det_res = self.post_processors["bbox"](out, image_sizes)
            rel_det_res = self.post_processors["rel"](out, ent_det_res, image_sizes)

            out = HOTR._postprocess(
                ent_det_res, rel_det_res, batched_inputs, images.image_sizes
            )

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_rel_logits': e}
                for a, b, c, d, e in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_tgt):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_rel_logits': e, 'pred_obj_logits': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_tgt[:-1])]
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]
            boxes = box_ops.box_xyxy_to_cxcywh(
                bi["instances"].gt_boxes.tensor
                / torch.tensor([w, h, w, h], dtype=torch.float32)
            )
            target["boxes"] = boxes.to(self.device)
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor(
                [bi["height"], bi["width"]], device=self.device
            )
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)

            ####  relationship parts
            target["relationships"] = bi["relationships"].to(self.device)
            target["rel_labels"] = bi["relationships"].rel_label
            target["rel_label_no_mask"] = bi["relationships"].rel_label_no_mask

            rel_pair_tensor = bi["relationships"].rel_pair_tensor
            target["gt_rel_pair_tensor"] = rel_pair_tensor
            target["rel_vector"] = torch.cat(
                (boxes[rel_pair_tensor[:, 0], :2], boxes[rel_pair_tensor[:, 1], :2]),
                dim=1,
            ).to(
                self.device
            )  # Kx2 + K x2 => K x 4

            targets.append(target)

        return targets

    @staticmethod
    def _postprocess(ent_det_res, rel_det_res, batched_inputs, image_sizes):
        """
        dump every attributes of prediction result into the Relationships structures
        """
        # note: private function; subject to changes

        processed_results = []
        # for results_per_image, input_per_image, image_size in zip(
        for det_res_per_image, rel_res_per_img, _, image_size in zip(
                ent_det_res, rel_det_res, batched_inputs, image_sizes
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(det_res_per_image["boxes"].float())
            result.scores = det_res_per_image["scores"].float()
            result.pred_classes = det_res_per_image["labels"]
            result.pred_score_dist = det_res_per_image["prob"]

            if rel_res_per_img.get("rel_branch_box") is not None:
                rel_inst = Instances(image_size)
                rel_inst.pred_boxes = Boxes(rel_res_per_img["rel_branch_box"].float())
                rel_inst.scores = rel_res_per_img["rel_branch_score"].float()
                rel_inst.pred_classes = rel_res_per_img["rel_branch_label"]
            else:
                rel_inst = result

            pred_rel = Relationships(
                instances=rel_inst,
                rel_pair_tensor=rel_res_per_img["rel_trp"][:, :2],
                pred_rel_classs=rel_res_per_img["rel_pred_label"],  # start from 1
                pred_rel_scores=rel_res_per_img["rel_score"],
                pred_rel_trp_scores=rel_res_per_img["rel_trp_score"],
                pred_rel_dist=rel_res_per_img["pred_prob_dist"],
                pred_init_prop_idx=rel_res_per_img["init_prop_indx"],
            )

            if rel_res_per_img.get("rel_vec") is not None:
                pred_rel.pred_rel_vec = rel_res_per_img.get("rel_vec")

            if rel_res_per_img.get("pred_rel_confidence") is not None:
                pred_rel.pred_rel_confidence = rel_res_per_img[
                    "pred_rel_confidence"
                ].unsqueeze(-1)

            if rel_res_per_img.get("selected_mask") is not None:
                for k, v in rel_res_per_img.get("selected_mask").items():
                    pred_rel.__setattr__(k, v)

            if rel_res_per_img.get("pred_rel_ent_obj_box") is not None:
                pred_rel.__setattr__(
                    "pred_rel_ent_obj_box", rel_res_per_img.get("pred_rel_ent_obj_box")
                )
                pred_rel.__setattr__(
                    "pred_rel_ent_sub_box", rel_res_per_img.get("pred_rel_ent_sub_box")
                )

                for k in ['match_rel_vec_sub', 'match_rel_vec_obj', 'match_sub_conf', 'match_obj_conf',
                          'match_vec_n_conf_sub', 'match_vec_n_conf_obj', 'match_rel_sub_cls',
                          'match_rel_obj_cls', 'match_sub_giou', 'match_obj_giou',
                          "match_sub_entities_indexing", "match_obj_entities_indexing",
                          "match_sub_entities_indexing_rule", "match_obj_entities_indexing_rule",
                          'match_scr_sub', 'match_scr_obj', ]:

                    if rel_res_per_img.get(k) is not None:
                        pred_rel.set(k, rel_res_per_img.get(k))

            processed_results.append({"instances": result, "relationships": pred_rel})

        return processed_results


class RelPostProcess(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_ent_class = cfg.MODEL.DETR.NUM_CLASSES
        self.co_nms_thres = cfg.MODEL.REL_DETR.OVERLAP_THRES


    @torch.no_grad()
    def forward(self, outputs, det_res, target_sizes, max_proposal_pairs=300,):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        pred_rel_logits = outputs["pred_rel_logits"]

        device = pred_rel_logits.device

        if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            pred_rel_probs = torch.sigmoid(pred_rel_logits)
        else:
            pred_rel_probs = torch.softmax(pred_rel_logits, -1)

        # Preidction Branch for HOI detection
        """ Compute HOI triplet prediction score for V-COCO.
        Our scoring function follows the implementation details of UnionDet.
        """

        sub_match_indx_prob = F.softmax(outputs['pred_hidx'], -1)
        obj_match_indx_prob = F.softmax(outputs['pred_oidx'], -1)

        h_idx_score, h_indices = sub_match_indx_prob.max(-1)
        o_idx_score, o_indices = obj_match_indx_prob.max(-1)

        rel_proposals_predict = []
        # iterate for batch size
        for batch_ind, ent_det in enumerate(det_res):

            ent_score = ent_det['scores']
            ent_label = ent_det['labels']
            ent_box = ent_det['boxes']
            ent_box_normed = ent_det['boxes_norm']

            match_scr_sub = sub_match_indx_prob[batch_ind]
            match_scr_obj = obj_match_indx_prob[batch_ind]


            init_max_match_ent = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING


            max_match_ent = init_max_match_ent if match_scr_sub.shape[-1] > init_max_match_ent else match_scr_sub.shape[
                -1]
            rel_match_sub_scores, rel_match_sub_ids = torch.topk(match_scr_sub, max_match_ent, dim=-1)

            # num_rel_queries; num_rel_queries
            max_match_ent = init_max_match_ent if match_scr_obj.shape[-1] > init_max_match_ent else match_scr_obj.shape[
                -1]
            rel_match_obj_scores, rel_match_obj_ids = torch.topk(match_scr_obj, max_match_ent, dim=-1)


            if self.cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                pred_rel_prob = pred_rel_probs[batch_ind]
                num_q, cls_num = pred_rel_prob.shape

                pred_num_per_edge = self.cfg.MODEL.REL_DETR.NUM_PRED_EDGES

                topk = num_q * pred_num_per_edge  # num of query * pred_num_per_edge

                topk_values_all, topk_indexes_all = torch.sort(pred_rel_prob.reshape(-1), dim=-1,
                                                               descending=True)  # num_query * num_cls

                pred_rel_prob = topk_values_all[:topk]  # scores for each relationship predictions
                # (num of query * pred_num_per_edge)
                total_pred_idx = topk_indexes_all[:topk] // cls_num
                pred_rel_labels = topk_indexes_all[:topk] % cls_num
                pred_rel_labels += 1

                # =>  (num_queries * num_pred_rel,  num_group_entities)
                rel_match_sub_ids = rel_match_sub_ids[total_pred_idx]
                rel_match_obj_ids = rel_match_obj_ids[total_pred_idx]

                total_pred_idx = total_pred_idx.contiguous().unsqueeze(1).repeat(1, max_match_ent) \
                    .view(-1).unsqueeze(1)

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_prob = pred_rel_prob.reshape(-1, 1).repeat(1, max_match_ent).view(-1)
                pred_rel_labels = pred_rel_labels.reshape(-1, 1).repeat(1, max_match_ent).view(-1).unsqueeze(1)

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
                pred_rel_labels = pred_rel_labels.repeat(1, max_match_ent).view(-1).unsqueeze(1)

                total_pred_idx = torch.arange(num_rel_queries).unsqueeze(1).repeat(1, pred_num_per_edge)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                total_pred_idx = total_pred_idx.reshape(-1, 1)
                total_pred_idx = total_pred_idx.repeat(1, max_match_ent).view(-1).contiguous().unsqueeze(1)

            rel_match_sub_ids_flat = rel_match_sub_ids.view(-1).contiguous()
            rel_match_obj_ids_flat = rel_match_obj_ids.view(-1).contiguous()

            rel_trp_scores = pred_rel_prob * ent_score[rel_match_sub_ids_flat] * \
                             ent_score[rel_match_obj_ids_flat]  # (num_queries,  1)

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

            pred_rel_triplet = torch.cat((rel_match_sub_ids2cat.long().to(device),
                                          rel_match_obj_ids2cat.long().to(device),
                                          pred_rel_labels.long().to(device),
                                          rel_trp_scores.to(device),
                                          pred_rel_pred_score.to(device),
                                          total_pred_idx.to(device)), 1)

            
            pred_rel_triplet = pred_rel_triplet.to("cpu")
            init_pred_rel_triplet = pred_rel_triplet.to("cpu")

            # first stage filtering
            # removed the self connection
            non_self_conn_idx = (rel_match_obj_ids_flat - rel_match_sub_ids_flat) != 0
            pred_rel_triplet = init_pred_rel_triplet[non_self_conn_idx]

            _, top_rel_idx = torch.sort(pred_rel_triplet[:, REL_TRP_SCR], descending=True)
            top_rel_idx = top_rel_idx[: self.cfg.MODEL.REL_DETR.NUM_MAX_REL_PRED]

            non_self_conn_idx = non_self_conn_idx[top_rel_idx]
            pred_rel_triplet = pred_rel_triplet[top_rel_idx]


            ent_label = ent_det['labels'].detach().cpu().numpy()
            ent_box = ent_det['boxes']

            self_iou = generalized_box_iou(box_cxcywh_to_xyxy(ent_box),
                                           box_cxcywh_to_xyxy(ent_box)).detach().cpu().numpy()

            sub_idx = pred_rel_triplet[:, SUB_IDX].long().detach().cpu().numpy()
            obj_idx = pred_rel_triplet[:, OBJ_IDX].long().detach().cpu().numpy()

            rel_pred_label = pred_rel_triplet[:, REL_LABEL].long().detach().cpu().numpy()
            rel_pred_score = pred_rel_triplet[:, REL_PRED_SCR].detach().cpu().numpy()
            
            
            def rel_prediction_nms(pred_idx_set:list, new_come_pred_idx):
                """

                Args:
                    pred_idx_set:
                    new_come_pred_idx:

                Returns:

                """
                occ_flag = False

                new_come_sub_idx = sub_idx[new_come_pred_idx]
                new_come_obj_idx = obj_idx[new_come_pred_idx]

                new_come_sub_label = ent_label[new_come_sub_idx]
                new_come_obj_label = ent_label[new_come_obj_idx]

                new_come_pred_label = rel_pred_label[new_come_pred_idx]
                new_come_pred_score = rel_pred_score[new_come_pred_idx]

                for idx, each_pred_idx in enumerate(pred_idx_set):
                    curr_sub_idx = sub_idx[each_pred_idx]
                    curr_obj_idx = obj_idx[each_pred_idx]

                    curr_sub_label = ent_label[curr_sub_idx]
                    curr_obj_label = ent_label[curr_obj_idx]

                    curr_pred_label = rel_pred_label[each_pred_idx]
                    curr_pred_score = rel_pred_score[each_pred_idx]

                    entities_match = curr_sub_idx == new_come_sub_idx and curr_obj_idx == new_come_obj_idx
                    if not entities_match:
                        sub_iou = self_iou[new_come_sub_idx, curr_sub_idx]
                        obj_iou = self_iou[new_come_obj_idx, curr_obj_idx]

                        entities_match = (
                                sub_iou > self.co_nms_thres and obj_iou > self.co_nms_thres
                                and curr_sub_label == new_come_sub_label
                                and curr_obj_label == new_come_obj_label
                        )

                    if (entities_match and curr_pred_label == new_come_pred_label):
                        occ_flag = True
                        if (new_come_pred_score > curr_pred_score):
                            pred_idx_set[idx] = new_come_pred_idx

                if not occ_flag:
                    pred_idx_set.append(new_come_pred_idx)

                return pred_idx_set


            non_max_suppressed_idx = []
            for trp_idx in range(len(pred_rel_triplet)):
                non_max_suppressed_idx = rel_prediction_nms(non_max_suppressed_idx, trp_idx)
            non_max_suppressed_idx = torch.Tensor(non_max_suppressed_idx).long().to(device)
            ################

            # top K selection
            tmp = torch.zeros((pred_rel_triplet.shape[0]), dtype=torch.bool).to(device)
            tmp[non_max_suppressed_idx] = True
            non_max_suppressed_idx = tmp  # generate the boolean mask for and operation
            pred_rel_triplet_selected = pred_rel_triplet[torch.logical_and(non_max_suppressed_idx,
                                                                            non_self_conn_idx)]

            # lower score thres
            # low_score_filter_idx = pred_rel_triplet_selected[:, REL_TRP_SCR] > 0.001
            # pred_rel_triplet_selected = pred_rel_triplet_selected[low_score_filter_idx]

            # top K selection
            _, top_rel_idx = torch.sort(pred_rel_triplet_selected[:, REL_TRP_SCR], descending=True)
            pred_rel_triplet_selected = pred_rel_triplet_selected[top_rel_idx[: max_proposal_pairs]]


            pred_dict = {
                'rel_trp': pred_rel_triplet[:, :3].long(),
                "rel_pred_label": pred_rel_triplet[:, REL_LABEL].long(),
                "rel_score": pred_rel_triplet[:, REL_PRED_SCR],
                "rel_trp_score": pred_rel_triplet[:, REL_TRP_SCR],
                "pred_prob_dist": pred_rel_probs[batch_ind][pred_rel_triplet[:, INIT_PROP_IDX].long()],
                "init_prop_indx": pred_rel_triplet[:, INIT_PROP_IDX].long(),
                "match_scr_sub": rel_match_sub_ids_flat[pred_rel_triplet[:, INIT_PROP_IDX].long()],
                "match_scr_obj": rel_match_obj_ids_flat[pred_rel_triplet[:, INIT_PROP_IDX].long()]
            }

            rel_proposals_predict.append(pred_dict)



        return rel_proposals_predict

