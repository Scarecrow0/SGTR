import copy
import sys

import torch
import torch.nn.functional as F
from torch import nn
from cvpods.configs.base_config import ConfigDict

from cvpods.layers import ShapeSpec, position_encoding_dict
from cvpods.modeling.backbone import Transformer
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.modeling.meta_arch.detr import SetCriterion, PostProcess, MLP
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_inference import (
    RelPostProcess,
    RelPostProcessSingleBranch,
)
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_losses import (
    RelSetCriterion,
    RelHungarianMatcher,
)
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from cvpods.modeling.roi_heads.relation_head.utils_motifs import to_onehot
from cvpods.structures import ImageList, Instances, Boxes
from cvpods.structures import boxes as box_ops
from cvpods.structures.relationship import Relationships
from cvpods.utils.dump.intermediate_dumper import create_save_root, store_data, is_empty

from .rel_detr import (
    EntitiesIndexingHead,
    EntitiesIndexingHeadRuleBased,
    EntitiesIndexingHeadPredAtt,
)
from .predicate_node_generator import PredicateNodeGenerator


sys.path.append("..")

from ..rel_detr_losses import AuxRelHungarianMatcher, AuxRelSetCriterion

__all__ = ["OneStageEncDecVRD"]


class OneStageEncDecVRD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        # Build Transformer
        self.ent_aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_queries = cfg.MODEL.DETR.NUM_QUERIES

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
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


        # Build Object Queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }

        losses = ["labels", "boxes", "cardinality"]

        # entities matcher and loss

        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DETR.COST_CLASS,
            cost_bbox=cfg.MODEL.DETR.COST_BBOX,
            cost_giou=cfg.MODEL.DETR.COST_GIOU,
        )

        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.MODEL.DETR.EOS_COEFF,
            losses=losses,
        )

        ###########################################
        #  relationship head Transformer

        self.use_gt_box = cfg.MODEL.REL_DETR.USE_GT_ENT_BOX
        if cfg.MODEL.LOAD_PROPOSALS or self.use_gt_box:
            encoder_out_shape = ShapeSpec(
                channels=hidden_dim, stride=backbone_out_shapes.stride
            )
            self.precompute_roi_init({'res5': encoder_out_shape})

        if self.use_gt_box:
            self.gt_aux_class = nn.Linear(hidden_dim, self.num_classes + 1)
            self.gt_aux_bbox = MLP(hidden_dim, hidden_dim, 4, 3)

        self.rel_aux_loss = not cfg.MODEL.REL_DETR.NO_AUX_LOSS
        self.num_rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_rel_queries = cfg.MODEL.REL_DETR.NUM_QUERIES

        if self.cfg.MODEL.REL_DETR.TRANSFORMER.NUM_DEC_LAYERS < 2:
            # no more than 1 layer, no need the aux loss
            self.rel_aux_loss = False

        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            self.obj_class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
            self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.sub_class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
            self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # post entities decoder

        self.entities_indexing_heads = None
        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_INDEXING:
            self.indexing_module_type = (
                cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.INDEXING_TYPE
            )

            self.entities_indexing_heads_rule = EntitiesIndexingHeadRuleBased(cfg)

            if self.indexing_module_type == "feat_att":
                self.entities_indexing_heads = nn.ModuleDict(
                    {
                        "sub": EntitiesIndexingHead(cfg),
                        "obj": EntitiesIndexingHead(cfg),
                    }
                )
            elif self.indexing_module_type in ["rule_base", 'rel_vec']:
                self.entities_indexing_heads = self.entities_indexing_heads_rule
            elif self.indexing_module_type == "pred_att":
                self.entities_indexing_heads = EntitiesIndexingHeadPredAtt(cfg)
            else:
                assert False

        #  rel decoder
        hidden_dim = cfg.MODEL.REL_DETR.TRANSFORMER.D_MODEL
        encoder_out_shape = ShapeSpec(
            channels=hidden_dim, stride=backbone_out_shapes.stride
        )

        self.relation_decoder = PredicateNodeGenerator(cfg, {'res5': encoder_out_shape})

        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            self.relation_decoder.sub_class_embed = self.sub_class_embed
            self.relation_decoder.sub_bbox_embed = self.sub_bbox_embed

            self.relation_decoder.obj_bbox_embed = self.obj_bbox_embed
            self.relation_decoder.obj_class_embed = self.obj_class_embed

        # input projection
        self.rel_input_proj = nn.Conv2d(
            backbone_out_shapes.channels, hidden_dim, kernel_size=1
        )

        # Build FFN
        self.rel_query_embed = nn.Embedding(self.num_rel_queries, hidden_dim)
        self.rel_query_pos_embed = nn.Embedding(self.num_rel_queries, hidden_dim)

        if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            self.rel_class_embed = nn.Linear(hidden_dim, self.num_rel_classes)
        else:
            self.rel_class_embed = nn.Linear(hidden_dim, self.num_rel_classes + 1)

        self.rel_vector_embed = nn.Linear(hidden_dim, 4)

        self.rel_position_embedding = position_encoding_dict[
            cfg.MODEL.REL_DETR.POSITION_EMBEDDING
        ](
            num_pos_feats=hidden_dim // 2,
            temperature=cfg.MODEL.REL_DETR.TEMPERATURE,
            normalize=True
            if cfg.MODEL.REL_DETR.POSITION_EMBEDDING == "sine"
            else False,
            scale=None,
        )

        # relationship vector branch, to produce the interaction direction
        # Build Object Queries
        ent_box_l1 = cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF
        ent_box_giou = cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF
        ent_labels_ce = cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_CLS_LOSS_COEFF
        ent_indexing = cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_INDEXING_LOSS_COEFF

        self.weight_dict.update(
            {
                "loss_rel_ce": cfg.MODEL.REL_DETR.CLASS_LOSS_COEFF,
                "loss_rel_vector": cfg.MODEL.REL_DETR.REL_VEC_LOSS_COEFF,
                "loss_aux_obj_entities_boxes": ent_box_l1,
                "loss_aux_sub_entities_boxes": ent_box_l1,
                "loss_aux_obj_entities_boxes_giou": ent_box_giou,
                "loss_aux_sub_entities_boxes_giou": ent_box_giou,
                "loss_aux_obj_entities_labels_ce": ent_labels_ce,
                "loss_aux_sub_entities_labels_ce": ent_labels_ce,
            }
        )

        # relationship matcher
        matcher = RelHungarianMatcher(
            cfg,
            cost_rel_class=cfg.MODEL.REL_DETR.COST_CLASS,
            cost_rel_vec=cfg.MODEL.REL_DETR.COST_REL_VEC,
            cost_class=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_ENT_CLS,
            cost_bbox=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_L1,
            cost_giou=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_GIOU,
            cost_indexing=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_INDEXING,
        )

        # relationship criterion,
        losses = set(copy.deepcopy(cfg.MODEL.REL_DETR.LOSSES))
        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            losses.add("rel_entities_aware")
            if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_INDEXING:
                losses.add("rel_entities_indexing")

        if (
                cfg.MODEL.REL_DETR.DYNAMIC_QUERY
                and cfg.MODEL.REL_DETR.DYNAMIC_QUERY_AUX_LOSS_WEIGHT is not None
        ):
            losses.add("rel_dynamic_query")

        self.disalign_train = False
        try:
            if cfg.MODEL.DISALIGN.ENABLED:
                self.disalign_train = True
        except AttributeError:
            pass

        if self.disalign_train:
            losses.add("disalign_loss")
            self.rel_class_embed_disalign_weight_cp = False
            self.rel_class_embed_disalign = nn.Linear(hidden_dim, self.num_rel_classes + 1)
        else:
            if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                losses.add("rel_labels_fl")
            else:
                losses.add("rel_labels")

        self.rel_criterion = RelSetCriterion(
            cfg,
            self.num_rel_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.MODEL.REL_DETR.EOS_COEFF,
            losses=list(losses),
        )

        if self.ent_aux_loss or self.rel_aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                self.aux_weight_dict.update(
                    {
                        k + f"/layer{i}": v * cfg.MODEL.REL_DETR.AUX_LOSS_WEIGHT
                        for k, v in self.weight_dict.items()
                    }
                )
            self.weight_dict.update(self.aux_weight_dict)

        if self.rel_aux_loss:
            losses = set(copy.deepcopy(cfg.MODEL.REL_DETR.LOSSES))

            if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
                losses.add("rel_entities_aware")


            if self.disalign_train:
                losses.add("disalign_loss")
            else:
                if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                    losses.add("rel_labels_fl")
                else:
                    losses.add("rel_labels")

            if not cfg.MODEL.REL_DETR.USE_SAME_MATCHER:
                matcher = AuxRelHungarianMatcher(
                    cfg,
                    cost_rel_class=cfg.MODEL.REL_DETR.COST_CLASS,
                    cost_rel_vec=cfg.MODEL.REL_DETR.COST_REL_VEC,
                    cost_class=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_ENT_CLS,
                    cost_bbox=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_L1,
                    cost_giou=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_GIOU,
                    cost_indexing=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_INDEXING,
                )

            aux_weight_dict = copy.deepcopy(self.weight_dict)

            for i in range(cfg.MODEL.REL_DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                for k, v in self.weight_dict.items():
                    aux_weight_dict[k] = v

            self.rel_criterion_aux = AuxRelSetCriterion(
                cfg,
                self.num_rel_classes,
                matcher=matcher,
                weight_dict=aux_weight_dict,
                eos_coef=cfg.MODEL.REL_DETR.EOS_COEFF,
                losses=list(losses),
            )

        self.post_processors = {
            "bbox": PostProcess(),
            "rel": RelPostProcess(cfg),
        }  # relationship PostProcess

        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_ENTITIES_PRED:
            self.post_processors['rel'] = RelPostProcessSingleBranch(cfg)

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

        # visualization
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self._reset_parameters()

        self.to(self.device)

    def precompute_roi_init(self, roi_input_shape):
        self.box_pooler = self._init_box_pooler(roi_input_shape)
        self.roi_fc = FastRCNNConvFCHead(
            self.cfg,
            ShapeSpec( channels=self.hidden_dim,height=7,width=7,),
            param_dicts=ConfigDict(dict(
                POOLER_RESOLUTION=7, POOLER_SAMPLING_RATIO=0,
                NUM_CONV=2, CONV_DIM=256, NUM_FC=1, FC_DIM=256, NORM="BN"
            ))
        )

    @staticmethod
    def _init_box_pooler(input_shape):

        pooler_resolution = 7
        pooler_scales = tuple(1.0 / input_shape[k].stride for k, v in input_shape.items())
        sampling_ratio = 0
        pooler_type = 'ROIAlignV2'
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [v.channels for k, v in input_shape.items()]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def _reset_parameters(self):

        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            num_layers = self.cfg.MODEL.REL_DETR.TRANSFORMER.NUM_DEC_LAYERS

            def initialize_ent_pred(class_embed, bbox_embed):
                class_embed = nn.ModuleList([class_embed for _ in range(num_layers)])
                bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_layers)])

                return class_embed, bbox_embed

            (self.obj_class_embed, self.obj_bbox_embed) = initialize_ent_pred(self.obj_class_embed, self.obj_bbox_embed)
            (self.sub_class_embed, self.sub_bbox_embed) = initialize_ent_pred(self.sub_class_embed, self.sub_bbox_embed)

    # input pre-process
    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]

            boxes_xyxy = bi["instances"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = box_ops.box_xyxy_to_cxcywh(boxes_xyxy).to(self.device)

            # cxcywh 0-1, w-h
            target["boxes"] = boxes.to(self.device)
            target["boxes_init"] = box_ops.box_xyxy_to_cxcywh(bi["instances"].gt_boxes.tensor).to(
                self.device)  # cxcy 0-1

            # xyxy 0-1, w-h
            target["boxes_xyxy_init"] = bi["instances"].gt_boxes.tensor.to(self.device)
            target["boxes_xyxy"] = boxes_xyxy.to(self.device)

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

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    # model forward
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :mth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.a
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        # GT matching during the training for
        # try to obtain the GT
        targets = self.convert_anno_format(batched_inputs)

        device = images.tensor.device

        image_sizes = torch.stack(
            [
                torch.tensor(
                    [bi.get("height", img_size[0]), bi.get("width", img_size[1]), ],
                    device=self.device,
                )
                for bi, img_size in zip(batched_inputs, images.image_sizes)
            ]
        )


        (
            mask, src, ent_hs, mem_enc, 
            lyrs_outputs_class, lyrs_outputs_coord, 
            ent_box_pred, ent_cls_pres, 
        ) = self.entities_detr(batched_inputs, targets, images)

        ## relation heads
        rel_src_pos_embed = self.rel_position_embedding(src, mask)

        if hasattr(self.relation_decoder, 'set_box_scale_factor'):
            img_h, img_w = image_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            self.relation_decoder.set_box_scale_factor(scale_fct[:, None, :])
        
        ent_node_box = ent_box_pred
        ent_node_cls = ent_cls_pres

        rel_hs, ext_inter_feats, rel_decoder_out_res = self.relation_decoder(
            self.rel_input_proj(src),
            mask,
            self.rel_query_embed.weight,
            rel_src_pos_embed,
            self.rel_query_pos_embed.weight,
            shared_encoder_memory=mem_enc,
            ent_hs=ent_hs,
            ent_coords=ent_node_box,
            ent_cls=ent_node_cls,
        )

        pred_rel_logits = self.rel_class_embed(rel_hs.feature)  # n_lyr, batch_size, num_queries, N

        pred_rel_vec = self.rel_vector_embed(rel_hs.feature)  # batch_size, num_queries, 4

        pred_rel_vec = pred_rel_vec.sigmoid()

        #  pack prediction results
        semantic_predictions = {
            "pred_logits": ent_cls_pres,
            "pred_boxes": ent_box_pred,
            "pred_rel_logits": pred_rel_logits[-1],
            # layer, batch_size, num_queries, 4 => batch_size, num_queries, 4
            "pred_rel_vec": pred_rel_vec[-1]
            # take the output from the last layer
        }

        rel_hs_ent_aware_sub = None
        rel_hs_ent_aware_obj = None

        pred_rel_sub_box = None
        pred_rel_obj_box = None
        pred_rel_obj_logits = None
        pred_rel_sub_logits = None

        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            (ent_hs,
             pred_rel_obj_box, pred_rel_obj_logits,
             pred_rel_sub_box, pred_rel_sub_logits,
             rel_hs_ent_aware_obj, rel_hs_ent_aware_sub, 
             pred_ent_rel_vec) = self.predict_rel_ent_semantics(
                ent_hs, ext_inter_feats, image_sizes, rel_hs, semantic_predictions, rel_decoder_out_res
            )
            # all possible prediction
            # ['pred_logits', 'pred_boxes', 'pred_rel_logits', 'pred_rel_vec', 'pred_rel_obj_logits',
            #  'pred_rel_obj_box', 'pred_rel_sub_logits', 'pred_rel_sub_box', 'dyna_anchor_obj_box',
            #  'dyna_anchor_sub_box', 'uni_ass_sub_boxs', 'uni_ass_sub_logits', 'uni_ass_sub_index',
            #  'uni_ass_obj_boxs', 'uni_ass_obj_logits', 'uni_ass_obj_index']


        pred_rel_confidence = None

        rel_aux_out = self.generate_aux_out(image_sizes, ent_hs, ent_cls_pres, ent_box_pred,
                                            pred_rel_logits, pred_rel_vec,
                                            rel_hs_ent_aware_sub, rel_hs_ent_aware_obj,
                                            pred_rel_sub_box, pred_rel_obj_box,
                                            pred_rel_obj_logits, pred_rel_sub_logits, rel_decoder_out_res)


        use_pre_comp_box = batched_inputs[0]["relationships"].has_meta_info("precompute_prop")

        if use_pre_comp_box:
            semantic_predictions["precompute_prop"] = [
                each["relationships"].get_meta_info("precompute_prop")
                for each in batched_inputs
            ]

        if self.training:
            if self.use_gt_box or use_pre_comp_box:
                # apply supervision for feature extraction heads
                semantic_predictions['pred_logits'] = lyrs_outputs_class[-1] 
                semantic_predictions['pred_boxes'] = lyrs_outputs_coord[-1] 

            loss_dict = self.loss_eval(batched_inputs, image_sizes, images, lyrs_outputs_class, lyrs_outputs_coord,
                                       pred_rel_confidence, rel_aux_out, semantic_predictions)
            return loss_dict

        else:
            rel_decoder_out_res['mask'] = mask
            if self.rel_aux_loss:
                semantic_predictions['aux_outputs'] = rel_aux_out
            pred_res = self.inference(images, batched_inputs, targets, semantic_predictions,
                                      rel_decoder_out_res)
            return pred_res

    def entities_detr(self, batched_inputs, targets, images):
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


        max_size = self.num_queries

        def padding_tensor(tensor):
            while len(tensor) < max_size:
                padd = torch.zeros((max_size - len(tensor), tensor.shape[-1]), device=tensor.device)
                tensor = torch.cat((tensor, padd), dim=0)
                # print(len(tensor))
            return tensor


        ent_hs, mem_enc = self.transformer(
            self.input_proj(src),
            mask,
            self.query_embed.weight,
            pos,
            enc_return_lvl=self.cfg.MODEL.REL_DETR.TRANSFORMER.SHARE_ENC_FEAT_LAYERS,
            # return  memory for selected encoder layer
        )

        outputs_class = self.class_embed(ent_hs)  # B, N, Dim
        outputs_coord = self.bbox_embed(ent_hs).sigmoid()
        ent_box_pred = outputs_coord[-1]
        ent_cls_pres = outputs_class[-1]


        return (
            mask, src,
            ent_hs, mem_enc,
            outputs_class, outputs_coord,
            ent_box_pred, ent_cls_pres, 
        )

    # from raw prediction to semantic results
    def predict_rel_ent_semantics(self, ent_hs, ext_inter_feats, image_sizes, rel_hs,
                                  semantic_predictions, rel_decoder_out_res):
        # default entities representation is overide by the unify relationship representation
        rel_hs_ent_aware_sub = rel_hs.feature
        rel_hs_ent_aware_obj = rel_hs.feature
        if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.CROSS_DECODER:
            rel_hs_ent_aware_sub = ext_inter_feats[0].feature
            rel_hs_ent_aware_obj = ext_inter_feats[1].feature

            pred_rel_sub_box = []
            pred_rel_obj_box = []
            pred_rel_obj_logits = []
            pred_rel_sub_logits = []

            for lid in range(len(rel_hs_ent_aware_sub)):
                pred_rel_sub_logits.append(self.sub_class_embed[lid](rel_hs_ent_aware_sub[lid]))
                pred_rel_sub_box.append(self.sub_bbox_embed[lid](rel_hs_ent_aware_sub[lid]).sigmoid())
                pred_rel_obj_logits.append(self.obj_class_embed[lid](rel_hs_ent_aware_obj[lid]))
                pred_rel_obj_box.append(self.obj_bbox_embed[lid](rel_hs_ent_aware_obj[lid]).sigmoid())

            pred_rel_sub_logits = torch.stack(pred_rel_sub_logits)
            pred_rel_sub_box = torch.stack(pred_rel_sub_box)
            pred_rel_obj_logits = torch.stack(pred_rel_obj_logits)
            pred_rel_obj_box = torch.stack(pred_rel_obj_box)  # layer, bz, num_q, dim
        
        pred_ent_rel_vec = torch.cat((pred_rel_sub_box[..., :2], pred_rel_obj_box[..., :2]), dim=-1)

        semantic_predictions.update(
            {
                "pred_rel_obj_logits": pred_rel_obj_logits[-1],
                "pred_rel_obj_box": pred_rel_obj_box[-1],
                "pred_rel_sub_logits": pred_rel_sub_logits[-1],
                "pred_rel_sub_box": pred_rel_sub_box[-1],
                "pred_ent_rel_vec": pred_ent_rel_vec[-1]
            }
        )


        if self.entities_indexing_heads is not None:
            ent_hs = ent_hs[-1]

            (
                sub_idxing, obj_idxing, sub_idxing_rule, obj_idxing_rule,
            ) = self.graph_assembling(
                semantic_predictions, image_sizes, ent_hs,
                rel_hs_ent_aware_sub[-1], rel_hs_ent_aware_obj[-1],
            )
            semantic_predictions.update({
                "sub_entities_indexing": sub_idxing,
                "obj_entities_indexing": obj_idxing,
                "sub_ent_indexing_rule": sub_idxing_rule,
                "obj_ent_indexing_rule": obj_idxing_rule,
            })

        return ent_hs, pred_rel_obj_box, pred_rel_obj_logits, pred_rel_sub_box, pred_rel_sub_logits, rel_hs_ent_aware_obj, rel_hs_ent_aware_sub, pred_ent_rel_vec

    def generate_aux_out(self, image_sizes, ent_hs, outputs_class, outputs_coord, pred_rel_logits, pred_rel_vec,
                         rel_hs_ent_aware_sub, rel_hs_ent_aware_obj, pred_rel_sub_box, pred_rel_obj_box,
                         pred_rel_obj_logits, pred_rel_sub_logits, rel_decoder_out_res):
        aux_out = []
        for ir in range(len(pred_rel_logits) - 1):
            tmp_out = {
                # take the output from the last layer
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                # layer, batch_size, num_queries, 4
                #     => batch_size, num_queries, 4
                "pred_rel_logits": pred_rel_logits[ir],
                "pred_rel_vec": pred_rel_vec[ir],
            }

            if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
                tmp_out.update({
                    "pred_rel_obj_logits": pred_rel_obj_logits[ir],
                    "pred_rel_obj_box": pred_rel_obj_box[ir],
                    "pred_rel_sub_logits": pred_rel_sub_logits[ir],
                    "pred_rel_sub_box": pred_rel_sub_box[ir],
                })

                if rel_decoder_out_res.get('unify_assembling') is not None:
                    uni_ass_res = rel_decoder_out_res['unify_assembling'][-1]
                    for idx, role in enumerate(['sub', 'obj']):
                        for att_name in ['boxs', 'logits', "index"]:
                            tmp_out.update({
                                f'uni_ass_{role}_{att_name}': uni_ass_res[f'agg_{att_name}'][idx]
                            })

            if self.entities_indexing_heads is not None:
                (sub_idxing, obj_idxing, sub_idxing_rule, obj_idxing_rule) = self.graph_assembling(
                    tmp_out, image_sizes, ent_hs,
                    rel_hs_ent_aware_sub[ir], rel_hs_ent_aware_obj[ir],
                )

                tmp_out.update({
                    "sub_entities_indexing": sub_idxing,
                    "obj_entities_indexing": obj_idxing,
                    "sub_ent_indexing_rule": sub_idxing_rule,
                    "obj_ent_indexing_rule": obj_idxing_rule,
                })

                # if rel_decoder_out_res.get('unify_assembling') is not None:
                #     uni_ass_res = rel_decoder_out_res['unify_assembling'][-1]
                #     for idx, role in enumerate(['sub', 'obj']):
                #         tmp_out.update({f'{role}_entities_indexing': uni_ass_res[f'agg_index'][idx]})

            aux_out.append(tmp_out)

        return aux_out

    def graph_assembling(
            self, out, image_sizes, ent_hs, rel_hs_ent_aware_sub, rel_hs_ent_aware_obj
    ):
        sub_idxing_rule, obj_idxing_rule = self.entities_indexing_heads_rule(
            out, image_sizes
        )

        if self.indexing_module_type in ["rule_base", "pred_att", 'rel_vec']:

            if self.indexing_module_type in ["rule_base", 'rel_vec']:
                sub_idxing, obj_idxing = sub_idxing_rule, obj_idxing_rule
            elif self.indexing_module_type == "pred_att":
                sub_idxing, obj_idxing = self.entities_indexing_heads(out)

        elif self.indexing_module_type == "feat_att":
            sub_idxing = self.entities_indexing_heads["sub"](
                ent_hs, rel_hs_ent_aware_sub
            )
            obj_idxing = self.entities_indexing_heads["obj"](
                ent_hs, rel_hs_ent_aware_obj
            )

        return sub_idxing, obj_idxing, sub_idxing_rule, obj_idxing_rule

    # loss eval
    def loss_eval(self, batched_inputs, image_sizes, images, outputs_class, outputs_coord, pred_rel_confidence,
                  rel_aux_out, semantic_predictions):
        targets = self.convert_anno_format(batched_inputs)
        if self.ent_aux_loss:
            semantic_predictions["aux_outputs"] = [
                {"pred_logits": ent_logits, "pred_boxes": ent_box, }
                for ent_logits, ent_box in zip(
                    outputs_class[:-1], outputs_coord[:-1],
                )
            ]
        loss_dict, ent_match_idx = self.criterion(semantic_predictions, targets, with_match_idx=True)
        ent_det_res = self.post_processors["bbox"](semantic_predictions, image_sizes)

        if semantic_predictions.get("aux_outputs") is not None:
            # clear the entities aux_loss
            semantic_predictions.pop("aux_outputs")
        aux_rel_loss_dict = {}
        if not self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.REUSE_ENT_MATCH:
            ent_match_idx = None
        rel_loss_dict, match_res = self.rel_criterion(
            semantic_predictions,
            targets,
            # re-use the entities matching results
            det_match_res=ent_match_idx,
        )
        # generate the entities regrouping predicitons of each layer prediction
        # for auxiliary loss
        aux_rel_match = None
        if self.rel_aux_loss:
            semantic_predictions['aux_outputs'] = rel_aux_out
            aux_rel_loss_dict, aux_rel_match = self.aux_rel_loss_cal(
                semantic_predictions,  # last layer pred
                pred_rel_confidence,
                targets, ent_match_idx, match_res,
            )

        loss_dict.update(rel_loss_dict)
        loss_dict.update(aux_rel_loss_dict)
        for k, v in loss_dict.items():
            loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
        return loss_dict

    def aux_rel_loss_cal(self, out, pred_rel_confidence, targets, ent_match_idx, match_res, ):

        aux_rel_loss_dict, aux_rel_match = self.rel_criterion_aux(
            out,
            targets,
            match_res=match_res,
            aux_only=True,
            ent_match_idx=ent_match_idx,
            # re-use the foreground entities matching results
            entities_match_cache=self.rel_criterion.entities_match_cache,
        )

        return aux_rel_loss_dict, aux_rel_match

    # inference time
    def inference(self, images, batched_inputs, targets, semantic_out, out_res=None):
        """
        Run inference on the given inputs.

        Args:
            images:
            batched_inputs (list[dict]): same as in :meth:`forward`
            semantic_out: the dict of models prediction, concatenated batch together

        Returns:

        """
        assert not self.training

        target_sizes = torch.stack(
            [
                torch.tensor(
                    [bi.get("height", img_size[0]), bi.get("width", img_size[1])],
                    device=self.device,
                )
                for bi, img_size in zip(batched_inputs, images.image_sizes)
            ]
        )
        ent_det_res = self.post_processors["bbox"](semantic_out, target_sizes)

        # list[{"scores": s, "labels": l, "boxes": b}]


        (rel_det_res, init_rel_det_res) = self.post_processors["rel"](
            semantic_out, ent_det_res, target_sizes
        )

        pred_res = OneStageEncDecVRD._postprocess(
            ent_det_res, rel_det_res, batched_inputs, images.image_sizes
        )



        outputs_without_aux = {
            k: v for k, v in semantic_out.items() if k != "aux_outputs" and k != "ref_pts_pred"
        }
        (indices, match_cost, detailed_cost_dict) = self.rel_criterion.matcher(
            outputs_without_aux, targets, return_init_idx=True
        )

        # save the matching for evaluation in test/validation time
        if "sub_entities_indexing" in outputs_without_aux:
            # add the indexing module performance in test time
            num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING
            indices_reduce = []
            for p, g in indices:
                indices_reduce.append((torch.div(p, num_ent_pairs, rounding_mode='trunc'), g))

            loss_ent_idx = self.rel_criterion.loss_aux_entities_indexing(
                outputs_without_aux, targets, indices_reduce, None
            )
            for each in pred_res:
                for k, v in loss_ent_idx.items():
                    if "acc" in k:
                        each["relationships"].add_meta_info(k, v)

        return pred_res

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

            if rel_res_per_img.get("rel_branch_box") is not None:
                rel_inst = Instances(image_size)
                rel_inst.pred_boxes = Boxes(rel_res_per_img["rel_branch_box"].float())
                rel_inst.scores = rel_res_per_img["rel_branch_score"].float()
                rel_inst.pred_classes = rel_res_per_img["rel_branch_label"]
                rel_inst.pred_score_dist = rel_res_per_img["rel_branch_dist"]
                det_result = rel_inst
            else:
                det_result = Instances(image_size)
                det_result.pred_boxes = Boxes(det_res_per_image["boxes"].float())
                det_result.scores = det_res_per_image["scores"].float()
                det_result.pred_classes = det_res_per_image["labels"]
                det_result.pred_score_dist = det_res_per_image["prob"]

            pred_rel = Relationships(
                instances=det_result,
                rel_pair_tensor=rel_res_per_img["rel_trp"][:, :2],
                pred_rel_classs=rel_res_per_img["rel_pred_label"],  # start from 1
                pred_rel_scores=rel_res_per_img["rel_score"],
                pred_rel_trp_scores=rel_res_per_img["rel_trp_score"],
                pred_rel_dist=rel_res_per_img["pred_prob_dist"],
                pred_init_prop_idx=rel_res_per_img["init_prop_indx"],
            )

            if rel_res_per_img.get("rel_vec") is not None:
                pred_rel.rel_vec = rel_res_per_img.get("rel_vec")

            if rel_res_per_img.get("pred_rel_confidence") is not None:
                pred_rel.pred_rel_confidence = rel_res_per_img[
                    "pred_rel_confidence"
                ].unsqueeze(-1)

            if rel_res_per_img.get("selected_mask") is not None:
                for k, v in rel_res_per_img.get("selected_mask").items():
                    pred_rel.__setattr__(k, v)

            if rel_res_per_img.get("pred_rel_ent_obj_box") is not None:
                for role in ['sub', 'obj']:
                    for k_name in [f'pred_rel_ent_{role}_box',
                                   f'pred_rel_ent_{role}_label',
                                   f'pred_rel_ent_{role}_score']:
                        pred_rel.__setattr__(k_name, rel_res_per_img.get(k_name))

                    if rel_res_per_img.get(f"dyna_anchor_{role}_box") is not None:
                        pred_rel.__setattr__(f"dyna_anchor_{role}_box",
                                             rel_res_per_img[f"dyna_anchor_{role}_box"])


            processed_results.append({"instances": det_result, "relationships": pred_rel})

        return processed_results


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
