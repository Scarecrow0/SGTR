import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, position_encoding_dict
from cvpods.modeling import FPN
from cvpods.modeling.backbone import Transformer
from cvpods.modeling.backbone.fpn import LastLevelMaxPool
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.modeling.meta_arch.detr import PostProcess, SetCriterion, MLP
from cvpods.structures import ImageList, Instances, Boxes
from cvpods.structures import boxes as box_ops
from cvpods.utils import get_event_storage

__all__ = ["TwoStageDETRVRD"]


class TwoStageDETRVRD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        # DETR for entities
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Build Backbone
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        # Build Transformer
        self.transformer = Transformer(cfg)

        self.aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_queries = cfg.MODEL.DETR.NUM_QUERIES
        hidden_dim = self.transformer.d_model

        # Build FFN
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # Build Object Queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        backbone_out_shapes = self.backbone.output_shape()["res5"]
        self.input_proj = nn.Conv2d(backbone_out_shapes.channels, hidden_dim, kernel_size=1)

        self.position_embedding = position_encoding_dict[cfg.MODEL.DETR.POSITION_EMBEDDING](
            num_pos_feats=hidden_dim // 2,
            temperature=cfg.MODEL.DETR.TEMPERATURE,
            normalize=True if cfg.MODEL.DETR.POSITION_EMBEDDING == "sine" else False,
            scale=None,
        )

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }

        if self.aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                self.aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(self.aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]

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

        self.post_processors = {"bbox": PostProcess()}

        # 2 stage relationship
        # print(self.backbone.output_shape())
        # self.rel_FPN = FPN(self.backbone,
        #                    in_features=cfg.MODEL.RESNETS.OUT_FEATURES,
        #                    out_channels=self.rel_input_dim,
        #                    norm="BN",
        #                    top_block=LastLevelMaxPool(),
        #                    fuse_type="sum",)

        self.rel_input_dim = cfg.MODEL.REL_HEAD_IN_FEAT_DIM
        self.rel_input_proj = nn.Conv2d(backbone_out_shapes.channels, self.rel_input_dim, kernel_size=1)
        rel_in_chnl = ShapeSpec(channels=self.rel_input_dim, stride=32)
        self.relation_head = cfg.build_relation_head(cfg, {"res5": rel_in_chnl})

        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        ############
        ## image pixel normalization
        # use detr default
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        if not cfg.MODEL.RESNETS.STRIDE_IN_1X1:
            # Custom or torch pretrain weights
            self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        else:
            # MSRA pretrain weights
            self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
        """
        from cvpods.utils import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def convert_anno_format(self, batched_inputs):
        if "instances" not in batched_inputs[0]:
            return None

        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]
            boxes = box_ops.box_xyxy_to_cxcywh(
                bi["instances"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32)
            )
            target["boxes"] = boxes.to(self.device)
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor([bi["height"], bi["width"]], device=self.device)
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)
            targets.append(target)

        return targets

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

        # entities detection part are always fixed
        with torch.no_grad():
            self.backbone = self.backbone.eval()
            self.transformer = self.transformer.eval()
            self.class_embed = self.class_embed.eval()
            self.bbox_embed = self.bbox_embed.eval()

            B, C, H, W = images.tensor.shape
            device = images.tensor.device

            mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
            for img_shape, m in zip(images.image_sizes, mask):
                m[: img_shape[0], : img_shape[1]] = False

            features = self.backbone(images.tensor)

            src = features["res5"]
            mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).bool()[0]
            pos = self.position_embedding(src, mask)

        ent_hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(ent_hs)
        outputs_coord = self.bbox_embed(ent_hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        losses = {}
        targets = None
        if self.training:
            #########################
            # loss calculation for entities detr
            targets = self.convert_anno_format(batched_inputs)
            assert targets is not None
            if self.aux_loss:
                out["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            detector_losses, match_indx = self.criterion(out, targets, with_match_idx=True)

            for k, v in detector_losses.items():
                detector_losses[k] = v * self.weight_dict[k] if k in self.weight_dict else v

            losses.update(detector_losses)

        #########################
        # post process for results of entities detr
        target_sizes = torch.stack(
            [
                torch.tensor([
                    bi.get("height", img_size[0]),
                    bi.get("width", img_size[1])],
                    device=self.device)
                for bi, img_size in zip(batched_inputs, images.image_sizes)
            ]
        )
        res = self.post_processors["bbox"](out, target_sizes)

        entity_det_instances = []
        for idx, (results_per_image, _, image_size) in enumerate(zip(res, batched_inputs,
                                                                     images.image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(results_per_image["boxes"].float())
            result.scores = results_per_image["scores"].float()
            result.pred_classes = results_per_image["labels"]
            result.pred_score_dist = F.softmax(out["pred_logits"][idx], dim=-1)[:, :-1]

            if targets is not None:
                mat_idx = match_indx[idx]
                target_classes = torch.full(
                    (len(result),), self.num_classes, dtype=torch.int64, device=self.device
                )
                target_classes[mat_idx[0]] = targets[idx]["labels"][mat_idx[1]]
                result.gt_classes = target_classes

            entity_det_instances.append(result)

        ################
        ## two stage relationship head

        if "relationships" in batched_inputs[0]:
            gt_relationships = [
                x["relationships"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_relationships = None

        features["res5"] = self.rel_input_proj(features["res5"])
        # fpn_feat = self.rel_FPN(None, features)

        ent_features = [each for each in ent_hs[-1]]
        rel_detected_res, relation_loss = self.relation_head(
            features, entity_det_instances, gt_relationships, entities_features=ent_features
        )

        if self.training:
            losses.update(relation_loss)
            return losses
        else:

            rel_det_res = TwoStageDETRVRD._postprocess(
                rel_detected_res, batched_inputs, images.image_sizes
            )
            return rel_det_res

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        size_divisibility = self.backbone.size_divisibility
        images = ImageList.from_tensors(images, size_divisibility)
        return images

    @staticmethod
    def _postprocess(rel_detected_res, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                rel_detected_res, batched_inputs, image_sizes
        ):
            inst_res = results_per_image.instances
            processed_results.append(
                {
                    "instances": inst_res,  # Instance object
                    "relationships": results_per_image,
                }  # Relationships object
            )

        return processed_results
