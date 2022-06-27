import logging
from typing import List

import torch
from torch import nn
import torch.distributed as dist
from torchvision.ops.boxes import box_area
import numpy as np

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.modeling.basenet import basenet
from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.losses import sigmoid_focal_loss_jit
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances
from cvpods.utils import log_first_n


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@basenet
class YOLOF(nn.Module):
    """
    Implementation of YOLOF.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES
        self.in_features = cfg.MODEL.YOLOF.ENCODER.IN_FEATURES
        self.pos_ignore_thresh = cfg.MODEL.YOLOF.POS_IGNORE_THRESHOLD
        self.neg_ignore_thresh = cfg.MODEL.YOLOF.NEG_IGNORE_THRESHOLD
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.YOLOF.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.YOLOF.FOCAL_LOSS_GAMMA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.YOLOF.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.YOLOF.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.YOLOF.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.encoder = cfg.build_encoder(cfg, backbone_shape)
        self.decoder = cfg.build_decoder(cfg)
        self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.YOLOF.BBOX_REG_WEIGHTS,
            add_ctr_clamp=cfg.MODEL.YOLOF.ADD_CTR_CLAMP,
            ctr_clamp=cfg.MODEL.YOLOF.CTR_CLAMP,
        )
        self.matcher = UniformMatcher(cfg.MODEL.YOLOF.MATCHER_TOPK)

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1),
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        )
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.decoder(self.encoder(features[0]))
        anchors = self.anchor_generator(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(box_cls, self.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(box_delta, 4)]

        if self.training:
            indices = self.get_ground_truth(anchors, pred_anchor_deltas, gt_instances)
            losses = self.losses(
                indices, gt_instances, anchors, pred_logits, pred_anchor_deltas
            )
            return losses
        else:
            results = self.inference(
                [box_cls], [box_delta], anchors, images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
        self, indices, gt_instances, anchors, pred_class_logits, pred_anchor_deltas
    ):
        pred_class_logits = cat(pred_class_logits, dim=1).view(-1, self.num_classes)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1).view(-1, 4)

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor
        # Boxes(Tensor(N*R, 4))
        predicted_boxes = self.box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors
        )
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)

        ious = []
        pos_ious = []
        for i in range(N):
            src_idx, tgt_idx = indices[i]
            iou, _ = box_iou(predicted_boxes[i, ...], gt_instances[i].gt_boxes.tensor)
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            a_iou, _ = box_iou(anchors[i].tensor, gt_instances[i].gt_boxes.tensor)
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        ious = torch.cat(ious)
        ignore_idx = ious > self.neg_ignore_thresh
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh

        src_idx = torch.cat(
            [
                src + idx * anchors[0].tensor.shape[0]
                for idx, (src, _) in enumerate(indices)
            ]
        )
        gt_classes = torch.full(
            pred_class_logits.shape[:1],
            self.num_classes,
            dtype=torch.int64,
            device=pred_class_logits.device,
        )
        gt_classes[ignore_idx] = -1
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(gt_instances, indices)]
        )
        target_classes_o[pos_ignore_idx] = -1
        gt_classes[src_idx] = target_classes_o

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        
        if dist.is_initialized():
            dist.all_reduce(num_foreground)
            num_foreground = num_foreground * 1.0 / dist.get_world_size()

        # cls loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        # reg loss
        target_boxes = torch.cat(
            [t.gt_boxes.tensor[i] for t, (_, i) in zip(gt_instances, indices)], dim=0
        )
        target_boxes = target_boxes[~pos_ignore_idx]
        matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
            src_idx[~pos_ignore_idx]
        ]
        loss_box_reg = (
            1 - torch.diag(generalized_box_iou(matched_predicted_boxes, target_boxes))
        ).sum()

        return {
            "loss_cls": loss_cls / max(1, num_foreground),
            "loss_box_reg": loss_box_reg / max(1, num_foreground),
        }

    @torch.no_grad()
    def get_ground_truth(self, anchors, bbox_preds, targets):
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor.reshape(N, -1, 4)
        # Boxes(Tensor(N*R, 4))
        box_delta = cat(bbox_preds, dim=1)
        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.matcher(box_pred, all_anchors, targets)
        return indices

    def inference(self, box_cls, box_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`YOLOFHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image,
                box_reg_per_image,
                anchors_per_image,
                tuple(image_size),
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor
            )

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(
            boxes_all,
            scores_all,
            class_idxs_all,
            self.nms_threshold,
            nms_type=self.nms_type,
        )
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class UniformMatcher(nn.Module):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.
    """

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad()
    def forward(self, pred_boxes, anchors, targets):
        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        # Also concat the target boxes
        tgt_bbox = torch.cat([v.gt_boxes.tensor for v in targets])

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox), p=1
        )
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(tgt_bbox), p=1
        )

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v.gt_boxes.tensor) for v in targets]
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        indices = [
            tuple(
                torch.topk(c[i], k=self.match_times, dim=0, largest=False)[1]
                .numpy()
                .tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(c[i], k=self.match_times, dim=0, largest=False)[1]
                .numpy()
                .tolist()
            )
            for i, c in enumerate(C1.split(sizes, -1))
        ]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [np.array(idx_ + idx1_) for (idx_, idx1_) in zip(idx, idx1)]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in all_indices
        ]


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
