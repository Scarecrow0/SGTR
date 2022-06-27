import logging

import numpy as np
import torch
from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import ImageList
from cvpods.utils import get_event_storage, log_first_n

__all__ = ["TwoStageVRD"]


class TwoStageVRD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.proposal_generator = cfg.build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = cfg.build_roi_heads(cfg, self.backbone.output_shape())
        self.relation_head = cfg.build_relation_head(cfg, self.backbone.output_shape())

        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        )
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
        if not self.training:
            with torch.no_grad():
                return self.inference(batched_inputs)

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

        if "relationships" in batched_inputs[0]:
            gt_relationships = [
                x["relationships"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_relationships = None

        # entities detection part are always fixed
        with torch.no_grad():
            features = self.backbone(images.tensor)

            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances
                )
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

        entity_det_instances, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        _, relation_loss = self.relation_head(
            features, entity_det_instances, gt_relationships
        )

        if self.vis_period:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(relation_loss)

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)

        if "relationships" in batched_inputs[0] and batched_inputs[0].get("relationships") is not None:
            gt_relationships = [
                x["relationships"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_relationships = None

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            ent_det_results, _ = self.roi_heads(images, features, proposals, None)
        else:
            # the entities locations has been given
            detected_instances = [x.to(self.device) for x in detected_instances]
            ent_det_results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        # relationship heads

        rel_detected_res, _ = self.relation_head(
            features, ent_det_results, gt_relationships
        )

        if do_postprocess:
            rel_detected_res = TwoStageVRD._postprocess(
                rel_detected_res, batched_inputs, images.image_sizes
            )

        return rel_detected_res

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
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
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            inst_res = detector_postprocess(results_per_image.instances, height, width)
            results_per_image.update_instance(inst_res)
            processed_results.append(
                {
                    "instances": inst_res,  # Instance object
                    "relationships": results_per_image,
                }  # Relationships object
            )

        return processed_results
