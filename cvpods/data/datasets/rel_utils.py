
from cvpods.structures import (
    Boxes,
    BoxMode,
    Instances,
)

from cvpods.structures.relationship import Relationships

import torch


def annotations_to_relationship(annos, image_size, with_relations):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (dict): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = BoxMode.convert(annos["entities_bbox"], annos["bbox_mode"], BoxMode.XYXY_ABS)

    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
    boxes.clip(image_size)

    classes = annos["entities_labels"]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    target.gt_classes_non_masked = torch.from_numpy(annos["entities_labels_non_masked"])
    
    relationships = None

    if with_relations:
        relationships = Relationships(target, annos['relation_tuple'][:, :2],
                                      rel_label=annos['relation_tuple'][:, -1],
                                      rel_label_no_mask=annos['relation_label_non_masked'],
                                      relation_tuple=annos['relation_tuple'])


    if annos.get("precompute_proposals") is not None:
        '''
                "entities_bbox": box_array, 
                "entities_labels": bbox.get_field('pred_labels').numpy(), 
                "entities_scores": bbox.get_field('pred_scores').numpy(), 
                "entities_scores_dist": bbox.get_field('pred_score_dist').numpy(), 
        '''
        precomp_prop = Instances(image_size)
        precomp_prop.box = Boxes(torch.tensor(annos['precompute_proposals']['entities_bbox'], dtype=torch.float32))
        precomp_prop.box.clip(image_size)

        precomp_prop.pred_labels = torch.tensor(annos['precompute_proposals']['entities_labels'])
        precomp_prop.entities_scores = torch.tensor(annos['precompute_proposals']['entities_scores'])
        precomp_prop.entities_scores_dist = torch.tensor(annos['precompute_proposals']['entities_scores_dist'])
        relationships.add_meta_info("precompute_prop", precomp_prop)

    return target, relationships
