"""
Data loading functions for the GQA dataset: https://cs.stanford.edu/people/dorarad/gqa/about.html
"""
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict, Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from cvpods.data import DATASETS
from cvpods.data.datasets.open_image import OpenImageDataset
from cvpods.data.detection_utils import read_image
from cvpods.structures import BoxMode
from .bi_lvl_rsmp import apply_resampling, apply_resampling_ent, resampling_dict_generation_ent
from .paths_route import _PREDEFINED_SPLITS_OpenImage_SGDET
from .rel_utils import annotations_to_relationship
from .vg import get_VG_statistics
from ...utils.dump.intermediate_dumper import add_dataset_metadata

logger = logging.getLogger("cvpods." + __name__)


@DATASETS.register()
class GQADataset(OpenImageDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(GQADataset, self).__init__(cfg, dataset_name, transforms, is_train)

    def __getitem__(self, index):
        # read images
        if self.repeat_dict is not None or self.ent_repeat_dict is not None:
            index = self.idx_list[index]

        img = read_image(self.filenames[index], format=self.data_format)

        # obtain the gt as the dict form
        annotations = self.get_groundtruth(index)

        # apply the transform / augmentation
        image, annotations = self._apply_transforms(img, annotations)

        # dump as the instance objects
        image_shape = image.shape[:2]
        target, relationships = annotations_to_relationship(annotations, image_shape, self.relation_on)

        # wrap back to the dict form with needed fields.
        dataset_dict = {
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "instances": target,
            "relationships": relationships,
            "file_name": self.filenames[index],
            'image_id': index,
            'height': image.shape[0],
            'width': image.shape[1]
        }

        return dataset_dict

    def __len__(self):
        return len(self.idx_list)

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]

        entities_box = self.gt_boxes[index].reshape(-1, 4)  # guard against no boxes
        entities_labels = self.gt_classes[index]

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index, relation,
                                                             self.repeat_dict,
                                                             self.drop_rate)
        else:
            relation_non_masked = relation

        # add relation to target
        num_box = len(entities_box)

        relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)

        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None:
                        relation_map_non_masked[int(relation_non_masked[i, 0]),
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None:
                    relation_map_non_masked[int(relation_non_masked[i, 0]),
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])

        anno = {
            "bbox_mode": BoxMode.XYXY_ABS,
            "entities_bbox": entities_box,
            "entities_labels": entities_labels,
            "entities_labels_non_masked": entities_labels,
            # "entities_attributes": entities_attributes,
            "relation_map": relation_map.long(),
            "relation_tuple": torch.LongTensor(relation),
            "relation_label_non_masked": torch.LongTensor(relation_non_masked)[:, -1],

        }

        return anno

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)

        for i in range(len(self)):
            idx = self.idx_list[i]
            self.aspect_ratios[i] = 1
            # if self.img_info[idx]['width'] / self.img_info[idx]['height'] > 1:
            #     self.aspect_ratios[i] = 1

    def _load_annotations(self):
        """

        :param annotation_file:
        :param img_dir:
        :param img_range:
        :param filter_empty_rels:
        :return:
            image_index: numpy array corresponding to the index of images we're using
            boxes: List where each element is a [num_gt, 4] array of ground
                        truth boxes (x1, y1, x2, y2)
            gt_classes: List where each element is a [num_gt] array of classes
            relationships: List where each element is a [num_r, 3] array of
                        (box_ind_1, box_ind_2, predicate) relationships
        """

        anno_file_path = self.meta['anno_file']

        anno_file_dir = '/'.join(anno_file_path.split('/')[:-1])

        with open(anno_file_path, 'rb') as f:
            self.annotation_file = json.load(f)

        if self.split == 'train' or self.split == 'val':
            f_mode = 'train'
        else:
            f_mode = 'val'

        with open(os.path.join(anno_file_dir, '%s_images_id_list.json' % f_mode), 'r') as f:
            image_ids = json.load(f)

        # for debug
        if self.cfg.DEBUG:
            num_img = 600
        else:
            num_img = len(self.annotation_file)

        image_ids = image_ids[: num_img]

        self.annotation_file = {each: self.annotation_file[each] for each in image_ids}

        (split_mask, gt_boxes,
         gt_classes, relationships) = load_graphs(
            self.annotation_file,
            image_ids,
            self.meta['classes_to_ind'],
            self.meta['predicates_to_ind'],
            num_val_im=int(num_img * 0.08),
            mode=self.split,
            training_triplets=None,
            min_graph_size=-1,
            max_graph_size=-1,
            random_subset=False,
            filter_empty_rels=True,
            filter_zeroshots=True,
            exclude_left_right=True
        )

        img_info = []
        for i, anno in enumerate(image_ids):
            if split_mask[i]:
                image_info = {
                    'img_fn': os.path.join(self.meta["image_root"], anno + '.jpg')
                }
                img_info.append(image_info)

        return gt_boxes, gt_classes, relationships, img_info


    def _get_metadata(self):
        meta = {}
        image_root, json_file = _PREDEFINED_SPLITS_OpenImage_SGDET['gqa'][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["anno_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)

        anno_root = '/'.join(json_file.split('/')[:-1])
        meta["anno_root"] = osp.join(self.data_root, anno_root) \
            if "://" not in image_root else osp.join(image_root, anno_root)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_OpenImage_SGDET["evaluator_type"]['gqa']

        with open(os.path.join(meta["anno_root"], 'gqa_cate_metainfo.json'), 'r') as f:
            meta_data = json.load(f)

        if self.cfg.DUMP_INTERMEDITE:
            add_dataset_metadata(meta_data)
        meta.update(meta_data)

        return meta, meta_data['ind_to_classes'], meta_data['ind_to_predicates']

    def evaluate(self, predictions):
        pass

    @property
    def ground_truth_annotations(self):
        pass


# helper function
def load_image_filenames(image_ids, mode, image_dir):
    """
    Loads the image filenames from GQA from the JSON file that contains them.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the GQA images are located
    :return: List of filenames corresponding to the good images
    """
    fns = []
    for im_id in image_ids:
        basename = '{}.jpg'.format(im_id)
        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):  # comment for faster loading
            fns.append(basename)

    assert len(fns) == len(image_ids), (len(fns), len(image_ids))
    assert len(fns) == (72140 if mode in ['train', 'val'] else 10234), (len(fns), mode)
    return fns


def load_graphs(all_sgs_json, image_ids, classes_to_ind, predicates_to_ind, num_val_im=-1,
                min_graph_size=-1, max_graph_size=-1, mode='train',
                training_triplets=None, random_subset=False,
                filter_empty_rels=True, filter_zeroshots=True,
                exclude_left_right=False):
    """
    Load GT boxes, relations and dataset split
    :param graphs_file_template: template SG filename (replace * with mode)
    :param split_modes_file: JSON containing mapping of image id to its split
    :param mode: (train, val, or test)
    :param training_triplets: a list containing triplets in the training set
    :param random_subset: whether to take a random subset of relations as 0-shot
    :param filter_empty_rels: (will be filtered otherwise.)
    :return: image_index: a np array containing the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    if exclude_left_right:
        print('\n excluding some relationships from GQA!\n')
        filter_rels = []
        for rel in ['to the left of', 'to the right of']:
            filter_rels.append(predicates_to_ind[rel])
        filter_rels = set(filter_rels)

    # Load the image filenames split (i.e. image in train/val/test):
    # train - 0, val - 1, test - 2
    image_index = np.arange(len(image_ids))  # all training/test images
    if num_val_im > 0:
        if mode in ['val']:
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros(len(image_ids)).astype(np.bool)
    split_mask[image_index] = True

    print(mode, np.sum(split_mask))

    image_idxs = {}
    for i, imid in enumerate(image_ids):
        image_idxs[imid] = i

    # Get everything by SG
    boxes = []
    gt_classes = []
    relationships = []
    for imid in image_ids:

        if not split_mask[image_idxs[imid]]:
            continue

        sg_objects = all_sgs_json[imid]['objects']
        # Sort the keys to ensure object order is always the same
        sorted_oids = sorted(list(sg_objects.keys()))

        # Filter out images without objects/bounding boxes
        if len(sorted_oids) == 0:
            split_mask[image_idxs[imid]] = False
            continue

        boxes_i = []
        gt_classes_i = []
        raw_rels = []
        oid_to_idx = {}
        no_objs_with_rels = True
        for oid in sorted_oids:

            obj = sg_objects[oid]

            # Compute object GT bbox
            b = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
            try:
                assert np.all(b[:2] >= 0), (b, obj)  # sanity check
                assert np.all(b[2:] > 0), (b, obj)  # no empty box

            except:
                continue  # skip objects with empty bboxes or negative values

            oid_to_idx[oid] = len(gt_classes_i)
            if len(obj['relations']) > 0:
                no_objs_with_rels = False

            # Compute object GT class
            gt_class = classes_to_ind[obj['name']]
            gt_classes_i.append(gt_class)

            # convert to x1, y1, x2, y2
            box = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            # box = np.concatenate((b[:2] - b[2:] / 2, b[:2] + b[2:] / 2))
            boxes_i.append(box)

            # Compute relations from this object to others in the current SG
            for rel in obj['relations']:
                raw_rels.append([oid, rel['object'], rel['name']])  # s, o, r

        # Filter out images without relations - TBD
        if no_objs_with_rels:
            split_mask[image_idxs[imid]] = False
            continue

        if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-10 will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # 11-Inf will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        # Update relations to include SG object ids
        rels = []
        for rel in raw_rels:
            if rel[0] not in oid_to_idx or rel[1] not in oid_to_idx:
                continue  # skip rels for objects with empty bboxes

            R = predicates_to_ind[rel[2]]

            if exclude_left_right:
                if R in filter_rels:
                    continue

            rels.append([oid_to_idx[rel[0]],
                         oid_to_idx[rel[1]],
                         R])

        rels = np.array(rels)
        n = len(rels)
        if n == 0:
            split_mask[image_idxs[imid]] = False
            continue

        elif training_triplets:
            if random_subset:
                ind_zs = np.random.permutation(n)[:int(np.round(n / 15.))]
            else:
                ind_zs = []
                for rel_ind, tri in enumerate(rels):
                    o1, o2, R = tri
                    tri_str = '{}_{}_{}'.format(gt_classes_i[o1],
                                                R,
                                                gt_classes_i[o2])
                    if tri_str not in training_triplets:
                        ind_zs.append(rel_ind)
                        # print('%s not in the training set' % tri_str, tri)
                ind_zs = np.array(ind_zs)

            if filter_zeroshots:
                if len(ind_zs) > 0:
                    try:
                        rels = rels[ind_zs]
                    except:
                        print(len(rels), ind_zs)
                        raise
                else:
                    rels = np.zeros((0, 3), dtype=np.int32)

            if filter_empty_rels and len(ind_zs) == 0:
                split_mask[image_idxs[imid]] = False
                continue

        # Add current SG information to the dataset
        boxes_i = np.array(boxes_i)
        gt_classes_i = np.array(gt_classes_i)

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def dump_cate_metainfo(train_sgs, val_sgs):
    """
    Loads the file containing the GQA label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
             classes_to_ind: map from object classes to indices
             predicates_to_ind: map from predicate classes to indices
    """
    info = {'label_to_idx': {}, 'predicate_to_idx': {}, 'attribute_to_idx': {}}

    obj_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            obj_classes.add(obj['name'])
    ind_to_classes = sorted(list(obj_classes)) + ['__background__']
    for obj_lbl, name in enumerate(ind_to_classes):
        info['label_to_idx'][name] = obj_lbl

    rel_classes = set()
    attribute_cls = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            for rel in obj['relations']:
                rel_classes.add(rel['name'])
            # print(obj.keys())
            if obj.get('attributes') is not None:
                for att in obj['attributes']:
                    attribute_cls.add(att)

    ind_to_attribute = ['__background__'] + sorted(list(attribute_cls))
    ind_to_predicates = ['__background__'] + sorted(list(rel_classes))

    for rel_lbl, name in enumerate(ind_to_predicates):
        info['predicate_to_idx'][name] = rel_lbl

    for rel_lbl, name in enumerate(ind_to_attribute):
        info['attribute_to_idx'][name] = rel_lbl

    assert info['label_to_idx']['__background__'] == len(ind_to_classes) - 1, (
        len(ind_to_classes), info['label_to_idx']['__background__'])
    assert info['predicate_to_idx']['__background__'] == 0

    return {'ind_to_classes': ind_to_classes,
            'ind_to_predicates': ind_to_predicates,
            "ind_to_attributes": ind_to_attribute,
            'classes_to_ind': info['label_to_idx'],
            'attribute_to_idx': info['attribute_to_idx'],
            'predicates_to_ind': info['predicate_to_idx']}
