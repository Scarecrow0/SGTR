import copy
import json
import logging
import os
import os.path as osp
import pickle
import random
from collections import OrderedDict, defaultdict, Counter
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from cvpods.data import DATASETS
from cvpods.data.base_dataset import BaseDataset
from cvpods.data.detection_utils import read_image
from cvpods.structures import BoxMode
from cvpods.utils.distributed import is_main_process, synchronize, get_rank
from .bi_lvl_rsmp import resampling_dict_generation, apply_resampling, resampling_dict_generation_ent, \
    apply_resampling_ent
from .paths_route import _PREDEFINED_SPLITS_OpenImage_SGDET
from .rel_utils import annotations_to_relationship
from .vg import get_VG_statistics, VGStanfordDataset
from ...utils.dump.intermediate_dumper import add_dataset_metadata

"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger("cvpods." + __name__)


@DATASETS.register()
class OpenImageDataset(VGStanfordDataset):

    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(VGStanfordDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.cfg = cfg

        if 'train' in dataset_name:
            self.split = 'train'
        elif 'val' in dataset_name:
            self.split = 'val'
        elif 'test' in dataset_name:
            self.split = 'test'
            self.split = 'test'

            self.name = dataset_name

        if cfg.LOAD_FROM_SHM:
            self.data_root = '/dev/shm/dataset'

        self.data_format = cfg.INPUT.FORMAT
        self.filter_empty = cfg.DATASETS.FILTER_EMPTY_ANNOTATIONS
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        self.relation_on = cfg.MODEL.ROI_RELATION_HEAD.ENABLED

        self.filter_non_overlap = cfg.DATASETS.FILTER_NON_OVERLAP and self.split == 'train'
        self.filter_duplicate_rels = cfg.DATASETS.FILTER_DUPLICATE_RELS and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = False

        (self.meta,
         self.ind_to_classes,
         self.ind_to_predicates) = self._get_metadata()  # load the dataset path and the categories informations

        self.annotation_file = None
        (self.gt_boxes,
         self.gt_classes,
         self.relationships,
         self.img_info) = self._load_annotations()

        self.filenames = [each['img_fn'] for each in self.img_info]
        self.idx_list = list(range(len(self.filenames)))
        self.id_to_imeval_with_gtg_map = {k: v for k, v in enumerate(self.idx_list)}
        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        # resampling

        self.resampling_on = False
        self.ent_resampling_on = False
        self.ent_repeat_dict = None
        self.repeat_dict= None

        self.resampling_method = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.METHOD
        assert self.resampling_method in ['bilvl', 'lvis']
        try:
            self.ent_resampling_on = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENTITY.ENABLED
        except AttributeError:
            pass

        (fg_matrix, bg_matrix,
         (rel_counter_init, ent_counter_init)) = get_VG_statistics(self, must_overlap=True)
        self.rel_counter_init = rel_counter_init
        self.ent_counter_init = ent_counter_init

        if self.relation_on:
            self.resampling_on = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENABLED

            if self.resampling_on and self.split == 'train':

                self.repeat_factor = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.REPEAT_FACTOR
                self.drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.INSTANCE_DROP_RATE

                # creat repeat dict in main process, other process just wait and load
                if get_rank() == 0:
                    self.repeat_dict = resampling_dict_generation(cfg, self, self.meta['ind_to_predicates'], logger)
                    with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
                        pickle.dump(self.repeat_dict, f)

                    logger.info("predicate_repeat_dict:")
                    for i, pred_cls in enumerate(self.ind_to_predicates):
                        logger.info(f"{pred_cls}  {self.repeat_dict['cls_rf'][i]}")

                synchronize()
                if get_rank() != 0:
                    repeat_dict_dir = os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl")
                    logger.info("load repeat_dict from " + repeat_dict_dir)
                    with open(repeat_dict_dir, 'rb') as f:
                        self.repeat_dict = pickle.load(f)
                synchronize()

                duplicate_idx_list = []
                for idx in range(len(self.filenames)):
                    r_c = self.repeat_dict[idx]
                    duplicate_idx_list.extend([idx for _ in range(r_c)])
                self.idx_list = duplicate_idx_list

        if self.ent_resampling_on and self.split == 'train':
            self.ent_repeat_factor = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENTITY.REPEAT_FACTOR
            self.ent_drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENTITY.INSTANCE_DROP_RATE

            if get_rank() == 0:
                self.ent_repeat_dict = resampling_dict_generation_ent(cfg, self, self.meta['ind_to_classes'],
                                                                      logger)
                with open(os.path.join(cfg.OUTPUT_DIR, "ent_repeat_dict.pkl"), "wb") as f:
                    pickle.dump(self.ent_repeat_dict, f)

                logger.info("ent_repeat_dict:")
                for i, ent_cls in enumerate(self.ind_to_classes):
                    logger.info(f"{ent_cls}  {self.ent_repeat_dict['cls_rf'][i]}")

            synchronize()

            if get_rank() != 0:
                repeat_dict_dir = os.path.join(cfg.OUTPUT_DIR, "ent_repeat_dict.pkl")
                logger.info("load ent_repeat_dict from " + repeat_dict_dir)
                with open(repeat_dict_dir, 'rb') as f:
                    self.ent_repeat_dict = pickle.load(f)
            synchronize()

            duplicate_idx_list = []
            for idx in range(len(self.filenames)):
                r_c = self.ent_repeat_dict[idx]
                duplicate_idx_list.extend([idx for _ in range(r_c)])
            self.idx_list = duplicate_idx_list

            self.resampling_on = False

        # dataset statistics
        if self.split == 'train':
            self._set_group_flag()

            if is_main_process():
                self.get_statistics()
            synchronize()

    def __getitem__(self, index):
        # read images
        if self.repeat_dict is not None or self.ent_repeat_dict is not None:
            index = self.idx_list[index]

        img = read_image(self.filenames[index], format=self.data_format)

        if img.shape[0] != self.img_info[index]['height'] or img.shape[1] != self.img_info[index]['width']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.shape), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

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

    def _apply_transforms(self, image, annotations=None):
        """
            Apply a list of :class:`TransformGen` on the input image, and
            returns the transformed image and a list of transforms.

            We cannot simply create and return all transforms without
            applying it to the image, because a subsequent transform may
            need the output of the previous one.

            Args:
                img (ndarray): uint8 or floating point images with 1 or 3 channels.
                annotations (dict):
            Returns:
                ndarray: the transformed image
                TransformList: contain the transforms that's used.
            """

        """
        The transform module takes the list of dict contains the each bbox:
        [ {
            bbox: [], bbox_mode: , "category_id":, iscrowd, segmentation
        }]
        we need to transform the initial box list to this form for transformation
        """

        transformed_annotations = []

        for ids in range(len(annotations['entities_bbox'])):
            transformed_annotations.append(
                {
                    'bbox': list(annotations['entities_bbox'][ids]),
                    'category_id': annotations['entities_labels'][ids],
                    'iscrowd': 0,
                    'bbox_mode': annotations['bbox_mode']
                }
            )

        def collet_list_dict_anno(annotations, initial_annotations):
            bboxs = []
            cat_ids = []
            bbox_mode = annotations[-1]['bbox_mode']

            for each in annotations:
                bboxs.append(each['bbox'])
                cat_ids.append(each['category_id'])
            initial_annotations['entities_bbox'] = np.array(bboxs)
            initial_annotations['entities_labels'] = np.array(cat_ids)
            initial_annotations['bbox_mode'] = bbox_mode

            return initial_annotations

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                img = deepcopy(image)
                annos = deepcopy(transformed_annotations)
                for tfm in tfms:
                    img, annos = tfm(img, annos)
                dataset_dict[key] = (img, collet_list_dict_anno(annos, annotations))

            return dataset_dict, None
        else:
            for tfm in self.transforms:
                image, transformed_annotations = tfm(image, transformed_annotations)

            annotations = collet_list_dict_anno(transformed_annotations, annotations)

            return image, annotations

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']

        entities_box = self.gt_boxes[index].reshape(-1, 4)  # guard against no boxes
        entities_labels = self.gt_classes[index]
        entities_labels_non_masked = copy.copy(entities_labels)
        if self.ent_repeat_dict is not None:
            (entities_labels,
             entities_labels_non_masked) = apply_resampling_ent(index, entities_labels,
                                                                self.ent_repeat_dict,
                                                                self.ent_drop_rate)

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
            "entities_labels_non_masked": entities_labels_non_masked,
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
            if self.img_info[idx]['width'] / self.img_info[idx]['height'] > 1:
                self.aspect_ratios[i] = 1

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

        anno_file_dir = self.meta['anno_file']
        self.annotation_file = json.load(open(anno_file_dir, 'r'))

        # for debug
        if self.cfg.DEBUG:
            num_img = 100
        else:
            num_img = len(self.annotation_file)

        self.annotation_file = self.annotation_file[: num_img]

        img_dir = self.meta["image_root"]

        filter_empty_rels = False if not self.relation_on and self.split == "train" else True

        empty_list = set()
        if filter_empty_rels:
            for i, each in enumerate(self.annotation_file):
                if len(each['rel']) == 0:
                    empty_list.add(i)
                if len(each['bbox']) == 0:
                    empty_list.add(i)

        print('empty relationship image num: ', len(empty_list))

        boxes = []
        entities_label = []
        relationships = []
        img_info = []
        for i, anno in enumerate(self.annotation_file):

            if i in empty_list:
                continue

            boxes_i = np.array(anno['bbox'])
            gt_classes_i = np.array(anno['det_labels'], dtype=int)

            rels = np.array(anno['rel'], dtype=int)

            # gt_classes_i += 1
            rels[:, -1] += 1

            image_info = {
                'width': anno['img_size'][0],
                'height': anno['img_size'][1],
                'img_fn': os.path.join(img_dir, anno['img_fn'] + '.jpg')
            }

            boxes.append(boxes_i)
            entities_label.append(gt_classes_i)
            relationships.append(rels)
            img_info.append(image_info)

        return boxes, entities_label, relationships, img_info

    # def get_statistics(self):
    #     data_statistics_name = ''.join(self.name.split('_')[0]) + '_statistics'
    #     save_file = os.path.join(self.cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    #
    #     if os.path.exists(save_file):
    #         logger.info('Loading data statistics from: ' + str(save_file))
    #         logger.info('-' * 100)
    #         return torch.load(save_file, map_location=torch.device("cpu"))
    #
    #     (fg_matrix, bg_matrix,
    #      (rel_counter_init, ent_counter_init)) = get_VG_statistics(self,
    #                                                                must_overlap=True)
    #     eps = 1e-3
    #     bg_matrix += 1
    #     fg_matrix[:, :, 0] = bg_matrix
    #     pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps
    #
    #     result = {
    #         'fg_matrix': torch.from_numpy(fg_matrix),
    #         'pred_dist': torch.from_numpy(pred_dist).float(),
    #         'obj_classes': self.meta['ind_to_classes'],
    #         'rel_classes': self.meta['ind_to_predicates'],
    #         # 'att_classes': self.meta['ind_to_attributes'],
    #     }
    #
    #     logger.info('Save data statistics to: ' + str(save_file))
    #     logger.info('-' * 100)
    #     torch.save(result, save_file)
    #
    #     sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]
    #
    #     # show dist after resampling
    #     rel_counter = Counter()
    #     for index in tqdm(self.idx_list):
    #         relation = self.relationships[index].copy()  # (num_rel, 3)
    #
    #         if self.filter_duplicate_rels:
    #             # Filter out dupes!
    #             assert self.split == 'train'
    #             all_rel_sets = defaultdict(list)
    #             for (o0, o1, r) in relation:
    #                 all_rel_sets[(o0, o1)].append(r)
    #             relation = [(k[0], k[1], np.random.choice(v))
    #                         for k, v in all_rel_sets.items()]
    #             relation = np.array(relation, dtype=np.int32)
    #
    #         if self.repeat_dict is not None:
    #             relation, _ = apply_resampling(index, relation,
    #                                            self.repeat_dict,
    #                                            self.drop_rate)
    #         for i in relation[:, -1]:
    #             if i > 0:
    #                 rel_counter[i] += 1
    #
    #     cate_num = []
    #     counter_name = []
    #     cate_set = []
    #     lt_part_dict = self.cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
    #     for cate_id in sorted_cate_list:
    #         if lt_part_dict[cate_id] == 'h':
    #             cate_set.append(0)
    #         if lt_part_dict[cate_id] == 'b':
    #             cate_set.append(1)
    #         if lt_part_dict[cate_id] == 't':
    #             cate_set.append(2)
    #         counter_name.append(self.meta['ind_to_predicates'][cate_id])  # list start from 0
    #         cate_num.append(rel_counter[cate_id])  # dict start from 1
    #
    #     pallte = ['r', 'g', 'b']
    #     color = [pallte[idx] for idx in cate_set]
    #
    #     fig, axs_c = plt.subplots(2, 1, figsize=(13, 10), tight_layout=True)
    #     fig.set_facecolor((1, 1, 1))
    #
    #     axs_c[0].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
    #     axs_c[0].grid()
    #     plt.sca(axs_c[0])
    #     plt.xticks(rotation=-90, )
    #
    #     axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
    #     axs_c[1].grid()
    #     axs_c[1].set_ylim(0, max(cate_num))
    #     plt.sca(axs_c[1])
    #     plt.xticks(rotation=-90, )
    #
    #     save_file = os.path.join(self.cfg.OUTPUT_DIR, f"rel_freq_dist.png")
    #     fig.savefig(save_file, dpi=300)
    #
    #     return result

    def _get_metadata(self):
        meta = {}
        image_root, json_file = _PREDEFINED_SPLITS_OpenImage_SGDET['oi'][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["anno_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)

        anno_root = '/'.join(json_file.split('/')[:-1])
        meta["anno_root"] = osp.join(self.data_root, anno_root) \
            if "://" not in image_root else osp.join(image_root, anno_root)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_OpenImage_SGDET["evaluator_type"]['oi']

        (ind_to_entities,
         ind_to_predicates,
         entities_to_ind,
         predicate_to_ind) = load_categories_info(osp.join(meta["anno_root"], 'categories_dict.json'))

        meta_data = {
            "ind_to_classes": ind_to_entities,
            "ind_to_predicates": ind_to_predicates,
            "ind_to_attributes": [],
            "entities_to_ind": entities_to_ind,
            'predicate_to_ind': predicate_to_ind
        }

        if self.cfg.DUMP_INTERMEDITE:
            add_dataset_metadata(meta_data)

        meta.update(meta_data)

        return (meta, ind_to_entities,
                ind_to_predicates,)

    def evaluate(self, predictions):
        pass

    @property
    def ground_truth_annotations(self):
        pass


def load_categories_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_entities = info['obj'] + ['__background__']
    ind_to_predicates = ['__background__'] + info['rel']
    entities_to_ind = OrderedDict({
        name: i for i, name in enumerate(ind_to_entities)
    })
    info['label_to_idx'] = entities_to_ind
    predicate_to_ind = OrderedDict({
        name: i for i, name in enumerate(ind_to_predicates)
    })
    info['predicate_to_idx'] = predicate_to_ind

    return ind_to_entities, ind_to_predicates, entities_to_ind, predicate_to_ind
