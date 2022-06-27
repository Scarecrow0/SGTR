import json
import logging
import os
import os.path as osp
import pickle
import random
from collections import defaultdict, Counter
from copy import deepcopy

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from cvpods.data import DATASETS
from cvpods.data.base_dataset import BaseDataset
from cvpods.data.detection_utils import read_image
from cvpods.structures import Boxes, BoxMode
from cvpods.structures.boxes import matched_boxlist_iou
from cvpods.utils.distributed import is_main_process, synchronize, get_rank
from .bi_lvl_rsmp import resampling_dict_generation, apply_resampling, resampling_dict_generation_ent, \
    apply_resampling_ent
from .paths_route import _PREDEFINED_SPLITS_VG_STANFORD_SGDET, _PREDEFINED_SPLITS_VG20_SGDET
from .rel_utils import annotations_to_relationship
from ...utils.dump.intermediate_dumper import add_dataset_metadata

"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger("cvpods." + __name__)

BOX_SCALE = 1024  # Scale at which we have the boxes

META_DATA = None


@DATASETS.register()
class VGStanfordDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(VGStanfordDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.cfg = cfg

        if cfg.LOAD_FROM_SHM:
            self.data_root = '/dev/shm/dataset'

        if 'train' in dataset_name:
            self.split = 'train'
        elif 'val' in dataset_name:
            self.split = 'val'
        elif 'test' in dataset_name:
            self.split = 'test'

        self.relation_on = cfg.MODEL.ROI_RELATION_HEAD.ENABLED

        self.name = dataset_name
        self.data_format = cfg.INPUT.FORMAT
        self.filter_empty = cfg.DATASETS.FILTER_EMPTY_ANNOTATIONS

        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.filter_non_overlap = cfg.DATASETS.FILTER_NON_OVERLAP and self.split == 'train'
        self.filter_duplicate_rels = cfg.DATASETS.FILTER_DUPLICATE_RELS and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = False

        (self.meta,
         self.ind_to_classes,
         self.ind_to_predicates,
         self.ind_to_attributes) = self._get_metadata()  # load the dataset path and the categories informations

        self.check_img_file = False

        self.filenames, self.img_info = load_image_filenames(
            self.meta["image_root"],
            osp.join(self.meta["anno_file"], "image_data.json"),
            self.check_img_file  # length equals to split_mask
        )

        (self.split_mask,
         self.gt_boxes,
         self.gt_classes,
         self.gt_attributes,
         self.relationships) = self._load_annotations(
            osp.join(self.meta["anno_file"], "VG-SGG-with-attri.h5"),
        )

        # obtain the selected split annotation
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        self.procompute_proposals = None

        if self.load_proposals:
            if self.split == 'train':
                self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TRAIN
            elif self.split in ['val', 'test']:
                self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TEST

            if len(self.proposal_files) != 0:
                self.procompute_proposals = self._load_precompute_proposals()

        if cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
            HEAD = []
            BODY = []
            TAIL = []
            for i, cate in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
                if cate == 'h':
                    HEAD.append(i)
                elif cate == 'b':
                    BODY.append(i)
                elif cate == 't':
                    TAIL.append(i)

            self.meta['predicate_longtail'] = (HEAD, BODY, TAIL)

        # resampling
        self.resampling_on = False
        self.ent_resampling_on = False
        self.ent_repeat_dict = None
        self.repeat_dict = None

        self.resampling_method = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.METHOD
        assert self.resampling_method in ['bilvl', 'lvis']
        try:
            self.ent_resampling_on = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENTITY.ENABLED
        except AttributeError:
            pass

        (fg_matrix, bg_matrix,
         (rel_counter_init, ent_counter_init)) = get_VG_statistics(self, must_overlap=True)

        if self.relation_on:
            self.resampling_on = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENABLED
            assert not (self.resampling_on and self.ent_resampling_on)

            if self.resampling_on and self.split == 'train':

                self.rel_counter_init = rel_counter_init
                self.ent_counter_init = ent_counter_init

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
                self.ent_repeat_dict = resampling_dict_generation_ent(cfg, self, self.meta['ind_to_classes'], logger)
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

    def _load_precompute_proposals(self):
        print(f"load precompute_proposals {self.proposal_files}")
        pre_compute_box = {}
        pred_res_dirs = [os.path.join(self.proposal_files, each)
                         for each in os.listdir(self.proposal_files)]

        # from torch.multiprocessing import Pool
        # with Pool(4) as p:
        #     pred_res_list = p.map(torch.load, pred_res_dirs)

        for each_dir in tqdm(pred_res_dirs):
            pred_res = torch.load(os.path.join(self.proposal_files, each_dir))

            if isinstance(pred_res, dict):
                for k, v in pred_res.items():
                    pre_compute_box[int(k)] = v
            elif isinstance(pred_res, list):
                for each in pred_res:
                    pre_compute_box[int(each.get_field('image_id'))] = each
        print("check procompute_proposals")
        for idx in tqdm(self.idx_list):
            assert pre_compute_box.get(idx) is not None
        return pre_compute_box

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        entities_box = box.reshape(-1, 4)  # guard against no boxes
        entities_labels = self.gt_classes[index].copy()
        entities_attributes = self.gt_attributes[index].copy()
        entities_labels_non_masked = self.gt_classes[index].copy()

        if self.ent_repeat_dict is not None:
            (entities_labels,
             entities_labels_non_masked) = apply_resampling_ent(index, entities_labels,
                                                                self.ent_repeat_dict,
                                                                self.ent_drop_rate)

        anno = {
            "bbox_mode": BoxMode.XYXY_ABS,
            "entities_bbox": entities_box,
            "entities_labels": entities_labels,
            "entities_labels_non_masked": entities_labels_non_masked,
            "entities_attributes": entities_attributes,
        }

        if self.procompute_proposals is not None:
            bbox = self.procompute_proposals[index]
            scale_x, scale_y = w / bbox.size[0], h / bbox.size[1]
            box_array = bbox.bbox.numpy()
            box_array[:, (0, 2)] *= scale_x
            box_array[:, (1, 3)] *= scale_y
            anno['precompute_proposals'] = {
                "bbox_mode": BoxMode.XYXY_ABS,
                "entities_bbox": box_array,
                "entities_labels": bbox.get_field('pred_labels').numpy(),
                "entities_scores": bbox.get_field('pred_scores').numpy(),
                "entities_scores_dist": bbox.get_field('pred_score_dist').numpy(),
            }

        if not self.relation_on:
            return anno

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
        anno.update({
            "relation_map": relation_map.long(),
            "relation_tuple": torch.LongTensor(relation),
            "relation_label_non_masked": torch.LongTensor(relation_non_masked)[:, -1],
        })

        if relation_map_non_masked is not None:
            anno["relation_map_non_masked"] = relation_map_non_masked.long()

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

    def __getitem__(self, index):

        # transform into inner index
        inner_index = self.idx_list[index]

        # read images
        img = read_image(self.filenames[inner_index], format=self.data_format)
        # img = np.transpose(img, (1, 0, 2) )
        # check_image_size

        if img.shape[0] != self.img_info[inner_index]['height'] or img.shape[1] != self.img_info[inner_index]['width']:
            print('=' * 20, ' ERROR index ', str(inner_index), ' ', str(img.shape), ' ',
                  str(self.img_info[inner_index]['width']),
                  ' ', str(self.img_info[inner_index]['height']), ' ', '=' * 20)

        # obtain the gt as the dict form

        annotations = self.get_groundtruth(inner_index, inner_idx=True)

        # apply the transform / augmentation
        image, annotations = self._apply_transforms(img, annotations)

        if annotations.get('precompute_proposals') is not None:
            _, annotation_tmp = self._apply_transforms(img, annotations['precompute_proposals'])
            annotations['precompute_proposals'] = annotation_tmp

        # dump as the instance objects
        image_shape = image.shape[:2]
        target, relationships = annotations_to_relationship(annotations, image_shape, self.relation_on)

        # wrap back to the dict form with needed fields.
        dataset_dict = {
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "instances": target,
            "relationships": relationships,
            "file_name": self.filenames[inner_index],
            'image_id': inner_index,
            'height': image.shape[0],
            'width': image.shape[1]
        }

        return dataset_dict

    def __reset__(self):
        raise NotImplementedError

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

    def _load_annotations(self, anno_file, ):
        """
        Load a json file with COCO's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.

        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str): the directory where the images in this json file exists.
            dataset_name (str): the name of the dataset (e.g., coco_2017_train).
                If provided, this function will also put "thing_classes" into
                the metadata associated with this dataset.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.

        Returns:
            list[dict]: a list of dicts in cvpods standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )

                    Return:
            image_index: numpy array corresponding to the index of images we're using
            boxes: List where each element is a [num_gt, 4] array of ground
                        truth boxes (x1, y1, x2, y2)
            gt_classes: List where each element is a [num_gt] array of classes
            relationships: List where each element is a [num_r, 3] array of
                        (box_ind_1, box_ind_2, predicate) relationships

        Notes:
            1. This function does not read the image files.
            The results do not have the "image" field.
        """
        logger.info(f"load anno from {anno_file}")

        roi_h5 = h5py.File(anno_file, 'r')
        data_split = roi_h5['split'][:]
        split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag

        init_len = len(np.where(split_mask)[0])
        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        print("no entities img:", init_len - len(np.where(split_mask)[0]))

        if self.relation_on or self.split != 'train':
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0
            print("no rel img:", init_len - len(np.where(split_mask)[0]))
        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0

        image_index = np.where(split_mask)[0]

        # for debug
        if self.cfg.DEBUG:
            num_im = 3000
            num_val_im = 10
        else:
            num_im = -1
            num_val_im = 5000

        if num_im > -1:
            image_index = image_index[: num_im]
        if num_val_im > 0:
            if self.split == 'val':
                image_index = image_index[: num_val_im]
            elif self.split == 'train':
                image_index = image_index[num_val_im:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        all_attributes = roi_h5['attributes'][:, :]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        _relation_predicates = roi_h5['predicates'][:, 0]
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (_relations.shape[0]
                == _relation_predicates.shape[0])  # sanity check

        # Get everything by image.

        dataset_dicts = []

        boxes = []
        gt_classes = []
        gt_attributes = []
        relationships = []
        for i in range(len(image_index)):

            i_obj_start = im_to_first_box[i]
            i_obj_end = im_to_last_box[i]
            i_rel_start = im_to_first_rel[i]
            i_rel_end = im_to_last_rel[i]

            boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
            # let the foreground start from 0
            gt_classes_i = all_labels[i_obj_start: i_obj_end + 1] - 1

            # the relationship foreground start from the 1, 0 for background
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]
            if i_rel_start >= 0:
                predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
                obj_idx = _relations[i_rel_start: i_rel_end
                                                  + 1] - i_obj_start  # range is [0, num_box)
                assert np.all(obj_idx >= 0)
                assert np.all(obj_idx < boxes_i.shape[0])
                # (num_rel, 3), representing sub, obj, and pred
                rels = np.column_stack((obj_idx, predicates))
            else:
                rels = np.zeros((0, 3), dtype=np.int32)

            if self.filter_non_overlap:
                assert self.split == 'train'
                # construct BoxList object to apply boxlist_iou method
                # give a useless (height=0, width=0)
                boxes_i_obj = Boxes(boxes_i)
                inters = matched_boxlist_iou(boxes_i_obj, boxes_i_obj)
                rel_overs = inters[rels[:, 0], rels[:, 1]]
                inc = np.where(rel_overs > 0.0)[0]

                if inc.size > 0:
                    rels = rels[inc]
                else:
                    split_mask[image_index[i]] = 0
                    continue

            boxes.append(boxes_i)
            gt_classes.append(gt_classes_i)
            gt_attributes.append(gt_attributes_i)
            relationships.append(rels)

        return split_mask, boxes, gt_classes, gt_attributes, relationships

    def _get_metadata(self):
        meta = {}
        image_root, json_file = _PREDEFINED_SPLITS_VG_STANFORD_SGDET['vgs'][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["anno_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_VG_STANFORD_SGDET["evaluator_type"]['vgs']

        (ind_to_classes,
         ind_to_predicates,
         ind_to_attributes) = load_categories_info(osp.join(meta["anno_file"], 'VG-SGG-dicts-with-attri.json'))

        meta_data = {
            "ind_to_classes": ind_to_classes,
            "ind_to_predicates": ind_to_predicates,
            "ind_to_attributes": ind_to_attributes
        }

        if self.cfg.DUMP_INTERMEDITE:
            add_dataset_metadata(meta_data)

        meta.update(meta_data)

        return (meta, ind_to_classes,
                ind_to_predicates,
                ind_to_attributes)

    def get_statistics(self):
        (fg_matrix, bg_matrix,
         (rel_counter_init, ent_counter_init)) = get_VG_statistics(self, must_overlap=True)

        if self.relation_on:
            data_statistics_name = ''.join(self.name.split('_')[0]) + '_statistics'
            save_file = os.path.join(self.cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))

            if os.path.exists(save_file):
                logger.info('Loading data statistics from: ' + str(save_file))
                logger.info('-' * 100)
                return torch.load(save_file, map_location=torch.device("cpu"))

            sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]

            cate_num_init = [i[1] for i in rel_counter_init.most_common()]

            num_pred_cls = len(self.meta['ind_to_predicates'])

            eps = 1e-3
            bg_matrix += 1
            fg_matrix[:, :, 0] = bg_matrix
            pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps

            fg_matrix = torch.from_numpy(fg_matrix)
            inst_cnt = fg_matrix.permute(2, 0, 1).reshape(num_pred_cls, -1).sum(1)[1:].numpy()

            result = {
                'fg_matrix': fg_matrix,
                'pred_dist': torch.from_numpy(pred_dist).float(),
                'obj_classes': self.meta['ind_to_classes'],
                'rel_classes': self.meta['ind_to_predicates'],
                'att_classes': self.meta['ind_to_attributes'],
                'inst_cnt': inst_cnt
            }

            logger.info('Save data statistics to: ' + str(save_file))
            logger.info('-' * 100)
            torch.save(result, save_file)

            # build up the long tail set markers
            cate_set = []
            counter_name = []
            lt_part_dict = self.cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
            for cate_id in sorted_cate_list:
                if lt_part_dict[cate_id] == 'h':
                    cate_set.append(0)
                if lt_part_dict[cate_id] == 'b':
                    cate_set.append(1)
                if lt_part_dict[cate_id] == 't':
                    cate_set.append(2)
                counter_name.append(self.meta['ind_to_predicates'][cate_id])  # list start from 0
            pallte = ['r', 'g', 'b']
            color = [pallte[idx] for idx in cate_set]

            # show dist after resampling
            if self.resampling_on:
                rel_counter = Counter()
                for index in tqdm(self.idx_list):
                    relation = self.relationships[index].copy()  # (num_rel, 3)
                    if self.filter_duplicate_rels:
                        # Filter out dupes!
                        assert self.split == 'train'
                        all_rel_sets = defaultdict(list)
                        for (o0, o1, r) in relation:
                            all_rel_sets[(o0, o1)].append(r)
                        relation = [(k[0], k[1], np.random.choice(v))
                                    for k, v in all_rel_sets.items()]
                        relation = np.array(relation, dtype=np.int32)

                    if self.repeat_dict is not None:
                        relation, _ = apply_resampling(index, relation,
                                                       self.repeat_dict,
                                                       self.drop_rate)
                    for i in relation[:, -1]:
                        if i > 0:
                            rel_counter[i] += 1

                cate_num = []
                for cate_id in sorted_cate_list:
                    cate_num.append(rel_counter[cate_id])  # dict start from 1

                stats_res_real = {
                    'obj_classes': self.meta['ind_to_classes'],
                    'rel_classes': self.meta['ind_to_predicates'],
                    'att_classes': self.meta['ind_to_attributes'],
                    'inst_cnt': torch.Tensor([rel_counter[i] for i in range(num_pred_cls)])[1:].numpy()
                }
                data_statistics_name = ''.join(self.name.split('_')[0]) + '_statistics.real'
                save_file = os.path.join(self.cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
                logger.info('Save data statistics to: ' + str(save_file))
                torch.save(stats_res_real, save_file)

                fig, axs_c = plt.subplots(4, 1, figsize=(13, 22), tight_layout=True)
                fig.set_facecolor((1, 1, 1))

                axs_c[0].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
                axs_c[0].grid()
                plt.sca(axs_c[0])
                plt.xticks(rotation=-90, )

                axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
                axs_c[1].grid()
                axs_c[1].set_ylim(0, 8000)
                plt.sca(axs_c[1])
                plt.xticks(rotation=-90, )

                axs_c[2].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
                axs_c[2].bar(counter_name, cate_num, width=0.6, zorder=1)
                axs_c[2].grid()
                # axs_c[2].set_ylim(0, 5000)
                plt.sca(axs_c[2])
                plt.xticks(rotation=-90, )

                axs_c[3].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
                axs_c[3].grid()
                plt.sca(axs_c[3])
                plt.xticks(rotation=-90, )
            else:
                fig, axs_c = plt.subplots(1, 1, figsize=(13, 5), tight_layout=True)
                fig.set_facecolor((1, 1, 1))

                axs_c.bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
                axs_c.grid()
                plt.sca(axs_c)
                plt.xticks(rotation=-90, )

            save_file = os.path.join(self.cfg.OUTPUT_DIR, f"rel_freq_dist.png")
            fig.savefig(save_file, dpi=300)

        sorted_ent_cate_list = [i[0] for i in ent_counter_init.most_common()]
        self.get_ent_statistics(sorted_ent_cate_list, ent_counter_init)

        ent_longtail_part = []
        for cls_id in range(len(self.meta['ind_to_classes'][:-1])):
            if ent_counter_init[cls_id] > 15000:
                ent_longtail_part.append('h')
            elif ent_counter_init[cls_id] > 3000:
                ent_longtail_part.append('b')
            else:
                ent_longtail_part.append('t')

        if self.cfg.DEBUG:
            self.meta['ent_longtail_part'] = ['t', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't',
                                              't', 'b', 't', 't', 'b', 'b', 'h', 'b', 't', 't', 'b', 't', 'b', 't', 't',
                                              't', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 't', 't', 'b', 'b', 'b',
                                              't', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 't', 'b', 'b', 'b', 'b',
                                              'b', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 'b', 'b',
                                              'b', 't', 'h', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 'b',
                                              'h', 't', 'b', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                                              't', 't', 't', 't', 'b', 'h', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't',
                                              'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                                              'h', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'h', 't', 't', 't', 'h', 't']
            self.meta['sorted_ent_cate_list'] = [77, 135, 144, 110, 90, 21, 148, 114, 73, 60, 98, 56, 57, 125, 25, 75,
                                                 89, 86, 111, 39, 37, 44, 72, 27, 53, 65, 96, 123, 2, 143, 120, 126, 43,
                                                 59, 113, 112, 103, 3, 47, 58, 19, 133, 104, 61, 138, 134, 83, 52, 20,
                                                 99, 16, 129, 66, 74, 95, 128, 142, 42, 48, 9, 137, 63, 92, 22, 109, 18,
                                                 10, 40, 51, 76, 82, 13, 29, 17, 36, 80, 64, 136, 94, 146, 107, 79, 32,
                                                 87, 54, 149, 147, 30, 12, 14, 24, 4, 62, 97, 33, 116, 31, 70, 117, 124,
                                                 81, 23, 11, 26, 6, 108, 93, 145, 68, 121, 7, 84, 8, 46, 71, 28, 34, 15,
                                                 141, 102, 45, 131, 115, 41, 127, 132, 101, 88, 91, 122, 139, 5, 49,
                                                 100, 1, 85, 35, 119, 106, 38, 118, 105, 69, 130, 50, 78, 55, 140, 67]
        else:
            self.meta['sorted_ent_cate_list'] = ['sorted_ent_cate_list']
            self.meta['ent_longtail_part'] = ent_longtail_part


    def get_ent_statistics(self, sorted_ent_cate_list, ent_counter_init):
        counter_name = []
        for cate_id in sorted_ent_cate_list:
            counter_name.append(self.meta['ind_to_classes'][cate_id])

        figsize = (int(0.12 * len(counter_name)), 6)

        if self.ent_resampling_on:
            ent_counter = Counter()
            for index in tqdm(self.idx_list):
                entities_labels = self.gt_classes[index].copy()
                entities_labels_non_masked = self.gt_classes[index].copy()

                (entities_labels,
                 entities_labels_non_masked) = apply_resampling_ent(index, entities_labels,
                                                                    self.ent_repeat_dict,
                                                                    self.ent_drop_rate)

                for i in entities_labels:
                    ent_counter[i] += 1

            fig, axs_c = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            fig.set_facecolor((1, 1, 1))

            cate_num = []
            for cate_id in sorted_ent_cate_list:
                cate_num.append(ent_counter[cate_id])

            axs_c.bar(counter_name, cate_num, width=0.6, zorder=0)
            axs_c.grid()
            plt.sca(axs_c)
            plt.xticks(rotation=-90, )

            save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_freq_dist.png")
            fig.savefig(save_file, dpi=128)

        cate_num_init = [ent_counter_init[i] for i in sorted_ent_cate_list]
        fig, axs_c = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        fig.set_facecolor((1, 1, 1))
        axs_c.bar(counter_name, cate_num_init, width=0.6, zorder=0)
        axs_c.grid()
        plt.sca(axs_c)
        plt.xticks(rotation=-90, )

        save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_freq_dist_init.png")
        fig.savefig(save_file, dpi=128)

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        raise NotImplementedError


@DATASETS.register()
class VG20Dataset(VGStanfordDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(VG20Dataset, self).__init__(cfg, dataset_name, transforms, is_train)

    def _get_metadata(self):
        meta = {}
        image_root, json_file = _PREDEFINED_SPLITS_VG20_SGDET['vg20'][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["anno_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_VG20_SGDET["evaluator_type"]['vg20']

        (ind_to_classes,
         ind_to_predicates,
         ind_to_attributes) = load_categories_info(osp.join(meta["anno_file"], 'VG-SGG-dicts-with-attri.json'))

        meta_data = {
            "ind_to_classes": ind_to_classes,
            "ind_to_predicates": ind_to_predicates,
            "ind_to_attributes": ind_to_attributes
        }

        if self.cfg.DUMP_INTERMEDITE:
            add_dataset_metadata(meta_data)

        meta.update(meta_data)

        return (meta, ind_to_classes,
                ind_to_predicates,
                ind_to_attributes)


def clip_to_image(bbox, w, h, remove_empty=True):
    TO_REMOVE = 1
    bbox[:, 0].clamp_(min=0, max=w - TO_REMOVE)
    bbox[:, 1].clamp_(min=0, max=h - TO_REMOVE)
    bbox[:, 2].clamp_(min=0, max=w - TO_REMOVE)
    bbox[:, 3].clamp_(min=0, max=h - TO_REMOVE)
    if remove_empty:
        box = bbox
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return keep
    return bbox


def load_categories_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))

    # align the background categories as Num-Class as same with the CVPODS codebase
    for cate_name in info["label_to_idx"].keys():
        info["label_to_idx"][cate_name] -= 1

    if add_bg:
        info['label_to_idx']['__background__'] = 150
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(
        predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(
        attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file, check_img_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename) or not check_img_file:
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def get_VG_statistics(train_data, must_overlap=True):
    """save the initial data distribution for the frequency bias model

    Args:
        train_data ([type]): the self
        must_overlap (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes,
                          num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    rel_counter = Counter()
    ent_counter = Counter()
    for ex_ind in range(len(train_data.img_info)):
        gt_classes = train_data.gt_classes[ex_ind]
        gt_boxes = train_data.gt_boxes[ex_ind]

        for ent_cls in gt_classes:
            ent_counter[ent_cls] += 1

        # For the foreground, we'll just look at everything
        if train_data.relation_on:
            gt_relations = train_data.relationships[ex_ind]
            o1o2 = gt_classes[gt_relations[:, :2]]

            for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                fg_matrix[o1, o2, gtr] += 1
                rel_counter[gtr] += 1
            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix, (rel_counter, ent_counter)


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(
        np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, : 2],
                    boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:],
                    boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter
