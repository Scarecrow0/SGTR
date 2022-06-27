import json
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import pickle


def resampling_dict_generation(cfg, dataset, category_list, logger):
    HEAD = []
    BODY = []
    TAIL = []
    for cate_id, set_n in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
        if cate_id == 0:
            continue

        if set_n == 'h':
            HEAD.append(cate_id)
        if set_n == 'b':
            BODY.append(cate_id)
        if set_n == 't':
            TAIL.append(cate_id)

    logger.info("using resampling method for predicates:" + dataset.resampling_method)
    repeat_dict_dir = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.REPEAT_DICT_DIR

    if repeat_dict_dir is not None:
        if not os.path.exists(repeat_dict_dir):
            if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl")):
                repeat_dict_dir = os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl")

        logger.info("load repeat_dict from " + repeat_dict_dir)
        with open(repeat_dict_dir, 'rb') as f:
            dataset.repeat_dict = pickle.load(f)

    else:
        logger.info(
            "generate the repeat dict according to hyper_param on the fly")

        if dataset.resampling_method in ["bilvl", 'lvis']:
            # when we use the lvis sampling method,
            global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.REPEAT_FACTOR
            logger.info(f"global repeat factor: {global_rf};  ")
            if dataset.resampling_method == "bilvl":
                # share drop rate in lvis sampling method
                dataset.drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.INSTANCE_DROP_RATE
                logger.info(f"drop rate: {dataset.drop_rate};")
            else:
                dataset.drop_rate = 0.0
        else:
            raise NotImplementedError(dataset.resampling_method)

        F_c = np.zeros(len(category_list))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_labs = anno.get('relation_tuple')[:, -1].numpy()
            for each_rel in tgt_rel_labs:
                F_c[each_rel] += 1

        total = sum(F_c)
        F_c /= (total + 1e-11)

        rc_cls = {
            i: 1 for i in range(len(category_list))
        }
        global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.REPEAT_FACTOR

        reverse_fc = global_rf / (F_c[1:] + 1e-11) # num_cls
        reverse_fc = np.sqrt(reverse_fc)
        final_r_c = np.clip(reverse_fc, a_min=1.0, a_max=np.max(reverse_fc) + 1)
        # quantize by random number
        rands = np.random.rand(*final_r_c.shape)
        _int_part = final_r_c.astype(int)
        _frac_part = final_r_c - _int_part
        rep_factors = _int_part + (rands < _frac_part).astype(int)

        for i, rc in enumerate(rep_factors.tolist()):
        #     if dataset.rel_counter_init[i+1] < 2000:
                rc_cls[i + 1] = int(rc)
        #     else:
        #         rc_cls[i + 1] = int(rc*0.3) if int(rc*0.3) > 0 else 1

        repeat_dict = {}
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_labs = anno.get('relation_tuple')[:, -1].numpy()

            hit_rel_labels_r_c = []
            curr_rel_lables = []

            for rel_label in tgt_rel_labs:
                if rel_label not in curr_rel_lables:
                    curr_rel_lables.append(rel_label)
                    hit_rel_labels_r_c.append(rc_cls[rel_label])

            hit_rel_labels_r_c = np.array(hit_rel_labels_r_c)

            r_c = 1
            if len(hit_rel_labels_r_c) > 0:
                r_c = int(np.max(hit_rel_labels_r_c))
            repeat_dict[i] = r_c

        repeat_dict['cls_rf'] = rc_cls

        return repeat_dict


def apply_resampling(index: int, relation: np.ndarray,
                     repeat_dict: Dict, drop_rate):
    """

    Args:
        index:
        relation: N x 3 array
        repeat_dict: r_c, rc_cls image repeat number and repeat number of each category
        drop_rate:

    Returns:

    """
    relation_non_masked = relation.copy()

    # randomly drop the head and body categories for more balance distribution
    # reduce duplicate head and body for more balances
    rc_cls = repeat_dict['cls_rf']
    r_c = repeat_dict[index]

    if r_c >= 1:
        # rc <= 1,
        # no need repeat this images, just return

        selected_rel_idx = []
        for i, each_rel in enumerate(relation):
            rel_label = each_rel[-1]

            if rc_cls.get(rel_label) is not None:
                selected_rel_idx.append(i)

        # decrease the head classes of repeated image and non-repeated image
        # if the images are repeated, the total times are > 1, then we calculate the decrease time
        # according to the repeat times.
        # head_drop_rate is reduce the head instance num from the initial.
        if len(selected_rel_idx) > 0:
            selected_head_rel_idx = np.array(selected_rel_idx, dtype=int)
            ignored_rel = np.random.uniform(0, 1, len(selected_head_rel_idx))
            img_repeat_times = r_c

            rel_repeat_time = np.array([rc_cls[rel] for rel in relation[:, -1]])
            drop_prob = (1 - (rel_repeat_time / (img_repeat_times + 1e-6))) * drop_rate

            # if img_repeat_times == 1:
            #     drop_prob[rel_repeat_time == img_repeat_times] = 0.90
            # drop_prob[rel_repeat_time <= 2] *= 3
            # drop_prob[relation[:, -1] == 31] = 0.80
            # drop_prob[relation[:, -1] == 20] = 0.50
            # drop_prob[relation[:, -1] == 48] = 0.47
            # drop_prob[relation[:, -1] == 30] = 0.35
            # drop_prob[relation[:, -1] == 22] = 0.96
            # drop_prob[relation[:, -1] == 29] = 0.97
            # drop_prob[relation[:, -1] == 8] = 0.97
            # drop_prob[relation[:, -1] == 50] = 0.96
            # drop_prob[relation[:, -1] == 21] = 0.96
            # drop_prob[relation[:, -1] == 1] = 0.96
            # drop_prob[relation[:, -1] == 43] = 0.95
            # drop_prob[relation[:, -1] == 49] = 0.96
            # drop_prob[relation[:, -1] == 40] = 0.96
            # drop_prob[relation[:, -1] == 23] = 0.96
            # drop_prob[relation[:, -1] == 38] = 0.94
            # drop_prob[relation[:, -1] == 41] = 0.92

            # drop_prob[relation[:, -1] == 31] = 0.97
            # drop_prob[relation[:, -1] == 20] = 0.90
            # drop_prob[relation[:, -1] == 48] = 0.87
            # drop_prob[relation[:, -1] == 30] = 0.78
            # drop_prob[relation[:, -1] == 22] = 0.8
            # drop_prob[relation[:, -1] == 29] = 0.8
            # drop_prob[relation[:, -1] == 8] = 0.8
            # drop_prob[relation[:, -1] == 50] = 0.8
            # drop_prob[relation[:, -1] == 21] = 0.8
            # drop_prob[relation[:, -1] == 1] = 0.8
            # drop_prob[relation[:, -1] == 43] = 0.8
            # drop_prob[relation[:, -1] == 49] = 0.8
            # drop_prob[relation[:, -1] == 40] = 0.8
            # drop_prob[relation[:, -1] == 23] = 0.8
            # drop_prob[relation[:, -1] == 38] = 0.8
            # drop_prob[relation[:, -1] == 41] = 0.8

            # drop_prob[relation[:, -1] == 6] = 0.87
            # drop_prob[relation[:, -1] == 7] = 0.87


            ignored_rel = ignored_rel < np.clip(drop_prob, 0.0, 1.0)
            # ignored_rel[rel_repeat_time >= 2] = False

            # if len(np.nonzero(ignored_rel == 0)[0]) == 0:
            #     ignored_rel[np.random.randint(0, len(ignored_rel))] = False
            selected_head_rel_idx = np.array(selected_head_rel_idx, dtype=int)
            relation[selected_head_rel_idx[ignored_rel], -1] = -1


    return relation, relation_non_masked



def resampling_dict_generation_ent(cfg, dataset, category_list, logger):

    logger.info("using resampling method for entities:" + dataset.resampling_method)
    repeat_dict_dir = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING.ENTITY.REPEAT_DICT_DIR

    if repeat_dict_dir is not None:
        if not os.path.exists(repeat_dict_dir):
            if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "repeat_dict_ent.pkl")):
                repeat_dict_dir = os.path.join(cfg.OUTPUT_DIR, "repeat_dict_ent.pkl")

        logger.info("load repeat_dict from " + repeat_dict_dir)
        with open(repeat_dict_dir, 'rb') as f:
            dataset.ent_repeat_dict = pickle.load(f)

    else:
        logger.info(
            "generate the repeat dict according to hyper_param on the fly")

        if dataset.resampling_method in ["bilvl", 'lvis']:
            # when we use the lvis sampling method,
            global_rf = dataset.ent_repeat_factor
            logger.info(f"global repeat factor: {global_rf};  ")
            if dataset.resampling_method == "bilvl":
                logger.info(f"drop rate: {dataset.ent_drop_rate};")
        else:
            raise NotImplementedError(dataset.resampling_method)

        F_c = np.zeros(len(category_list))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_ent_labs = anno.get('entities_labels')
            for each_rel in tgt_ent_labs:
                F_c[each_rel] += 1

        total = sum(F_c)
        F_c /= (total + 1e-11)

        rc_cls = {
            i: 1 for i in range(len(category_list))
        }

        reverse_fc = global_rf / (F_c[:-1] + 1e-11)
        reverse_fc = np.sqrt(reverse_fc)
        final_r_c = np.clip(reverse_fc, a_min=1.0, a_max=np.max(reverse_fc) + 1)
        # quantitize by random number
        rands = np.random.rand(*final_r_c.shape)
        _int_part = final_r_c.astype(int)
        _frac_part = final_r_c - _int_part
        rep_factors = _int_part + (rands < _frac_part).astype(int)

        for i, rc in enumerate(rep_factors.tolist()):
            rc_cls[i] = int(rc)

        repeat_dict = {}
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_ent_labs = anno.get('entities_labels')

            hit_ent_labels_r_c = []
            curr_ent_lables = []

            for ent_label in tgt_ent_labs:
                if ent_label not in curr_ent_lables:
                    curr_ent_lables.append(ent_label)
                    hit_ent_labels_r_c.append(rc_cls[ent_label])

            hit_ent_labels_r_c = np.array(hit_ent_labels_r_c)

            r_c = 1
            if len(hit_ent_labels_r_c) > 0:
                r_c = int(np.max(hit_ent_labels_r_c))
            repeat_dict[i] = r_c

        repeat_dict['cls_rf'] = rc_cls

        return repeat_dict


def apply_resampling_ent(index: int, ent_cls: np.ndarray,
                     repeat_dict: Dict, drop_rate):
    """

    Args:
        index:
        relation: N x 3 array
        repeat_dict: r_c, rc_cls image repeat number and repeat number of each category
        drop_rate:

    Returns:

    """
    ent_cls_non_masked = ent_cls.copy()

    # randomly drop the head and body categories for more balance distribution
    # reduce duplicate head and body for more balances
    rc_cls = repeat_dict['cls_rf']
    r_c = repeat_dict[index]

    if r_c >= 1:
        # rc <= 1,
        # no need repeat this images, just return
        selected_ent_idx = []
        for i, ent_label in enumerate(ent_cls):

            if rc_cls.get(ent_label) is not None:
                selected_ent_idx.append(i)

        # decrease the head classes of repeated image and non-repeated image
        # if the images are repeated, the total times are > 1, then we calculate the decrease time
        # according to the repeat times.
        # head_drop_rate is reduce the head instance num from the initial.
        if len(selected_ent_idx) > 0:
            selected_head_rel_idx = np.array(selected_ent_idx, dtype=int)
            ignored_rel = np.random.uniform(0, 1, len(selected_head_rel_idx))
            total_repeat_times = r_c

            repeat_time = np.array([rc_cls[ec] for ec in ent_cls])

            drop_rate = (1 - (repeat_time / (total_repeat_times + 1e-5))) * drop_rate
            ignored_rel = ignored_rel < np.clip(drop_rate, 0.0, 1.0)
            selected_head_rel_idx = np.array(selected_head_rel_idx, dtype=int)
            ent_cls[selected_head_rel_idx[ignored_rel]] = -1

    return ent_cls, ent_cls_non_masked
