import contextlib
import copy
import io
import json
import logging
import os
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from cvpods.evaluation import DatasetEvaluator
from cvpods.evaluation.registry import EVALUATOR
from cvpods.utils import create_table_with_header, create_small_table
from cvpods.utils.distributed import get_rank
from .boxlist import BoxList
from .vg_sgg_eval_tools import SGRecall, SGNoGraphConstraintRecall, SGPairAccuracy, SGMeanRecall, SGRelVecRecall, SGZeroShotRecall, \
    SGStagewiseRecall, SGNGMeanRecall
from ..utils.distributed import comm

logger = logging.getLogger("cvpods." + __name__)


@EVALUATOR.register()
class VisualGenomeSGGEvaluator(DatasetEvaluator):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, dataset_name,
                 meta,
                 cfg,
                 distributed,
                 output_dir=None,
                 dump=False):

        self._results = OrderedDict()
        self.dataset_name = dataset_name
        self._dump = dump
        self._predictions = OrderedDict()
        self._groundtruths = OrderedDict()
        self._predictions_tmp = OrderedDict()

        self.cfg = cfg

        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = cfg.OUTPUT_DIR
        self._metadata = meta

        self._cpu_device = torch.device("cpu")

        self.dump_idx = 0
        self._dump_infos = []

    def _tasks_from_config(self, cfg):
        mode_list = ['bbox']

        if self.cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                    rel_mode = 'predcls'
                else:
                    rel_mode = 'sgcls'
            else:
                rel_mode = 'sgdet'

            mode_list.append(rel_mode)

        return mode_list

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = OrderedDict()
        self._groundtruths = OrderedDict()

    def process(self, inputs, outputs):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            in VG dataset, we have:
                dataset_dict = {
                    'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
                    "instances": target,
                    "relationships": relationships,
                    "file_name": self.filenames[index],
                    'image_id': index,
                    'height': image.shape[0],
                    'width': image.shape[1]
                }
            output: dump things back to the BoxList form ientical with the
                    initial SGG codebase for adapt the evaluation protocol
        """

        # transforms the data into the required format
        for input, output in zip(inputs, outputs):
            # assert  output["width"] == input["width"] and  input["height"] == output["height"]
            image_id = input['image_id']

            image_width = input["width"]  # the x axis of image
            image_height = input["height"]  # the y axis of images
            image_size = (image_width, image_height)
            bbox = input["instances"].gt_boxes.tensor
            groundtruths = BoxList(bbox, image_size, mode="xyxy")
            groundtruths.add_field('labels', input["instances"].gt_classes)

            if self.cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
                gt_relationships = input["relationships"]
                groundtruths.add_field('relation_tuple', gt_relationships.relation_tuple)

            self._groundtruths[image_id] = groundtruths.to(self._cpu_device)

            if self.cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
                pred_instances = output["relationships"].instances
            else:
                pred_instances = output["instances"]

            pred_instances = output["instances"]

            image_height, image_width = pred_instances.image_size
            image_size = (image_width, image_height)
            bbox = pred_instances.pred_boxes.tensor
            prediction = BoxList(bbox, image_size, mode="xyxy")
            prediction.add_field('pred_labels', pred_instances.pred_classes)
            prediction.add_field('pred_scores', pred_instances.scores)
            prediction.add_field('pred_score_dist', pred_instances.pred_score_dist)
            prediction.add_field('image_id', image_id)

            if self.cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
                pred_relationships = output["relationships"]
                # obtain the related relationships predictions attributes
                prediction.add_field('rel_pair_idxs', pred_relationships.rel_pair_tensor)
                prediction.add_field('pred_rel_dist', pred_relationships.pred_rel_dist)
                prediction.add_field('pred_rel_score', pred_relationships.pred_rel_scores)
                prediction.add_field('pred_rel_label', pred_relationships.pred_rel_classs)

                if pred_relationships.has('pred_rel_confidence'):
                    prediction.add_field('rel_confidence', pred_relationships.pred_rel_confidence)

                if pred_relationships.has('rel_vec'):
                    prediction.add_field('rel_vec', pred_relationships.rel_vec)

                if pred_relationships.has('match_sub_entities_indexing'):
                    for k in ["match_sub_entities_indexing", "match_obj_entities_indexing",
                              "match_sub_entities_indexing_rule", "match_obj_entities_indexing_rule", ]:
                        prediction.add_field(k, pred_relationships.get(k))

                    for k in pred_relationships.meta_info_fields():
                        if 'acc' in k:
                            prediction.add_field(k, pred_relationships.get_meta_info(k))

            self._predictions[image_id] = prediction.to(self._cpu_device)
            self._predictions_tmp[image_id] = prediction.to(self._cpu_device)
        # if self._dump:
        #     if sys.getsizeof(self._predictions_tmp) >  2**16:
        #         if self._dump and self._output_dir is not None:
        #             out_dir = os.path.join(self._output_dir, "inference_new", self.dataset_name)
        #             if not os.path.exists(out_dir):
        #                 os.makedirs(out_dir)
        #             file_path = os.path.join(out_dir, f"inference_prediction{get_rank()}-{self.dump_idx}.pkl")
        #             print("prediction is saving to", file_path)
        #             torch.save(self._predictions_tmp, file_path)
        #             self.dump_idx += 1
        #             self._predictions_tmp = OrderedDict()

    def chunk_gather(self):

        predictions = self._predictions
        groundtruths = self._groundtruths

        print("before gather", len(predictions))
        # print("before gather", len(groundtruths))

        # gather things in trucks
        # due to the relation prediction is larger than detection (we may take 2048 relation pairs)
        # to avoid the OOM if GPU, we do the gather in trucks
        if self._distributed:

            predictions_gathered = OrderedDict()
            groundtruths_gathered = OrderedDict()

            assert len(groundtruths) == len(predictions)

            chunk_size = 512

            id_list = [ids for ids in groundtruths.keys()]

            for truck_num in range(len(id_list) // chunk_size + 1):
                # print(comm.get_rank(), "gather truck", truck_num)

                gts = comm.gather({
                    sel_id: groundtruths[sel_id]
                    for sel_id in id_list[truck_num * chunk_size: (truck_num + 1) * chunk_size]
                }, dst=0)

                preds = comm.gather({
                    sel_id: predictions[sel_id]
                    for sel_id in id_list[truck_num * chunk_size: (truck_num + 1) * chunk_size]
                }, dst=0)

                if comm.is_main_process():
                    for p, gt in zip(preds, gts):
                        predictions_gathered.update(p)
                        groundtruths_gathered.update(gt)

        # only main process do evaluation
        if comm.get_rank() != 0:
            return

        predictions_list = []
        groundtruths_list = []
        for im_id, pred in predictions_gathered.items():
            predictions_list.append(pred)
            groundtruths_list.append(groundtruths_gathered[im_id])

        self._predictions = predictions_list
        self._groundtruths = groundtruths_list

        print("after gather", len(self._predictions))
        # print("after gather", len(self._groundtruths))

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """

        # dump the remaining results 

        if self._dump and self._output_dir is not None:
            out_dir = os.path.join(self._output_dir, "inference_new", self.dataset_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            file_path = os.path.join(out_dir, f"inference_prediction{get_rank()}-{self.dump_idx}.pkl")
            print("prediction is saving to", file_path)
            torch.save(self._predictions_tmp, file_path)
            self.dump_idx += 1
            self._predictions_tmp = OrderedDict()

        eval_types = ['bbox']

        self.chunk_gather()

        # only main process do evaluation
        # return empty for following procedure
        if comm.get_rank() != 0:
            return {}

        if self.cfg.MODEL.ROI_RELATION_HEAD.ENABLED:
            eval_types.append('relation')

        # evaluate the entities_box

        predictions = self._predictions
        groundtruths = self._groundtruths

        if self._dump and self._output_dir is not None:
            out_dir = os.path.join(self._output_dir, "inference", self.dataset_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            file_path = os.path.join(out_dir, "inference_prediction.pkl")
            print("prediction is saving to", file_path)
            torch.save(predictions, file_path)

        if "bbox" in eval_types:
            # create a Coco-like object that we can use to evaluate detection!
            for longtail_set in [
                # [31, 20, 48, 30, 22, 29, 8, 50, 21, 1, 43, 49, 40, 23, 38, 41],
                #                 [6, 7, 33, 11, 46, 16, 47, 25, 19, 5, 9, 35, 24, 10, 4, 14, 13],
                #                 [12, 36, 44, 42, 32, 2, 45, 28, 26, 3, 17, 18, 34, 37, 27, 39, 15],
                None]:

                anns = []
                for image_id, gt in enumerate(groundtruths):
                    ent_id = set()
                    if longtail_set is not None:
                        for each in gt.get_field('relation_tuple').tolist():
                            # selected entity categories
                            if each[-1] in longtail_set:
                                ent_id.add(each[0])
                                ent_id.add(each[1])
                    else:
                        ent_id = range(len(gt.get_field('labels')))

                    ent_id = torch.Tensor(list(ent_id)).long()
                    labels = gt.get_field('labels')[ent_id].tolist()  # integer
                    boxes = gt.bbox[ent_id].tolist()  # xyxy
                    for cls, box in zip(labels, boxes):
                        anns.append({
                            'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                            'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                            'category_id': cls,
                            'id': len(anns),
                            'image_id': image_id,
                            'iscrowd': 0,
                        })
                fauxcoco = COCO()

                fauxcoco.dataset = {
                    'info': {'description': 'use coco script for vg detection evaluation'},
                    'images': [{'id': i} for i in range(len(groundtruths))],
                    'categories': [
                        {'supercategory': 'person', 'id': i, 'name': name}
                        for i, name in enumerate(self._metadata.ind_to_classes) if name != '__background__'
                    ],
                    'annotations': anns,
                }
                fauxcoco.createIndex()

                # format predictions to coco-like
                cocolike_predictions = []
                for image_id, prediction in enumerate(predictions):
                    box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
                    score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
                    label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
                    # for predcls, we set label and score to groundtruth
                    if 'predcls' in self._tasks:
                        label = prediction.get_field('labels').detach().cpu().numpy()
                        score = np.ones(label.shape[0])
                        assert len(label) == len(box)
                    image_id = np.asarray([image_id] * len(box))
                    cocolike_predictions.append(
                        np.column_stack((image_id, box, score, label))
                    )
                    # logger.info(cocolike_predictions)
                cocolike_predictions = np.concatenate(cocolike_predictions, 0)

                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    # evaluate via coco API
                    res = fauxcoco.loadRes(cocolike_predictions)
                    coco_eval = COCOeval(fauxcoco, res, 'bbox')
                    coco_eval.params.imgIds = list(range(len(groundtruths)))
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    # obtain the coco api printed output to the string
                    coco_eval.summarize()

                coco_summary = redirect_string.getvalue()

                logger.info(f"\n{coco_summary}")

                res = self._derive_coco_results(
                    coco_eval, "bbox", redirect_string,
                    class_names=self._metadata.ind_to_classes[:-1]
                )
                self._results["bbox"] = res

        if 'relation' in eval_types:
            predicates_categories = self._metadata.ind_to_predicates
            _, vg_sgg_eval_res = classic_vg_sgg_evaluation(self.cfg,
                                                           predictions,
                                                           groundtruths,
                                                           predicates_categories,
                                                           self.cfg.OUTPUT_DIR,
                                                           logger)

            self._results['sgg_vg_metrics'] = vg_sgg_eval_res

        return copy.deepcopy(self._results)

    def _derive_coco_results(self, coco_eval, iou_type, summary, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str): specific evaluation task,
                optional values are: "bbox", "segm", "keypoints".
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        small_table = create_small_table(results)
        logger.info("Evaluation results for {}: \n".format(iou_type) + small_table)
        if not np.isfinite(sum(results.values())):
            logger.info("Note that some metrics cannot be computed.")

        if class_names is None:  # or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]

        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        try:
            ent_longtail_part = self.cfg.DATASETS.ENTITY_LONGTAIL_DICT
            results_per_category = {}
            recall_per_category = {}
            long_tail_part_res = defaultdict(list)
            long_tail_part_res_recall = defaultdict(list)
            results_per_category_show = {}

            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]

                recall = recalls[:, idx, 0, -1]
                recall = recall[recall > -1]

                if precision.size:
                    ap = np.mean(precision)
                    recall = np.mean(recall)
                    long_tail_part_res[ent_longtail_part[idx]].append(ap)
                    long_tail_part_res_recall[ent_longtail_part[idx]].append(recall)
                else:
                    ap = float("nan")
                    recall = float("nan")
                    long_tail_part_res[ent_longtail_part[idx]].append(0)
                    long_tail_part_res_recall[ent_longtail_part[idx]].append(0)

                results_per_category_show[f'{name} {ent_longtail_part[idx]}'] = float(ap * 100)

                results_per_category[name] = float(ap * 100)
                recall_per_category[name] = float(recall * 100)

            (fig,
             longtail_part_performance) = self.per_class_performance_dump(results_per_category,
                                                                          long_tail_part_res)
            # tabulate it
            table = create_table_with_header(results_per_category_show, headers=["category", "AP"])
            logger.info("Per-category {} AP: \n".format(iou_type) + table)

            table = create_table_with_header(recall_per_category, headers=["category", "recall"])
            logger.info("Per-category {} recall: \n".format(iou_type) + table)

            save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_ap_per_cls.png")
            fig.savefig(save_file, dpi=300)
            logger.info("Longtail part {} AP: \n".format(iou_type) + longtail_part_performance)

            (fig,
             longtail_part_performance) = self.per_class_performance_dump(recall_per_category,
                                                                          long_tail_part_res_recall)
            save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_recall_per_cls.png")
            fig.savefig(save_file, dpi=300)
            logger.info("Longtail part {} recall: \n".format(iou_type) + longtail_part_performance)

            results.update({"AP-" + name: ap for name, ap in results_per_category.items()})
        except :
            pass

        return results

    def per_class_performance_dump(self, results_per_category, long_tail_part_res):
        ent_sorted_cls_list = self.cfg.DATASETS.ENTITY_SORTED_CLS_LIST
        cate_names = []
        ap_sorted = []
        for i in ent_sorted_cls_list:
            cate_name = self._metadata.ind_to_classes[i]
            cate_names.append(cate_name)

            if results_per_category.get(cate_name) is not None:
                ap_sorted.append(results_per_category[cate_name])
            else:
                ap_sorted.append(0)

        fig, axs_c = plt.subplots(1, 1, figsize=(18, 5), tight_layout=True)
        fig.set_facecolor((1, 1, 1))
        axs_c.bar(cate_names, ap_sorted, width=0.6, zorder=0)
        axs_c.grid()
        plt.sca(axs_c)
        plt.xticks(rotation=-90, )

        longtail_part_performance = ''
        for k, name in zip(['h', 'b', 't'], ['head', 'body', 'tail']):
            longtail_part_performance += f'{name}: {np.mean(long_tail_part_res[k]) * 100:.2f}; '

        return fig, longtail_part_performance


def classic_vg_sgg_evaluation(
        cfg,
        predictions,
        groundtruths,
        predicates_categories: list,
        output_folder,
        logger,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load(
        "/public/home/lirj2/projects/sgtr_release/datasets/vg/vg_motif_anno/zeroshot_triplet.pytorch",
        map_location=torch.device("cpu")).long().numpy()

    attribute_on = False
    num_attributes = 1

    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES + 1
    # multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    avg_metrics = 0
    result_str = '\n' + '=' * 100 + '\n'

    result_dict = {}
    result_dict_list_to_log = []

    result_str = '\n'
    evaluator = {}
    rel_eval_result_dict = {}
    # tradictional Recall@K
    eval_recall = SGRecall(rel_eval_result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # no graphical constraint
    eval_nog_recall = SGNoGraphConstraintRecall(cfg, rel_eval_result_dict)
    eval_nog_recall.register_container(mode)
    evaluator['eval_nog_recall'] = eval_nog_recall

    # test on different distribution
    eval_zeroshot_recall = SGZeroShotRecall(rel_eval_result_dict)
    eval_zeroshot_recall.register_container(mode)
    evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

    # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
    eval_pair_accuracy = SGPairAccuracy(rel_eval_result_dict)
    eval_pair_accuracy.register_container(mode)
    evaluator['eval_pair_accuracy'] = eval_pair_accuracy

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(rel_eval_result_dict, num_rel_category, predicates_categories,
                                    print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # used for NG-meanRecall@K
    eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, predicates_categories,
                                         print_detail=True)
    eval_ng_mean_recall.register_container(mode)
    evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

    eval_stagewise_recall = SGStagewiseRecall(cfg, predicates_categories, rel_eval_result_dict)
    eval_stagewise_recall.register_container(mode)
    evaluator['eval_stagewise_recall'] = eval_stagewise_recall

    
    eval_rel_vec_recall = SGRelVecRecall(cfg, result_dict, predicates_categories)
    eval_rel_vec_recall.register_container(mode)
    evaluator['eval_rel_vec_recall'] = eval_rel_vec_recall

    # prepare all inputs
    global_container = {}
    global_container['zeroshot_triplet'] = zeroshot_triplet
    global_container['result_dict'] = rel_eval_result_dict
    global_container['mode'] = mode
    # global_container['multiple_preds'] = multiple_preds
    global_container['num_rel_category'] = num_rel_category
    global_container['iou_thres'] = iou_thres
    global_container['attribute_on'] = attribute_on
    global_container['num_attributes'] = num_attributes

    indexing_acc = defaultdict(list)
    logger.info("evaluating relationship predictions..")
    for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(predictions)):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        if prediction.has_field("ent_idx_acc_top1"):
            for k in prediction.fields():
                if 'acc' in k:
                    v = prediction.get_field(k)
                    if not torch.isnan(v).any():
                        indexing_acc[k].append(v.item())

    indexing_acc_res_str = ""
    for k, v in indexing_acc.items():
        if len(v) > 0:
            v = np.array(v)
            indexing_acc_res_str += f'{k}: {np.mean(v):.3f}\n'

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)
    eval_ng_mean_recall.calculate_mean_recall(mode)


    def generate_eval_res_dict(evaluator, mode):
        res_dict = {}
        for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
            res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
        return res_dict

    def longtail_part_eval(evaluator, mode):
        longtail_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        assert "mean_recall" in evaluator.type
        res_dict = {}
        res_str = "\nlongtail part recall:\n"
        for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
            part_recall = {"h": [], "b": [], "t": [], }
            for idx, each_cat_recall in enumerate(cate_rec_list):
                part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)
            res_dict[f"sgdet_longtail_part_recall/top{topk}/head"] = np.mean(part_recall['h'])
            res_dict[f"sgdet_longtail_part_recall/top{topk}/body"] = np.mean(part_recall['b'])
            res_dict[f"sgdet_longtail_part_recall/top{topk}/tail"] = np.mean(part_recall['t'])
            res_str += f"Top{topk:4}: head: {np.mean(part_recall['h']):.4f} " \
                       f"body: {np.mean(part_recall['b']):.4f} " \
                       f"tail: {np.mean(part_recall['t']):.4f}\n"

        return res_dict, res_str

    def longtail_part_stagewise_eval(evaluator, mode):
        longtail_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        res_dict = {}
        res_str = "\nStagewise longtail part recall:\n"
        for hit_type, stat in evaluator.relation_per_cls_hit_recall.items():
            stat = stat[-1]
            recall_score = (stat[:, 0] / (stat[:, 1] + 1e-5))[1:].tolist()
            part_recall = {"h": [], "b": [], "t": [], }
            for idx, each_cat_recall in enumerate(recall_score):
                part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)

            res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top100/head"] = np.mean(part_recall['h'])
            res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top100/body"] = np.mean(part_recall['b'])
            res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top100/tail"] = np.mean(part_recall['t'])
            res_str += f"{hit_type}: head: {np.mean(part_recall['h']):.4f} " \
                       f"body: {np.mean(part_recall['b']):.4f} " \
                       f"tail: {np.mean(part_recall['t']):.4f}\n"
        res_str += '\n'
        return res_dict, res_str

    # show the distribution & recall_count
    pred_counter_dir = os.path.join(cfg.OUTPUT_DIR, "pred_counter.pkl")
    if os.path.exists(pred_counter_dir):
        with open(pred_counter_dir, 'rb') as f:
            pred_counter = pickle.load(f)

        def show_per_cls_performance_and_frequency(mean_recall_evaluator, per_cls_res_dict):
            cls_dict = mean_recall_evaluator.rel_name_list
            cate_recall = []
            cate_num = []
            cate_set = []
            counter_name = []
            for cate_set_idx, name_set in enumerate([HEAD, BODY, TAIL]):
                for cate_id in name_set:
                    cate_set.append(cate_set_idx)
                    counter_name.append(cls_dict[cate_id - 1])  # list start from 0
                    cate_recall.append(per_cls_res_dict[cate_id - 1])  # list start from 0
                    cate_num.append(pred_counter[cate_id])  # dict start from 1

            def min_max_norm(data):
                return (data - min(data)) / max(data)

            cate_num = min_max_norm(np.array(cate_num))
            cate_recall = np.array(cate_recall)
            # cate_recall = min_max_norm(np.array(cate_recall))

            fig, axs_c = plt.subplots(1, 1, figsize=(13, 5), tight_layout=True)
            pallte = ['r', 'g', 'b']
            color = [pallte[idx] for idx in cate_set]
            axs_c.bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
            axs_c.scatter(counter_name, cate_recall, color='k', zorder=10)

            plt.xticks(rotation=-90, )
            axs_c.grid()
            fig.set_facecolor((1, 1, 1))

            global eval_times
            eval_times += 1
            save_file = os.path.join(cfg.OUTPUT_DIR,
                                     f"rel_freq_dist2recall-{mean_recall_evaluator.type}-{eval_times}.png")
            fig.savefig(save_file, dpi=300)

        per_cls_res_dict = eval_mean_recall.result_dict[f'{mode}_{eval_mean_recall.type}_list'][100]
        show_per_cls_performance_and_frequency(eval_mean_recall, per_cls_res_dict)

        per_cls_res_dict = eval_ng_mean_recall.result_dict[f'{mode}_{eval_ng_mean_recall.type}_list'][100]
        show_per_cls_performance_and_frequency(eval_ng_mean_recall, per_cls_res_dict)

    longtail_part_res_dict, longtail_part_res_str = longtail_part_eval(eval_mean_recall, mode)
    ng_longtail_part_res_dict, ng_longtail_part_res_str = longtail_part_eval(eval_ng_mean_recall, mode)
    stgw_longtail_part_res_dict, stgw_longtail_part_res_str = longtail_part_stagewise_eval(eval_stagewise_recall, mode)

    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_nog_recall.generate_print_string(mode)
    result_str += eval_zeroshot_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    result_str += eval_ng_mean_recall.generate_print_string(mode)
    result_str += eval_stagewise_recall.generate_print_string(mode)
    result_str += eval_rel_vec_recall.generate_print_string(mode)
    result_str += longtail_part_res_str
    result_str += stgw_longtail_part_res_str
    result_str += f"(Non-Graph-Constraint) {ng_longtail_part_res_str}"

    if len(indexing_acc_res_str) > 0:
        result_str += "indexing module acc: \n" + indexing_acc_res_str

    result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),
                                    generate_eval_res_dict(eval_nog_recall, mode),
                                    generate_eval_res_dict(eval_zeroshot_recall, mode),
                                    generate_eval_res_dict(eval_mean_recall, mode),
                                    generate_eval_res_dict(eval_ng_mean_recall, mode),
                                    eval_stagewise_recall.generate_res_dict(mode),
                                    longtail_part_res_dict, ng_longtail_part_res_dict])

    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        result_str += eval_pair_accuracy.generate_print_string(mode)

    result_str += '=' * 100 + '\n'
    # average the all recall and mean recall with the weight
    avg_metrics = np.mean(rel_eval_result_dict[mode + '_recall'][100]) * 0.5 \
                  + np.mean(rel_eval_result_dict[mode + '_mean_recall'][100]) * 0.5

    if output_folder:
        torch.save(rel_eval_result_dict, os.path.join(output_folder, 'result_dict.pytorch'))

    logger.info(result_str)

    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'w') as f:
            f.write(result_str)
    result_dict = {}
    for each in result_dict_list_to_log:
        result_dict.update(each)
    return float(avg_metrics), result_dict


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions},
                   os.path.join(output_folder, "eval_results.pytorch"))

        # with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # jupyter information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
            ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_score').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    local_container['rel_dist'] = prediction.get_field(
        'pred_rel_dist').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    local_container['rel_cls'] = prediction.get_field(
        'pred_rel_label').detach().cpu().numpy()  # (#pred_rels, num_pred_class)
    if prediction.has_field('rel_vec'):
        local_container['rel_vec'] = prediction.get_field('rel_vec').detach().cpu().numpy()  

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        if evaluator.get("eval_pair_accuracy") is not None:
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    if evaluator.get("eval_nog_recall") is not None:
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    if evaluator.get("eval_pair_accuracy") is not None:
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    if evaluator.get("eval_mean_recall") is not None:
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)

    if evaluator.get("eval_ng_mean_recall") is not None:
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # 
    if evaluator.get('eval_rel_vec_recall') is not None:
        evaluator['eval_rel_vec_recall'].calculate_recall(global_container, local_container, mode)

    # stage wise recall
    if evaluator.get("eval_stagewise_recall") is not None:
        evaluator['eval_stagewise_recall'] \
            .calculate_recall(mode, global_container,
                              gt_boxlist=groundtruth.convert('xyxy').to("cpu"),
                              gt_relations=groundtruth.get_field('relation_tuple').long().detach().cpu(),
                              pred_boxlist=prediction.convert('xyxy').to("cpu"),
                              pred_rel_pair_idx=prediction.get_field('rel_pair_idxs').long().detach().cpu(),
                              pred_rel_dist=prediction.get_field('pred_rel_dist').detach().cpu())
    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets
