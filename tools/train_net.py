# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in cvpods.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use cvpods as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import json
import logging
import os
import pickle as pkl
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from colorama import Fore, Style
from git import Git

from cvpods.checkpoint import DetectionCheckpointer
from cvpods.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from cvpods.engine.trainer import AMPTrainer
from cvpods.evaluation import build_evaluator, verify_results
from cvpods.modeling import GeneralizedRCNNWithTTA
from cvpods.utils import comm
from cvpods.utils.distributed import get_rank



torch.multiprocessing.set_sharing_strategy('file_system')


# set on may slow down
# torch.autograd.set_detect_anomaly(True)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        dump_train = cfg.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("cvpods.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class AMPTrainerApply(AMPTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        dump_train = cfg.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("cvpods.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def stage_main(args, cfg, build):

    cfg.merge_from_list(args.opts)

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")
    debug_flag = f"{'(debug)' if cfg.DEBUG else ''}"

    cfg.EXPERIMENT_NAME = f'{time_str}-{cfg.EXPERIMENT_NAME}{debug_flag}'

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME)

    cfg, logger = default_setup(cfg, args)

    try:
        if not args.skip_git_commit and not debug_flag:
            if get_rank() == 0:
                commit_code(logger, cfg)

        if get_rank() == 0:
            fetch_git_status(logger)

    except :
        logger.warning("skip the git operation.")

    model_build_func = build
    """
    save the whole config file as a json
    """
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.json'), 'w') as f:
        f.write(json.dumps(cfg, indent=3))


    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """

    if cfg.TRAINER.FP16.ENABLED:
        logger.info('FP16 mixed percision On!')
        trainer = AMPTrainerApply(cfg, model_build_func)
    else:
        trainer = Trainer(cfg, model_build_func)

    trainer.resume_or_load(resume=args.resume, load_mapping=cfg.MODEL.WEIGHTS_LOAD_MAPPING)

    if args.eval_only:
        DetectionCheckpointer(
            trainer.model, save_dir=cfg.OUTPUT_DIR, resume=args.resume).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, trainer.model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, trainer.model))
        return res

    # check wheather worksapce has enough storeage space
    # assume that a single dumped model is 700Mb
    file_sys = os.statvfs(cfg.OUTPUT_DIR)
    free_space_Gb = (file_sys.f_bfree * file_sys.f_frsize) / 2**30
    eval_space_Gb = (cfg.SOLVER.LR_SCHEDULER.MAX_ITER // (cfg.SOLVER.CHECKPOINT_PERIOD + 1e-3) ) * 700 / 2**10
    if eval_space_Gb > free_space_Gb:
        logger.warning(f"{Fore.RED}Remaining space({free_space_Gb}GB) "
                       f"is less than ({eval_space_Gb}GB){Style.RESET_ALL}")

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    trainer.train()

    if comm.is_main_process() and cfg.MODEL.AS_PRETRAIN:
        # convert last ckpt to pretrain format
        convert_to_pretrained_model(
            input=os.path.join(cfg.OUTPUT_DIR, "model_final.pth"),
            save_path=os.path.join(cfg.OUTPUT_DIR, "model_final_pretrain_weight.pkl")
        )




def convert_to_pretrained_model(input, save_path):
    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder_q.") and not k.startswith("network"):
            continue
        old_k = k
        if k.startswith("encoder_q."):
            k = k.replace("encoder_q.", "")
        elif k.startswith("network"):
            k = k.replace("network.", "")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {
        "model": newmodel,
        "__author__": "MOCO" if k.startswith("encoder_q.") else "CLS",
        "matching_heuristics": True
    }

    with open(save_path, "wb") as f:
        pkl.dump(res, f)


def main(args):

    from config import config  # noqa: E402
    from net import build_model  # noqa: E402

    if isinstance(config, list):
        assert isinstance(build_model, list) and len(config) == len(build_model)
        for cfg, build in zip(config, build_model):
            stage_main(args, cfg, build)
    else:
        stage_main(args, config, build_model)


# fetch git status
def fetch_git_status(logger):
    git = Git("../../../..")
    commit_log = git.log().split("\n")
    branch_name = git.branch().split("\n")
    curr_branch = "master"
    for each in branch_name:
        if each.startswith("*"):
            curr_branch = each.strip("*").strip(" ")

    commit_id = commit_log[0]
    commit_date = commit_log[2]
    commit_comment = commit_log[4]
    status = git.status()

    logger.info(
        "\ncodebase git HEAD info:\nbranch: %s\n%s\n%s\n%s"
        % (curr_branch, commit_id, commit_date, commit_comment)
    )
    if "working directory clean" not in status and "working tree clean" not in status:
        logger.warning(
            "there has some un-commit modify in codebase, may cause this experiment un-reproducible"
        )


def commit_code(logger, cfg):
    stream = os.popen(f"git commit -a -m 'run for experiment: {cfg.OUTPUT_DIR}'")
    output = stream.read()
    logger.info("commit code success")
    logger.info(output)


if __name__ == "__main__":

    sys.path.insert(0, '.')
    from config import config  # noqa: E402
    from net import build_model  # noqa: E402

    args = default_argument_parser().parse_args()

    if not args.disable_gpu_check:
        gpu_list = comm.get_gpu_status()
        while len(gpu_list) < args.num_gpus:
            time.sleep(2)  # wait 2 seconds for scan
            gpu_list = comm.get_gpu_status()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list)[1:-1]

    if isinstance(config, list):
        assert len(config) > 0
        print("soft link first config in list to {}".format(config[0].OUTPUT_DIR))
        config[0].link_log()
    else:
        print("soft link to {}".format(config.OUTPUT_DIR))
        config.link_log()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
