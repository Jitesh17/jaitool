# -*- coding: utf-8 -*-
import copy
import itertools
import json
import logging
import os
import random
import sys
import weakref
from asyncio import tasks
from datetime import datetime
from functools import partial
from sys import exit as x
from typing import List, Optional, OrderedDict, Tuple, Union

import albumentations as A
import cv2
import jaitool.inference.d2_infer
import numpy as np
import printj  # pip install printj
import pyjeasy.file_utils as f
import shapely
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (DefaultPredictor, DefaultTrainer,
                               create_ddp_model, hooks)
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluator,
                                   inference_on_dataset, print_csv_format,
                                   verify_results)
from detectron2.utils import comm
from detectron2.utils.events import (CommonMetricPrinter, JSONWriter,
                                     TensorboardXWriter)
from detectron2.utils.logger import create_small_table, setup_logger
from jaitool.aug.augment import get_augmentation
from jaitool.aug.augment_loader import AugmentedLoader
from jaitool.draw import draw_bbox, draw_keypoints, draw_mask_bool
from jaitool.evaluation import LossEvalHook
from jaitool.structures.bbox import BBox
from pyjeasy import file_utils as f
from pyjeasy.check_utils import check_value
from pyjeasy.file_utils import (delete_dir, delete_dir_if_exists, dir_exists,
                                dir_files_list, file_exists, make_dir,
                                make_dir_if_not_exists)
from pyjeasy.image_utils import show_image
from shapely.geometry import Polygon
from tabulate import tabulate
from tqdm import tqdm


class D2Trainer:
    def __init__(
            self,
            coco_ann_path: str, img_path: str,
            val_coco_ann_path: str, val_img_path: str,
            output_dir_path: str, resume: bool = True,
            class_names: List[str] = None, num_classes: int = None,
            keypoint_names: List[str] = None, num_keypoints: int = None,
            model: str = "mask_rcnn_R_50_FPN_1x",
            instance_train: str = "training_instance1",
            min_size_train: int = None,
            max_size_train: int = None,
            min_size_test: int = None,
            max_size_test: int = None,
            max_iter: int = 10000,
            batch_size_per_image: int = 512,
            checkpoint_period: int = None,
            score_thresh: int = None,
            key_seg_together: bool = False,
            aug_on: bool = True,
            train_val: bool = False,
            aug_settings_file_path: str = None,
            aug_vis_save_path: str = 'aug_vis.png',
            show_aug_seg: bool = False,
            aug_n_rows: int = 3,
            aug_n_cols: int = 5,
            aug_save_dims: Tuple[int] = (3 * 500, 5 * 500),
            device: str = 'cuda',
            num_workers: int = 2,
            images_per_batch: int = 2,
            base_lr: float = 0.003,
            decrease_lr_by_ratio: float = 0.1,
            lr_steps: tuple = (30000,),
            detectron2_dir_path: str = None,
            val_on: bool = False,
            instance_test: str = "test_instance1",
            val_eval_period: int = 100,
            vis_period: int = 0,
            train_type: str = None,
            bg_dirs: str=None,
            eval_tasks: str = None,
    ):
        """
        D2Trainer
        =========

        Parameters:
        ------
        output_dir_path: str 
        class_names: List[str] = None, num_classes: int = None,
        keypoint_names: List[str] = None, num_keypoints: int = None,
        model: str = "mask_rcnn_R_50_FPN_1x",
        confidence_threshold: float = 0.5,
        min_size_train: int = None,
        max_size_train: int = None,
        key_seg_together: bool = False,
        detectron2_dir_path: str = "/home/jitesh/detectron/detectron2"
        """
        self.key_seg_together = key_seg_together
        self.coco_ann_path = coco_ann_path
        self.img_path = img_path
        self.val_coco_ann_path = val_coco_ann_path
        self.val_img_path = val_img_path
        self.output_dir_path = output_dir_path
        self.instance_train = instance_train
        self.resume = resume
        self.device = device
        self.num_workers = num_workers
        self.images_per_batch = images_per_batch
        self.batch_size_per_image = batch_size_per_image
        self.checkpoint_period = checkpoint_period
        self.score_thresh = score_thresh
        self.base_lr = base_lr
        self.decrease_lr_by_ratio = decrease_lr_by_ratio
        self.lr_steps = lr_steps
        self.max_iter = max_iter
        self.val_on = val_on
        self.instance_test = instance_test
        self.val_eval_period = val_eval_period
        self.vis_period = vis_period
        """ Load annotations json """
        with open(self.coco_ann_path) as json_file:
            self.coco_ann_data = json.load(json_file)
            self.categories = self.coco_ann_data["categories"]

        if class_names is None:
            # self.class_names = ['']
            self.class_names = [category["name"]
                                for category in self.categories]
        else:
            self.class_names = class_names
        if num_classes is None:
            self.num_classes = len(self.class_names)
        else:
            printj.red(f'num_classes: {num_classes}')
            printj.red(f'len(self.class_names): {len(self.class_names)}')
            assert num_classes == len(self.class_names)
            self.num_classes = num_classes
        if keypoint_names is None:
            self.keypoint_names = ['']
        else:
            self.keypoint_names = keypoint_names
        if num_keypoints is None:
            if keypoint_names == ['']:
                self.num_keypoints = 0
            else:
                self.num_keypoints = len(self.keypoint_names)
        else:
            assert num_keypoints == len(self.keypoint_names)
            self.num_keypoints = num_keypoints

        self.model = model
        if "COCO-Detection" in self.model:
            self.model = self.model
            train_type = 'bbox'
        elif "COCO-Keypoints" in self.model:
            self.model = self.model
            train_type = 'kpt'
        elif "COCO-InstanceSegmentation" in self.model:
            self.model = self.model
            train_type = 'seg'
        elif "COCO-PanopticSegmentation" in self.model:
            self.model = self.model
            train_type = 'seg'
        elif "LVIS-InstanceSegmentation" in self.model:
            self.model = self.model
            train_type = 'seg'
        elif "Misc" in model:
            self.model = model
            train_type = 'seg'
        elif "rpn" in model or "fast" in model:
            self.model = "COCO-Detection/" + model
            train_type = 'bbox'
        elif "keypoint" in model:
            self.model = "COCO-Keypoints/" + model
            train_type = 'kpt'
        elif "mask" in model:
            self.model = "COCO-InstanceSegmentation/" + model
            train_type = 'seg'
        elif train_type:
            self.model = model
            train_type = train_type
        else:
            printj.red.bold_on_black(f'{model} is not in the dictionary.\
                Choose the correct model.')
            raise Exception

        if ".yaml" in self.model:
            self.model = self.model
        else:
            self.model = self.model + ".yaml"

        if detectron2_dir_path:
            model_conf_path = f"{detectron2_dir_path}/configs/{self.model}"
        else:
            model_conf_path = model_zoo.get_config_file(self.model)
        # printj.yellow(f'{model_conf_path=}')
        if not file_exists(model_conf_path):
            printj.red(f"Invalid model: {model}\nOr")
            printj.red(f"File not found: {model_conf_path}")
            raise Exception

        """ register """
        register_coco_instances(
            name=self.instance_train,
            metadata={},
            json_file=self.coco_ann_path,
            image_root=self.img_path
        )
        MetadataCatalog.get(
            self.instance_train).thing_classes = self.class_names
        # sys.exit(self.class_names)
        if val_on:
            register_coco_instances(
                name=self.instance_test,
                metadata={},
                json_file=self.val_coco_ann_path,
                image_root=self.val_img_path
            )
            MetadataCatalog.get(
                self.instance_test).thing_classes = self.class_names
        """ cfg """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_conf_path)
        self.cfg.DATASETS.TRAIN = tuple([self.instance_train])
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = self.num_keypoints
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = self.images_per_batch
        self.cfg.SOLVER.BASE_LR = self.base_lr
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.OUTPUT_DIR = self.output_dir_path
        if self.lr_steps:
            self.cfg.SOLVER.GAMMA = self.decrease_lr_by_ratio
            self.cfg.SOLVER.STEPS = self.lr_steps
        if self.max_iter:
            self.cfg.SOLVER.MAX_ITER = self.max_iter
        if self.batch_size_per_image:
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_per_image
        if self.checkpoint_period:
            self.cfg.SOLVER.CHECKPOINT_PERIOD = self.checkpoint_period
        if self.vis_period:
            self.cfg.VIS_PERIOD = self.vis_period
        if score_thresh:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        if self.val_on:
            self.cfg.DATASETS.TEST = tuple([self.instance_test])
            self.cfg.TEST.EVAL_PERIOD = self.val_eval_period
        make_dir_if_not_exists(self.cfg.OUTPUT_DIR)
        if not self.resume:
            delete_dir_if_exists(self.cfg.OUTPUT_DIR)
            make_dir_if_not_exists(self.cfg.OUTPUT_DIR)
        if "mask" in self.model.lower() or "segmentation" in self.model.lower():
            self.cfg.MODEL.MASK_ON = True
        else:
            self.cfg.MODEL.MASK_ON = False
        # self.cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT=0.5
        # Train Size Parameters
        if min_size_train is not None:
            self.cfg.INPUT.MIN_SIZE_TRAIN = min_size_train
        if max_size_train is not None:
            self.cfg.INPUT.MAX_SIZE_TRAIN = max_size_train
        # Test Size Parameters
        if min_size_test is not None:
            self.cfg.INPUT.MIN_SIZE_TEST = min_size_test
        elif min_size_train is not None:
            self.cfg.INPUT.MIN_SIZE_TEST = min_size_train
        if max_size_test is not None:
            self.cfg.INPUT.MAX_SIZE_TEST = max_size_test
        elif max_size_train is not None:
            self.cfg.INPUT.MAX_SIZE_TEST = max_size_train

            self.cfg.INPUT.MIN_SIZE_TEST = min_size_train
        """ def train()  """
        self.aug_settings_file_path = aug_settings_file_path
        self.aug_on = aug_on
        self.train_val = train_val
        self.train_type = train_type
        self.aug_vis_save_path = aug_vis_save_path
        self.show_aug_seg = show_aug_seg

        self.aug_n_rows = aug_n_rows
        self.aug_n_cols = aug_n_cols
        self.aug_save_dims = aug_save_dims
        self.bg_dirs = bg_dirs
        self.eval_tasks = eval_tasks

    def train(self):
        if self.val_on:
            self.trainer = ValTrainer(
                cfg=self.cfg,
                aug_settings_file_path=self.aug_settings_file_path,
                aug_on=self.aug_on,
                train_val=self.train_val,
                train_type=self.train_type,
                aug_vis_save_path=self.aug_vis_save_path,
                show_aug_seg=self.show_aug_seg,
                val_on=self.val_on,
                aug_n_rows=self.aug_n_rows, aug_n_cols=self.aug_n_cols,
                aug_save_dims=self.aug_save_dims,
                img_path=self.img_path,
                val_img_path=self.val_img_path,
                bg_dirs=self.bg_dirs,
                eval_tasks=self.eval_tasks,
            )
        else:
            self.trainer = Trainer(
                cfg=self.cfg,
                aug_settings_file_path=self.aug_settings_file_path,
                aug_on=self.aug_on,
                train_val=self.train_val,
                train_type=self.train_type,
                aug_vis_save_path=self.aug_vis_save_path,
                show_aug_seg=self.show_aug_seg,
                val_on=self.val_on)
        self.trainer.resume_or_load(resume=self.resume)
        if self.resume:
            self.trainer.scheduler.milestones = self.cfg.SOLVER.STEPS
        self.trainer.train()


class Trainer(DefaultTrainer):
    def __init__(
            self, cfg,
            aug_settings_file_path=None,
            aug_on: bool = True,
            train_val: bool = False,
            train_type: str = 'seg',
            aug_vis_save_path: str = 'aug_vis.png',
            show_aug_seg: bool = False,
            val_on: bool = False):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        data_loader = self.build_train_loader(
            cfg=cfg,
            aug_settings_file_path=aug_settings_file_path,
            aug_on=aug_on,
            train_val=train_val,
            train_type=train_type,
            aug_vis_save_path=aug_vis_save_path,
            show_aug_seg=show_aug_seg)
        self._data_loader_iter = iter(data_loader)
        self.val_on = val_on

    @classmethod
    def build_train_loader(
            cls, cfg,
            aug_settings_file_path: str = None,
            aug_on: bool = True,
            train_val: bool = False,
            train_type: str = 'seg',
            aug_vis_save_path: str = 'aug_vis.png',
            show_aug_seg: bool = False):
        if aug_on:
            aug_seq = get_augmentation(load_path=aug_settings_file_path)
            aug_loader = AugmentedLoader(cfg=cfg, train_type=train_type, aug=aug_seq,
                                         aug_vis_save_path=aug_vis_save_path, show_aug_seg=show_aug_seg)
            return build_detection_train_loader(cfg, mapper=aug_loader)
        else:
            return build_detection_train_loader(cfg, mapper=None)


class ValTrainer(DefaultTrainer, TrainerBase):
    def __init__(
            self, cfg,
            aug_settings_file_path=None,
            aug_on: bool = True,
            train_val: bool = False,
            train_type: str = 'seg',
            aug_vis_save_path: str = 'aug_vis.png',
            show_aug_seg: bool = False,
            val_on: bool = False,
            aug_n_rows: int = 3, aug_n_cols: int = 5,
            aug_save_dims: Tuple[int] = (3 * 500, 5 * 500),
            img_path: str = '',
            val_img_path: str = '',
            bg_dirs: str = None,
            eval_tasks: str = None,
    ):
        """
        Args:
            cfg (CfgNode):
        """
        self.eval_tasks = eval_tasks
        aug_seq = None
        if aug_on:
            aug_seq = get_augmentation(load_path=aug_settings_file_path)
            # with open(aug_settings_file_path) as f:
            #     student = json.load(f)
            # self.aug_seq = json.dumps(student, indent=4, separators=(',', ': '), sort_keys=True)
            # self.aug_seq=aug_settings_file_path
        # printj.blue.bold_on_white(f"{self.aug_seq=}")
        # printj.blue.bold_on_green("before........................................")
        TrainerBase.__init__(self)
        # super().__init__(cfg)
        # printj.blue.bold_on_green("after........................................")

        logger = logging.getLogger("detectron2")
        # setup_logger is not called for d2
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        # for _ in range(10):
        #     print("5"*11)
        # data_loader = self.build_train_loader(cfg)
        # print(data_loader)
        # for _ in range(10):
        #     print("6"*11)
        # if aug_on:
        data_loader = self.build_train_loader(
            cfg=cfg,
            aug_seq=aug_seq,
            aug_on=aug_on,
            train_val=train_val,
            train_type=train_type,
            aug_vis_save_path=aug_vis_save_path,
            show_aug_seg=show_aug_seg,
            aug_n_rows=aug_n_rows, aug_n_cols=aug_n_cols,
            aug_save_dims=aug_save_dims,
            bg_dirs=bg_dirs)
        printj.cyan(f"{aug_on=}")
        printj.cyan(f"{data_loader=}")
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.tensorboard_dict = {
            "cfg": cfg, "aug_seq": aug_seq,
            "img_path": img_path, "val_img_path": val_img_path,
            "model": model,
            "aug_file_path": aug_settings_file_path,
        }
        self.register_hooks(self.build_hooks())
        # sys.exit()
        # if data_loader is not None:
        #     self._data_loader_iter = iter(data_loader)
        # self.val_on = val_on

    @classmethod
    def build_train_loader(
        cls, cfg,
        aug_seq=None,
        aug_on: bool = True,
        train_val: bool = False,
        train_type: str = 'seg',
        aug_vis_save_path: str = 'aug_vis.png',
        show_aug_seg: bool = False,
        aug_n_rows: int = 3, aug_n_cols: int = 5,
        aug_save_dims: Tuple[int] = (3 * 500, 5 * 500),
        bg_dirs: str = None,
    ):
        if aug_on:
            # for i in range(100):
            #     printj.red("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            # printj.red(f"{aug_settings_file_path=}")
            if aug_seq is None:
                return
            aug_loader = AugmentedLoader(cfg=cfg, train_type=train_type, aug=aug_seq,
                                         aug_vis_save_path=aug_vis_save_path, show_aug_seg=show_aug_seg,
                                         aug_n_rows=aug_n_rows, aug_n_cols=aug_n_cols,
                                         aug_save_dims=aug_save_dims, bg_dirs=bg_dirs,
                                         )
            return build_detection_train_loader(cfg, mapper=aug_loader)
        if not aug_on:
            # for i in range(100):
            #     printj.green("0000000000000000000000000000000000000000000000")
            return build_detection_train_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, eval_tasks=None, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomCOCOEvaluator(dataset_name, eval_tasks, True, output_folder, use_fast_impl=False)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, self.eval_tasks)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        
        # ret.insert(-1, LossEvalHook(
        #     self.cfg.TEST.EVAL_PERIOD,
        #     self.model,
        #     build_detection_test_loader(
        #         self.cfg,
        #         self.cfg.DATASETS.TEST[0],
        #         DatasetMapper(self.cfg, True)
        #     )
        # ))
        return ret
    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1, LossEvalHook(
    #         self.cfg.TEST.EVAL_PERIOD,
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             DatasetMapper(self.cfg, True)
    #         )
    #     ))
    #     return hooks
    @classmethod
    def test(cls, cfg, model, eval_tasks, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, eval_tasks)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter, self.tensorboard_dict)


def default_writers(output_dir: str, max_iter: Optional[int] = None, tensorboard_dict=None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        CustomTensorboardXWriter(
            log_dir=output_dir, tensorboard_dict=tensorboard_dict),
    ]


def _dict_to_str(param_dict, num_tabs: int) -> str:
    """
    Takes a parameter dictionary and converts it to a human-readable string.
    Recurses if there are multiple levels of dict. Used to print out hyperparameters.

    :param param_dict: A Dictionary of key, value parameters.
    :return: A string version of this dictionary.
    """
    if not isinstance(param_dict, dict):
        return str(param_dict)
    else:
        append_newline = "\n" if num_tabs > 0 else ""
        return append_newline + "\n".join(
            [
                "\t"
                + "  " * num_tabs
                + "{}:\t{}".format(x,
                                   _dict_to_str(param_dict[x], num_tabs + 1))
                for x in param_dict
            ]
        )


class CustomTensorboardXWriter(TensorboardXWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, tensorboard_dict=None, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

        cfg = tensorboard_dict["cfg"]
        hyperparameters_dict = {
            "Training_data": tensorboard_dict["img_path"],
            "Test_data": tensorboard_dict["val_img_path"],
            "TEST.EVAL_PERIOD": cfg.TEST.EVAL_PERIOD,
            "MODEL.WEIGHTS": cfg.MODEL.WEIGHTS,
            "MODEL.MASK_ON": cfg.MODEL.MASK_ON,
            "SOLVER.IMS_PER_BATCH": cfg.SOLVER.IMS_PER_BATCH,
            "SOLVER.BASE_LR": cfg.SOLVER.BASE_LR,
            "SOLVER.STEPS": cfg.SOLVER.STEPS,
            "SOLVER.GAMMA": cfg.SOLVER.GAMMA,
            "OUTPUT_DIR": cfg.OUTPUT_DIR,
            "SOLVER.MAX_ITER": cfg.SOLVER.MAX_ITER,
            "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "SOLVER.CHECKPOINT_PERIOD": cfg.SOLVER.CHECKPOINT_PERIOD,
            "VIS_PERIOD": cfg.VIS_PERIOD,
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "INPUT.MIN_SIZE_TRAIN": cfg.INPUT.MIN_SIZE_TRAIN,
            "INPUT.MAX_SIZE_TRAIN": cfg.INPUT.MAX_SIZE_TRAIN,
            "INPUT.MIN_SIZE_TEST": cfg.INPUT.MIN_SIZE_TEST,
            "INPUT.MAX_SIZE_TEST": cfg.INPUT.MAX_SIZE_TEST,
            "aug_file_path": tensorboard_dict["aug_file_path"]
        }
        self._writer.add_text(
            "Hyperparameters", _dict_to_str(hyperparameters_dict, 0))
        self._writer.add_text("Augmentation", f"{tensorboard_dict['aug_seq']}")

        from detectron2.export import TracingAdapter
        model = TracingAdapter(tensorboard_dict["model"], [{"image": torch.zeros(
            (3, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN))}])
        self._writer.add_graph(model, model.flattened_inputs)

        self._writer.add_hparams({
            'lr': cfg.SOLVER.BASE_LR,
            'bsize': cfg.SOLVER.IMS_PER_BATCH,
            'img_max_size': cfg.INPUT.MAX_SIZE_TRAIN,

        },
            {'hparam/accuracy': -1, 'hparam/loss': -1})

        self._last_write = -1


class CustomCOCOEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        super(CustomCOCOEvaluator, self).__init__(dataset_name,
        tasks,
        distributed,
        output_dir,
        use_fast_impl=use_fast_impl,
        kpt_oks_sigmas=kpt_oks_sigmas)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        # recalls ---------------------------------
        recalls = coco_eval.eval["recall"]
        # recall has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            results_per_category.append(("{}".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AR: \n".format(iou_type) + table)

        results.update({"AR-" + name: ar for name, ar in results_per_category})
        return results
# def train(path, coco_ann_path, img_path, output_dir_path, resume=True,
#     model = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"):
#     register_coco_instances(
#         name="box_bolt",
#         metadata={},
#         json_file=coco_ann_path,
#         image_root=img_path
#         # image_root=path
#     )
#     MetadataCatalog.get("box_bolt").thing_classes = ['bolt']
#     # MetadataCatalog.get("box_bolt").keypoint_names = ["kpt-a", "kpt-b", "kpt-c", "kpt-d", "kpt-e",
#     #                                                 "d-left", "d-right"]
#     # MetadataCatalog.get("box_bolt").keypoint_flip_map = [('d-left', 'd-right')]
#     # MetadataCatalog.get("box_bolt").keypoint_connection_rules = [
#     #     ('kpt-a', 'kpt-b', (0, 0, 255)),
#     #     ('kpt-b', 'kpt-c', (0, 0, 255)),
#     #     ('kpt-c', 'kpt-d', (0, 0, 255)),
#     #     ('kpt-d', 'kpt-e', (0, 0, 255)),
#     #     # ('d-left', 'd-right', (0, 0, 255)),
#     # ]
#     # model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"
#     # model = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
#     # model = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(model))
#     # cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'
#     cfg.DATASETS.TRAIN = ("box_bolt",)
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.003
#     cfg.SOLVER.MAX_ITER = 100000
#     cfg.SOLVER.CHECKPOINT_PERIOD = 1000
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)   # faster, and good enough for this toy dataset (default: 512)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
#     # cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 7
#     cfg.INPUT.MIN_SIZE_TRAIN = 1024
#     cfg.INPUT.MAX_SIZE_TRAIN = 1024
#     # cfg.INPUT.MIN_SIZE_TRAIN = 512

#     cfg.OUTPUT_DIR = output_dir_path
#     make_dir_if_not_exists(cfg.OUTPUT_DIR)
#     # resume=True
#     if not resume:
#         delete_dir_if_exists(cfg.OUTPUT_DIR)
#         make_dir_if_not_exists(cfg.OUTPUT_DIR)

#     # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     # trainer = COCO_Keypoint_Trainer(cfg)
#     # trainer = DefaultTrainer(cfg)
#     # from .aug_on import Trainer
#     # trainer = DefaultTrainer(cfg)
#     trainer = Trainer(cfg, aug_settings_file_path = "/home/jitesh/prj/SekisuiProjects/test/gosar/bolt/aug/aug_seq.json")
#     trainer.resume_or_load(resume=resume)
#     trainer.train()


def main():
    # path = "/home/jitesh/3d/data/coco_data/fp2_400_2020_06_05_14_46_57_coco-data"
    # path = "/home/jitesh/3d/data/coco_data/fp2_40_2020_06_05_10_37_48_coco-data"
    # img_path = "/home/jitesh/3d/data/UE_training_results/fp2_40"
    # path = "/home/jitesh/3d/data/coco_data/bolt_real4"
    # path = "/home/jitesh/3d/data/coco_data/hc1_1000_2020_06_30_18_43_56_coco-data"
    path = "/home/jitesh/3d/data/coco_data/bolt/b2_coco-data"
    # path = "/home/jitesh/3d/data/coco_data/hr1_300_coco-data"
    # path = "/home/jitesh/3d/data/coco_data/bolt_real1_training_result1"
    # img_path = "/home/jitesh/3d/data/UE_training_results/fp2_40"
    img_path = f'{path}/img'
    # model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"
    # model = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
    # model = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    model_name = model.split('/')[0].split('-')[1] + '_'\
        + model.split('/')[1].split('_')[2] + '_'\
        + model.split('/')[1].split('_')[3] + '_'\
        + model.split('/')[1].split('_')[5].split('.')[0]
    make_dir_if_not_exists(f'{path}/weights')
    _output_dir_path = f'{path}/weights/{model_name}'
    output_dir_path = f"{_output_dir_path}_1"
    resume = True
    # resume=False
    # if not resume:
    i = 1
    while os.path.exists(f"{_output_dir_path}_{i}"):
        i = i + 1
    if resume:
        output_dir_path = f"{_output_dir_path}_{i-1}"
    else:
        output_dir_path = f"{_output_dir_path}_{i}"
    # coco_ann_path = os.path.join(path, "json/bolt.json")
    # coco_ann_path = os.path.join(path, "json/bbox_resized.json")
    coco_ann_path = os.path.join(path, "json/bolt.json")
    # train(path, coco_ann_path, img_path, output_dir_path, resume=resume, model=model)
    d2 = D2Trainer(coco_ann_path=coco_ann_path,
                   img_path=img_path,
                   output_dir_path=output_dir_path,
                   resume=resume,
                   model=model,
                   aug_on=False,
                   #   num_workers=2,
                   #   images_per_batch=2,
                   #   base_lr=0.002,
                   #   max_iter=10000,
                   #   checkpoint_period=100,
                   #   batch_size_per_image=512,
                   #   num_classes=1,
                   #   max_size_train=1024,
                   #   min_size_train=1024,
                   #   aug_on=True,
                   detectron2_dir_path="/home/jitesh/prj/detectron2")
    d2.train()


if __name__ == '__main__':
    main()
    os.system('spd-say "The training is complete."')
