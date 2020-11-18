# -*- coding: utf-8 -*-
import copy
import json
import os
import random
from datetime import datetime
from functools import partial
from sys import exit as x
from typing import List, Union

import albumentations as A
import cv2
import jaitool.inference.d2_infer
import numpy as np
import printj  # pip install printj
import pyjeasy.file_utils as f
import shapely
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_train_loader, build_detection_test_loader)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor, DefaultTrainer
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
# from logger import logger
from shapely.geometry import Polygon
from tqdm import tqdm

# from annotation_utils.coco.structs import COCO_Dataset
# from annotation_utils.dataset.config.dataset_config /*/import (
    # DatasetConfig, DatasetConfigCollection, DatasetConfigCollectionHandler)
# from common_utils.common_types.bbox import BBox
# from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
# from common_utils.common_types.segmentation import Segmentation
# from common_utils.cv_drawing_utils import (cv_simple_image_viewer, draw_bbox,
                                        #    draw_keypoints, draw_segmentation)

# from imageaug import AugHandler
# from imageaug import Augmenter as aug
# from pasonatron.detectron2.lib.roi_heads import (ROI_HEADS_REGISTRY,
#                                                  CustomROIHeads)
# from pasonatron.detectron2.lib.trainer import COCO_Keypoint_Trainer
# from pasonatron.detectron2.util.augmentation.augmented_loader import mapper

class D2Trainer:
    def __init__(
            self,
            coco_ann_path: str, img_path: str, 
            val_coco_ann_path: str, val_img_path: str, 
            output_dir_path: str, resume: bool=True,
            class_names: List[str] = None, num_classes: int = None,
            keypoint_names: List[str] = None, num_keypoints: int = None,
            model: str = "mask_rcnn_R_50_FPN_1x",
            instance_train: str = "training_instance1",
            size_min: int = 1024,
            size_max: int = 1024,
            max_iter: int = 10000,
            batch_size_per_image: int = 512,
            checkpoint_period: int = 100,
            score_thresh: int = None,
            key_seg_together: bool = False,
            aug_on: bool=True,
            train_val: bool=False,
            aug_settings_file_path: str=None, 
            vis_save_path: str='aug_vis.png', 
            show_aug_seg: bool=False, 
            device: str='cuda',
            num_workers: int=2, 
            images_per_batch: int=2, 
            base_lr: float=0.003, 
            lr_steps: tuple=(30000,),
            detectron2_dir_path: str = None,
            val_on: bool=False,
            instance_test: str = "test_instance1",
            val_eval_period: int = 100,
            train_type: str = None,
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
        size_min: int = None,
        size_max: int = None,
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
        self.lr_steps = lr_steps
        self.max_iter = max_iter
        self.val_on = val_on
        self.instance_test = instance_test
        self.val_eval_period = val_eval_period
        """ Load annotations json """
        with open(self.coco_ann_path) as json_file:
            self.coco_ann_data = json.load(json_file)
            self.categories = self.coco_ann_data["categories"]
        
        if class_names is None:
            # self.class_names = ['']
            self.class_names = [category["name"] for category in self.categories]
        else:
            self.class_names = class_names
        if num_classes is None:
            self.num_classes = len(self.class_names)
        else:
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
        elif "rpn" in model or "fast" in model:
            self.model = "COCO-Detection/" + model
            train_type = 'bbox'
        elif "keypoint" in model:
            self.model = "COCO-Keypoints/" + model
            train_type = 'kpt'
        elif "mask" in model:
            self.model = "COCO-InstanceSegmentation/" + model
            train_type = 'seg'
        elif "Misc" in model:
            self.model = model
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
        MetadataCatalog.get(self.instance_train).thing_classes = self.class_names
        if val_on:
            register_coco_instances(
                name=self.instance_test,
                metadata={},
                json_file=self.val_coco_ann_path,
                image_root=self.val_img_path
            )
            MetadataCatalog.get(self.instance_test).thing_classes = self.class_names
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
        if self.lr_steps:
            self.cfg.SOLVER.STEPS = self.lr_steps
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
        if score_thresh:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.OUTPUT_DIR = self.output_dir_path
        if self.val_on:
            self.cfg.DATASETS.TEST = tuple([self.instance_test])
            self.cfg.TEST.EVAL_PERIOD = self.val_eval_period
        make_dir_if_not_exists(self.cfg.OUTPUT_DIR)
        if not self.resume:
            delete_dir_if_exists(self.cfg.OUTPUT_DIR)
            make_dir_if_not_exists(self.cfg.OUTPUT_DIR)
        if "mask" or "segmentation" in self.model.lower():
            self.cfg.MODEL.MASK_ON = True
        else:
            self.cfg.MODEL.MASK_ON = False
        # self.cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT=0.5
        if size_min is not None:
            self.cfg.INPUT.MIN_SIZE_TEST = size_min
        if size_max is not None:
            self.cfg.INPUT.MAX_SIZE_TEST = size_max
        """ def train()  """
        self.aug_settings_file_path=aug_settings_file_path
        self.aug_on=aug_on
        self.train_val=train_val
        self.train_type=train_type
        self.vis_save_path=vis_save_path
        self.show_aug_seg=show_aug_seg
        
    def train(self):
        if self.val_on:
            self.trainer = ValTrainer(
                cfg=self.cfg, 
                aug_settings_file_path=self.aug_settings_file_path,
                aug_on=self.aug_on, 
                train_val=self.train_val,
                train_type=self.train_type, 
                vis_save_path=self.vis_save_path, 
                show_aug_seg=self.show_aug_seg, 
                val_on=self.val_on)
        else:
            self.trainer = Trainer(
                cfg=self.cfg, 
                aug_settings_file_path=self.aug_settings_file_path,
                aug_on=self.aug_on, 
                train_val=self.train_val,
                train_type=self.train_type, 
                vis_save_path=self.vis_save_path, 
                show_aug_seg=self.show_aug_seg, 
                val_on=self.val_on)
        self.trainer.resume_or_load(resume=self.resume)
        if self.resume:
            self.trainer.scheduler.milestones=self.cfg.SOLVER.STEPS
        self.trainer.train()
    
    


class Trainer(DefaultTrainer):
    def __init__(
        self, cfg, 
        aug_settings_file_path=None,
        aug_on: bool=True,
        train_val: bool=False,
        train_type: str='seg', 
        vis_save_path: str='aug_vis.png', 
        show_aug_seg: bool=False, 
        val_on: bool=False):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        self.data_loader = self.build_train_loader(
            cfg=cfg, 
            aug_settings_file_path=aug_settings_file_path,
            aug_on=aug_on, 
            train_val=train_val,
            train_type=train_type, 
            vis_save_path=vis_save_path, 
            show_aug_seg=show_aug_seg) 
        self._data_loader_iter = iter(self.data_loader)
        self.val_on = val_on
        
    @classmethod
    def build_train_loader(
        cls, cfg, 
        aug_settings_file_path: str=None,
        aug_on: bool=True,
        train_val: bool=False,
        train_type: str='seg', 
        vis_save_path: str='aug_vis.png', 
        show_aug_seg: bool=False):
        if aug_on:
            aug_seq = get_augmentation(load_path=aug_settings_file_path)
            aug_loader = AugmentedLoader(cfg=cfg, train_type=train_type, aug=aug_seq, 
                                        vis_save_path=vis_save_path, show_aug_seg=show_aug_seg)
            return build_detection_train_loader(cfg, mapper=aug_loader)
        else:
            return build_detection_train_loader(cfg, mapper=None)


class ValTrainer(DefaultTrainer):
    def __init__(
        self, cfg, 
        aug_settings_file_path=None,
        aug_on: bool=True,
        train_val: bool=False,
        train_type: str='seg', 
        vis_save_path: str='aug_vis.png', 
        show_aug_seg: bool=False, 
        val_on: bool=False):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        self.data_loader = self.build_train_loader(
            cfg=cfg, 
            aug_settings_file_path=aug_settings_file_path,
            aug_on=aug_on, 
            train_val=train_val,
            train_type=train_type, 
            vis_save_path=vis_save_path, 
            show_aug_seg=show_aug_seg) 
        self._data_loader_iter = iter(self.data_loader)
        self.val_on = val_on
        
    @classmethod
    def build_train_loader(
        cls, cfg, 
        aug_settings_file_path: str=None,
        aug_on: bool=True,
        train_val: bool=False,
        train_type: str='seg', 
        vis_save_path: str='aug_vis.png', 
        show_aug_seg: bool=False):
        if aug_on:
            aug_seq = get_augmentation(load_path=aug_settings_file_path)
            aug_loader = AugmentedLoader(cfg=cfg, train_type=train_type, aug=aug_seq, 
                                        vis_save_path=vis_save_path, show_aug_seg=show_aug_seg)
            return build_detection_train_loader(cfg, mapper=aug_loader)
        else:
            return build_detection_train_loader(cfg, mapper=None)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


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
    resume=True
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
            #   size_max=1024,
            #   size_min=1024,
            #   aug_on=True,
              detectron2_dir_path="/home/jitesh/prj/detectron2")
    d2.train()


if __name__ == '__main__':
    main()
    os.system('spd-say "The training is complete."')
