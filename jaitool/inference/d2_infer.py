# from __future__ import annotations
import json
import timeit
from datetime import datetime
from functools import partial
from shutil import Error
from sys import exit as x
from typing import List, Union

import cv2
import jaitool.inference.d2_infer
import numpy as np
import printj
from pyjeasy.check_utils.check import check_file_exists  # pip install printj
import pyjeasy.file_utils as f
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from jaitool.draw import draw_bbox, draw_keypoints, draw_mask_bool
from jaitool.structures.bbox import BBox
from pyjeasy.check_utils import check_value
from pyjeasy.file_utils import (delete_dir, delete_dir_if_exists, dir_exists,
                                dir_files_list, file_exists, make_dir,
                                make_dir_if_not_exists)
from pyjeasy.image_utils import show_image
from seaborn import color_palette
from tqdm import tqdm


def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
class D2Inferer:
    def __init__(
            self,
            weights_path: str,
            class_names: List[str] = None, num_classes: int = None,
            keypoint_names: List[str] = None, num_keypoints: int = None,
            model: str = "mask_rcnn_R_50_FPN_1x",
            confidence_threshold: float = 0.5,
            size_min: int = None,
            size_max: int = None,
            key_seg_together: bool = False,
            detectron2_dir_path: str = "/home/jitesh/detectron/detectron2"
    ):
        """
        D2Inferer
        =========

        Parameters:
        ------
        weights_path: str 
        class_names: List[str] = None, num_classes: int = None,
        keypoint_names: List[str] = None, num_keypoints: int = None,
        model: str = "mask_rcnn_R_50_FPN_1x",
        confidence_threshold: float = 0.5,
        size_min: int = None,
        size_max: int = None,
        key_seg_together: bool = False,
        detectron2_dir_path: str = "/home/jitesh/detectron/detectron2"
        """
        if class_names is None:
            class_names = ['']
        if keypoint_names is None:
            keypoint_names = ['']
        self.key_seg_together = key_seg_together
        self.weights_path = weights_path
        self.class_names = class_names
        if num_classes is None:
            self.num_classes = len(class_names)
        else:
            assert num_classes == len(class_names)
            self.num_classes = num_classes
        self.keypoint_names = keypoint_names
        if num_keypoints is None:
            self.num_keypoints = len(keypoint_names)
        else:
            assert num_keypoints == len(keypoint_names)
            self.num_keypoints = num_keypoints
        self.confidence_threshold = confidence_threshold
        self.model = model
        if "COCO-Detection" in self.model:
            self.model = self.model
        elif "COCO-Keypoints" in self.model:
            self.model = self.model
        elif "COCO-InstanceSegmentation" in self.model:
            self.model = self.model
        elif "COCO-PanopticSegmentation" in self.model:
            self.model = self.model
        elif "LVIS-InstanceSegmentation" in self.model:
            self.model = self.model
        elif "rpn" in model:
            self.model = "COCO-Detection/" + model
        elif "keypoint" in model:
            self.model = "COCO-Keypoints/" + model
        elif "mask" in model:
            self.model = "COCO-InstanceSegmentation/" + model
        else:
            printj.red.bold_on_black(f'{model} is not in the dictionary.\
                Choose the correct model.')
            raise Exception

        if ".yaml" in self.model:
            self.model = self.model
        else:
            self.model = self.model + ".yaml"

        self.cfg = get_cfg()
        model_conf_path = f"{detectron2_dir_path}/configs/{self.model}"
        if not file_exists(model_conf_path):
            printj.red(f"Invalid model: {model}\nOr")
            printj.red(f"File not found: {model_conf_path}")
            raise Exception
        self.cfg.merge_from_file(model_conf_path)
        self.cfg.MODEL.WEIGHTS = self.weights_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = self.num_keypoints
        if "mask" or "segmentation" in self.model.lower():
            self.cfg.MODEL.MASK_ON = True
        # self.cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT=0.5
        if size_min is not None:
            self.cfg.INPUT.MIN_SIZE_TEST = size_min
        if size_max is not None:
            self.cfg.INPUT.MAX_SIZE_TEST = size_max
        self.predictor = DefaultPredictor(self.cfg)
        self.pred_dataset=[]
        

    def get_outputs(self, img: np.ndarray) -> dict:
        ''''''
        return self.predictor(img)

    def predict(self, img: np.ndarray) -> dict:
        """
        predict_dict = self.predict(img=img)
        
        score_list = predict_dict['score_list']
        
        bbox_list = predict_dict['bbox_list']
        
        pred_class_list = predict_dict['pred_class_list']
        
        pred_masks_list = predict_dict['pred_masks_list']
        
        pred_keypoints_list = predict_dict['pred_keypoints_list']
        
        vis_keypoints_list = predict_dict['vis_keypoints_list']
        
        kpt_confidences_list = predict_dict['kpt_confidences_list']
        
        for score, pred_class, bbox, mask, keypoints, vis_keypoints, kpt_confidences in zip(score_list,
                                                                                            pred_class_list,
                                                                                            bbox_list,
                                                                                            pred_masks_list,
                                                                                            pred_keypoints_list,
                                                                                            vis_keypoints_list,
                                                                                            kpt_confidences_list):
        """
        outputs = self.get_outputs(img)
        result = dict()
        score_list = [float(val)
                      for val in outputs['instances'].scores.cpu().numpy()]
        bbox_list = [BBox.from_list(val_list).to_int() for val_list in
                     outputs['instances'].pred_boxes.tensor.cpu().numpy()]
        pred_class_list = [self.class_names[idx]
                           for idx in outputs['instances'].pred_classes.cpu().numpy()]
        if "mask" or "segmentation" in self.model.lower():
            pred_masks_list = [mask
                               for mask in outputs['instances'].pred_masks.cpu().numpy()]
        else:
            pred_masks_list = [None] * len(score_list)
        if 'keypoint' in self.model.lower():
            pred_keypoints_list = [keypoints
                                   for keypoints in outputs['instances'].pred_keypoints.cpu().numpy()]

            vis_keypoints_list = [[[int(x), int(y)] for x, y, c in keypoints]
                                  for keypoints in pred_keypoints_list]
            kpt_confidences_list = [[c for x, y, c in keypoints]
                                    for keypoints in pred_keypoints_list]
        else:
            pred_keypoints_list = [None] * len(score_list)
            vis_keypoints_list = [None] * len(score_list)
            kpt_confidences_list = [None] * len(score_list)
        result['score_list'] = score_list
        result['bbox_list'] = bbox_list
        result['pred_class_list'] = pred_class_list
        result['pred_masks_list'] = pred_masks_list
        result['pred_keypoints_list'] = pred_keypoints_list
        result['vis_keypoints_list'] = vis_keypoints_list
        result['kpt_confidences_list'] = kpt_confidences_list
        return result

    @staticmethod
    def confirm_folder(path, mode):
        '''Deletes the directory if exist and create new directory.'''
        # make_dir_if_not_exists(path)
        if mode == 'save':
            if dir_exists(path):
                delete_dir(path)
                make_dir(path)
            else:
                make_dir(path)

    def infer_image(self, image_path: str,
                    show_max_score_only: bool = False,
                    show_class_label: bool = True,
                    show_class_label_score_only: bool = False,
                    show_keypoint_label: bool = True,
                    show_bbox: bool = True,
                    show_keypoints: bool = True,
                    show_segmentation: bool = True,
                    transparent_mask: bool = True,
                    transparency_alpha: float = 0.3,
                    ignore_keypoint_idx=None,
                    gt_path: str=None,) -> np.ndarray:
        '''Returns the Inference result of a single image.'''
        # if gt_path:
        #     with open(gt_path) as json_file:
        #         gt_data = json.load(json_file)
        if ignore_keypoint_idx is None:
            ignore_keypoint_idx = []
        img = cv2.imread(image_path)
        output = img.copy()
        predict_dict = self.predict(img=img)
        output = self.draw_gt(image_path.split("/")[-1], output)
        score_list = predict_dict['score_list']
        bbox_list = predict_dict['bbox_list']
        pred_class_list = predict_dict['pred_class_list']
        pred_masks_list = predict_dict['pred_masks_list']
        pred_keypoints_list = predict_dict['pred_keypoints_list']
        vis_keypoints_list = predict_dict['vis_keypoints_list']
        kpt_confidences_list = predict_dict['kpt_confidences_list']
        # printj.cyan(predict_dict['bbox_list'])
        
        
        palette = np.array(color_palette(None, self.num_classes+1))*255
        
        max_score_list = dict()
        max_score_pred_list = dict()
        if show_max_score_only:
            for i, class_name in enumerate(self.class_names):
                max_score_list[class_name] = -1
        for score, pred_class, bbox, mask, keypoints, vis_keypoints, kpt_confidences in zip(score_list,
                                                                                            pred_class_list,
                                                                                            bbox_list,
                                                                                            pred_masks_list,
                                                                                            pred_keypoints_list,
                                                                                            vis_keypoints_list,
                                                                                            kpt_confidences_list):
            if show_max_score_only:
                for i, class_name in enumerate(self.class_names):
                    if class_name == pred_class:
                        if max_score_list[class_name] < score:
                            max_score_list[class_name] = score
                            max_score_pred_list[class_name] = {
                                "score": score,
                                "pred_class": pred_class,
                                "bbox": bbox,
                                "mask": mask,
                                "keypoints": keypoints,
                                "vis_keypoints": vis_keypoints,
                                "kpt_confidences": kpt_confidences,
                                }
            else:
                if mask is not None and show_segmentation:
                    output = draw_mask_bool(img=output, mask_bool=mask, transparent=transparent_mask,
                                            alpha=transparency_alpha)
                if show_class_label_score_only:
                    output = draw_bbox(img=output, bbox=bbox,
                                    show_bbox=show_bbox, show_label=show_class_label, text=f'{round(score, 2)}')
                else:
                    output = draw_bbox(img=output, bbox=bbox,
                                    show_bbox=show_bbox, show_label=show_class_label, text=f'{pred_class} {round(score, 2)}')
                
                if keypoints is not None and show_keypoints:
                    output = draw_keypoints(img=output, keypoints=keypoints, show_keypoints=show_keypoints,
                                            keypoint_labels=self.keypoint_names, show_keypoints_labels=show_keypoint_label,
                                            ignore_kpt_idx=ignore_keypoint_idx)
                xmin, ymin, xmax, ymax = bbox.to_int().to_list()
                
                cat_id = self.class_names.index(pred_class)
                if self.gt_path:
                    for category in self.gt_data["categories"]:
                        if category["name"] == pred_class:
                            cat_id = category["id"]
                    self.pred_dataset.append(
                        {
                            "image_id": self.image_id, 
                            "category_id": cat_id, 
                            "bbox": BBox(xmin, ymin, xmax, ymax).to_list(output_format = 'pminsize'), 
                            "score": score,
                        }
                    )
                else:
                    self.pred_dataset.append(
                        {
                            "image_id": next(self.counter), 
                            "category_id": cat_id, 
                            "bbox": BBox(xmin, ymin, xmax, ymax).to_list(output_format = 'pminsize'), 
                            "score": score,
                        }
                    )
        
        if show_max_score_only:
            for i, class_name in enumerate(self.class_names):
                if max_score_list[class_name] > 0:
                    max_pred = max_score_pred_list[class_name]
                    if max_pred["mask"] is not None and show_segmentation:
                        output = draw_mask_bool(img=output, mask_bool=max_pred["mask"], color=palette[i+1], transparent=transparent_mask,
                                                alpha=transparency_alpha)
                    output = draw_bbox(img=output, bbox=max_pred["bbox"],
                                    show_bbox=show_bbox, show_label=show_class_label, text=f'{max_pred["pred_class"]} {round(max_pred["score"], 2)}')
                    if keypoints is not None and show_keypoints:
                        output = draw_keypoints(img=output, keypoints=max_pred["keypoints"], show_keypoints=show_keypoints,
                                                keypoint_labels=self.keypoint_names, show_keypoints_labels=show_keypoint_label,
                                                ignore_kpt_idx=ignore_keypoint_idx)
                        
        return output

    def infer(self, input_type: Union[str, int], 
              output_type: Union[str, int],
              input_path: Union[str, List[str]], 
              output_path: Union[str, List[str]],
              show_max_score_only: bool = False,
              show_class_label: bool = True,
              show_class_label_score_only: bool = False,
              show_keypoint_label: bool = True,
              show_bbox: bool = True,
              show_keypoints: bool = True,
              show_segmentation: bool = True,
              transparent_mask: bool = True,
                    transparency_alpha: float = 0.3,
                    ignore_keypoint_idx=None,
              gt_path: Union[str, List[str]] = None,
              result_json_path: str= None):
        """
        
        Valid options for,
        === 
        input_type:
        --- 
        ["image", "image_list", "image_directory", "image_directories_list", "video",
        "video_list", "video_directory" ]
        
        output_type: 
        ---
        ["show_image", "write_image", "write_video" ]
        
        Returns
        ---
        The inference result of all data formats.
        """
        self.counter = infinite_sequence()
        check_value(input_type,
                    check_from=["image", "image_list", "image_directory", "image_directories_list", "video",
                                "video_list", "video_directory"])
        check_value(value=output_type, check_from=[
                    "show_image", "write_image", "write_video"])
        self.gt_path = gt_path
        if self.gt_path:
            check_file_exists(gt_path)
            with open(gt_path) as json_file:
                self.gt_data = json.load(json_file)
        if result_json_path is None:
            result_json_path = f'{output_path}/result.json'
            
        predict_image = partial(self.infer_image,
                                show_max_score_only=show_max_score_only, 
                                show_class_label=show_class_label,
                                show_class_label_score_only=show_class_label_score_only,
                                show_keypoint_label=show_keypoint_label, 
                                show_bbox=show_bbox, show_keypoints=show_keypoints, show_segmentation=show_segmentation,
                                transparent_mask=transparent_mask, transparency_alpha=transparency_alpha,
                                ignore_keypoint_idx=ignore_keypoint_idx,
                                gt_path=gt_path)
        if input_type == "image":
            if file_exists(input_path):
                output = predict_image(input_path)
            else:
                raise Error
            if output_type == "show_image":
                show_image(output)
            elif output_type == "write_image":
                cv2.imwrite(output_path, output)
            elif output_type == "write_video":
                raise NotImplementedError
            else:
                raise Exception
        elif input_type == "image_list":
            for image_path in tqdm(input_path, colour='#66cc66'):
                output = predict_image(image_path)
                if output_type == "show_image":
                    if show_image(output):
                        break
                elif output_type == "write_image":
                    make_dir_if_not_exists(output_path)
                    filename = f.get_filename(image_path)
                    cv2.imwrite(f'{output_path}/{filename}', output)
                elif output_type == "write_video":
                    raise NotImplementedError
                else:
                    raise Exception
        elif input_type == "image_directory":
            image_path_list = f.dir_contents_path_list_with_extension(
                dirpath=input_path,
                extension=[".png", ".jpg", ".jpeg"])
            for image_path in tqdm(image_path_list, colour='#66cc66'):
                output = predict_image(image_path)
                # output = self.draw_gt(gt_path, gt_data, image_path, output)
                if output_type == "show_image":
                    if show_image(output):
                        break
                elif output_type == "write_image":
                    make_dir_if_not_exists(output_path)
                    filename = f.get_filename(image_path)
                    cv2.imwrite(f'{output_path}/{filename}', output)
                elif output_type == "write_video":
                    raise NotImplementedError
                else:
                    raise Exception
        elif input_type == "image_directories_list":
            for image_directory in tqdm(input_path):
                image_path_list = f.dir_contents_path_list_with_extension(
                    dirpath=input_path,
                    extension=[".png", ".jpg", ".jpeg"])
                for image_path in tqdm(image_path_list, colour='#66cc66'):
                    output = predict_image(image_path)
                    if output_type == "show_image":
                        if show_image(output):
                            break
                    elif output_type == "write_image":
                        filename = f.get_filename(image_path)
                        directory_name = f.get_directory_name(image_path)
                        if f.dir_exists(f'{output_path}/{directory_name}'):
                            f.delete_all_files_in_dir(
                                f'{output_path}/{directory_name}')
                        else:
                            f.make_dir(f'{output_path}/{directory_name}')
                        cv2.imwrite(
                            f'{output_path}/{directory_name}/{filename}', output)
                    elif output_type == "write_video":
                        raise NotImplementedError
                    else:
                        raise Exception
        elif input_type == "video":
            raise NotImplementedError
        elif input_type == "video_list":
            raise NotImplementedError
        elif input_type == "video_directory":
            raise NotImplementedError
        else:
            raise Exception
        with open(result_json_path, 'w') as outfile:
            json.dump(self.pred_dataset, outfile, indent=4)

    def draw_gt(self, image_name, output):
        if self.gt_path:
            for image in self.gt_data["images"]:
                if image["file_name"] == image_name:
                    self.image_id = image["id"]
            for ann in self.gt_data["annotations"]:
                if ann["image_id"] == self.image_id:
                    gt_bbox = BBox.from_list(bbox=ann["bbox"], input_format='pminsize')
                    output = draw_bbox(img=output, bbox=gt_bbox,
                        show_bbox=True, show_label=False, color=[0, 0, 255], thickness=2)
                # x()
        return output

if __name__ == "__main__":
    
    
    # palette = color_palette(None, 3)
    # # palette = [int(c*255) for color in palette for c in color]
    # # palette = [int(c*255) for color in palette for c in color]
    # printj.red(np.array(color_palette(None, 3))*255)
    # printj.red(palette)
    # x()
    inferer_seg = D2Inferer(
        weights_path='/home/jitesh/3d/data/coco_data/ty1w_coco-data/weights/InstanceSegmentation_R_50_1x_aug_val_1/model_0019999.pth',
        confidence_threshold=0.01,
        class_names=['tropicana'],
        model='mask_rcnn_R_50_FPN_1x',
        size_min=1000, size_max=1000,
    )
    now = datetime.now()
    dt_string3 = now.strftime("%m_%d_%H")
    # infer_dump_dir = f'/home/jitesh/3d/blender/Learning/tropicana/infer_5k_{dt_string3}'
    infer_dump_dir = f'/home/jitesh/3d/blender/Learning/tropicana_yellow/infer_5k_{dt_string3}'
    delete_dir_if_exists(infer_dump_dir)
    make_dir_if_not_exists(infer_dump_dir)
    
    dir_path = "/home/jitesh/Downloads/Photos (3)"
    # dir_path = "/home/jitesh/3d/blender/Learning/tropicana_yellow/img"
    # dir_path = "/home/jitesh/3d/data/coco_data/tropi1_coco-data/img"
    inferer_seg.infer(input_type="image_directory", 
              output_type="show_image",
              input_path=dir_path, 
              output_path=infer_dump_dir,
              show_class_label=True,
              show_max_score_only= True,
              transparency_alpha=.9,
              )
