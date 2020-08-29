import json
import os
from datetime import datetime
from sys import exit as x
from typing import List

import cv2
import numpy as np
import pandas as pd
import printj  # pip install printj
from jaitool.inference import D2Inferer as inferer
from jaitool.inference.models.hook import draw_info_box, draw_inference_on_hook2

from pyjeasy.file_utils import (dir_exists, file_exists, delete_dir, 
                                make_dir, make_dir_if_not_exists)
from pyjeasy.math_utils import dist
from annotation_utils.coco.structs import COCO_Annotation, COCO_Dataset
from common_utils import path_utils
from common_utils.check_utils import check_value
from common_utils.common_types import BBox
from common_utils.common_types.bbox import BBox
from common_utils.common_types.bbox import ConstantAR_BBox as BBox
from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
from common_utils.cv_drawing_utils import (SimpleVideoViewer,
                                           cv_simple_image_viewer, 
                                           draw_bbox,
                                           draw_bool_mask, 
                                           draw_keypoints,
                                           draw_skeleton)
# from common_utils.file_utils import (delete_dir, dir_exists, file_exists,
#                                      make_dir, make_dir_if_not_exists)
from common_utils.path_utils import (get_all_files_in_extension_list,
                                     get_all_files_of_extension, get_filename,
                                     get_rootname_from_path, get_script_dir,
                                     rel_to_abs_path)
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from typing import List
# from logger import logger
from tqdm import tqdm


def infer(path: str, weights_path: str, thresh: int = 0.5, key: str = 'R', infer_dump_dir: str = '', model: str = 'mask_rcnn_R_50_FPN_1x', size: int = 1024,
          class_names: List[str]=['hook'],
          gt_path: str = '/home/jitesh/3d/data/coco_data/hook_test/json/cropped_hook.json'):
    # class_names=['hook', 'pole']
    # class_names=['hook']
    conf_thresh = 0.001
    show_bbox_border = True
    gt_dataset = COCO_Dataset.load_from_path(json_path=gt_path)
    inferer_seg = inferer(
        weights_path=weights_path,
        confidence_threshold=0.1,
        # num_classes=1,
        # num_classes=2,
        class_names=class_names,
        # class_names=['hook'],
        model='keypoint_rcnn_R_50_FPN_1x',
        # model='faster_rcnn_X_101_32x8d_FPN_3x',
        # model='faster_rcnn_R_101_FPN_3x',
        # model=model,

    )
    inferer_seg.cfg.INPUT.MIN_SIZE_TEST = size
    inferer_seg.cfg.INPUT.MAX_SIZE_TEST = size
    inferer_seg.cfg.MODEL.MASK_ON = True

    weights_path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data7/weights/Keypoints_R_50_1x_aug_cm_seg_val_1/model_0009999.pth'
    weights_path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data7_0.1/weights/Keypoints_R_50_1x_aug_cm_seg_val_3/model_0009999.pth'
    weights_path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data7_0.1/weights/Keypoints_R_50_1x_aug_cm_seg_val_1/model_0007999.pth'
    weights_path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_1/model_0009999.pth'
    weights_path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_2/model_0004999.pth'
    # inferer_key = jDetectron2KeypointInferer(
    #     weights_path=weights_path,
    #     # ref_coco_ann_path=f'/home/jitesh/3d/data/coco_data/hook_real1/json/hook.json',
    #     # categories_path=f'/home/jitesh/3d/data/categories/hook_infer.json',
    #     # categories_path=f'/home/jitesh/3d/data/categories/hook_7ckpt.json',
    #     categories_path=f'/home/jitesh/3d/data/categories/hook_7ckpt_pole.json',
    #     target_category='hook',
    #     model_name='keypoint_rcnn_R_50_FPN_1x',
    #     bbox_threshold=bbox_thresh,
    #     kpt_threshold=kpt_thresh,
    #     key_box='hook',
    # )
    # k_size = 1024
    # inferer_key.cfg.INPUT.MIN_SIZE_TEST = k_size
    # inferer_key.cfg.INPUT.MAX_SIZE_TEST = k_size

    possible_modes = ['save', 'preview']
    mode = 'save'
    check_value(mode, valid_value_list=possible_modes)
    # make_dir_if_not_exists(infer_dump_dir)
    img_extensions = ['jpg', 'JPG', 'png', 'PNG']
    img_pathlist = get_all_files_in_extension_list(
        dir_path=f'{path}', extension_list=img_extensions)
    img_pathlist.sort()

    confirm_folder(infer_dump_dir, mode)
    # confirm_folder(f'{infer_dump_dir}/good_seg', mode)
    # confirm_folder(f'{infer_dump_dir}/good_cropped', mode)
    # confirm_folder(f'{infer_dump_dir}/good', mode)
    # confirm_folder(f'{infer_dump_dir}/G(>4D) P(>4D)', mode)
    # confirm_folder(f'{infer_dump_dir}/G(>4D) P(<4D)', mode)
    # confirm_folder(f'{infer_dump_dir}/G(<4D) P(>4D)', mode)
    # confirm_folder(f'{infer_dump_dir}/G(<4D) P(<4D)', mode)
    # confirm_folder(f'{infer_dump_dir}/bad', mode)
    confirm_folder(f'{infer_dump_dir}/infer_key_seg', mode)

    count = 0
    start = datetime.now()
    df = pd.DataFrame(data=[], columns=['gt_d', 'pred_d',
                                        'gt_ab', 'pred_ab',
                                        'gt_ratio', 'pred_ratio',
                                        'gt_ratio>4', 'pred_ratio>4',
                                        'correct_above4d_ratio', 'incorrect_above4d_ratio',
                                        'correct_below4d_ratio', 'incorrect_below4d_ratio',
                                        ])
    #  'image_path'])
    for i, img_path in enumerate(tqdm(img_pathlist, desc='Writing images',)):
        img_filename = get_filename(img_path)
        # if not '201005_70_縮小革命PB020261.jpg' in img_path:
        #     continue
        # if i > 19:
        #     continue
        printj.purple(img_path)
        img = cv2.imread(img_path)
        result = img
        # print(f'shape {img.shape}')
        # cv2.imshow('i', img)
        # cv2.waitKey(100000)
        # continue
        score_list, pred_class_list, bbox_list, pred_masks_list, pred_keypoints_list, vis_keypoints_list, kpt_confidences_list = inferer_seg.predict(
            img=img)
        # printj.blue(pred_masks_list)
        max_hook_score = -1
        max_pole_score = -1
        diameter = -1
        len_ab = -1
        found_hook = False
        found_pole = False
        for score, pred_class, bbox, mask, keypoints, vis_keypoints, kpt_confidences in zip(score_list, pred_class_list, bbox_list, pred_masks_list, pred_keypoints_list, vis_keypoints_list, kpt_confidences_list):

            if pred_class == 'pole':
                found_pole = True
                if max_pole_score < score:
                # if True:
                    max_pole_score = score
                    diameter = compute_diameter(mask)
                    # result = draw_bool_mask(img=result, mask=mask, color=[
                    #                     0, 255, 255],
                    #                     transparent=True
                    #                     )
                    pole_bbox_text = f'pole {str(round(score, 2))}'
                    pole_bbox = bbox
                    pole_mask = mask
                    # result = draw_bbox(img=result, bbox=bbox,
                    #                    text=pole_bbox_text, label_only=not show_bbox_border, label_orientation='bottom')
                    printj.blue(f'diameter={diameter}')
            if pred_class == 'hook':
                # printj.green.bold_on_yellow(score)
                found_hook = True
                if max_hook_score < score:
                # if True:
                    max_hook_score = score
                    hook_bbox = BBox.buffer(bbox)
                    hook_score = round(score, 2)
                    hook_mask = mask
                    hook_keypoints = keypoints
                    hook_vis_keypoints = vis_keypoints
                    hook_kpt_confidences = kpt_confidences
                    # xmin, ymin, xmax, ymax = bbox.to_int().to_list()
                    # _xmin, _ymin, _xmax, _ymax = _bbox.to_int().to_list()
                    # width = _xmax-_xmin
                    # height = _ymax-_ymin
                    # scale = 0.2
                    # xmin = max(int(_xmin - width*scale), 0)
                    # xmax = min(int(_xmax + width*scale), img.shape[1])
                    # ymin = max(int(_ymin - height*scale), 0)
                    # ymax = min(int(_ymax + height*scale), img.shape[0])

                    # printj.red(score)
                    # printj.red(bbox)
                    # return
                    # img = draw_bbox(img=img, bbox=_bbox, color=[
                    #                 0, 255, 255], thickness=2, text=f"{pred_class} {round(score, 3)}",
                    #                 label_orientation='top')
                    # img = draw_bbox(img=img, bbox=_bbox, color=[
                    #                 0, 255, 255], thickness=2, text=f"{pred_class} {round(score, 3)}",
                    #                 label_orientation='bottom')
                    # result = draw_bool_mask(img=result, mask=mask, color=[
                    #     255, 255, 0],
                    #     transparent=True
                    # )
                    # result = result
                    # bbox_text = str(round(score, 4))
                    # result = draw_bbox(img=result, bbox=bbox,
                    #                    text=bbox_text, label_only=not show_bbox_border)
                    bbox_label_mode = 'euler'
                    # result = draw_keypoints(
                    #     img=result, keypoints=vis_keypoints, radius=2, color=[0, 0, 255],
                    #     # keypoint_labels=kpt_labels, show_keypoints_labels=True, label_thickness=1,
                    #     # ignore_kpt_idx=conf_idx_list
                    #     )
                    kpt_labels = ["kpt-a", "kpt-b", "kpt-cb",
                                  "kpt-c", "kpt-cd", "kpt-d", "kpt-e"]
                    kpt_skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
                    conf_idx_list = np.argwhere(
                        np.array(kpt_confidences) > conf_thresh).reshape(-1)
                    not_conf_idx_list = np.argwhere(
                        np.array(kpt_confidences) <= conf_thresh).reshape(-1).astype(int)
                    conf_keypoints, conf_kpt_labels = np.array(
                        vis_keypoints)[conf_idx_list], np.array(kpt_labels)[conf_idx_list]
                    not_conf_keypoints, not_conf_kpt_labels = np.array(
                        vis_keypoints)[not_conf_idx_list], np.array(kpt_labels)[not_conf_idx_list]
                    cleaned_keypoints = np.array(
                        vis_keypoints.copy()).astype(np.float32)
                    # result = draw_bool_mask(img=result, mask=mask, color=[
                    #     255, 255, 0],
                    #     transparent=True
                    # )
                    # result, len_ab = draw_inference_on_hook2(img=result, cleaned_keypoints=cleaned_keypoints, kpt_labels=kpt_labels, kpt_skeleton=kpt_skeleton,
                    #                                         score=score, bbox=_bbox, vis_keypoints=vis_keypoints, kpt_confidences=kpt_confidences, conf_idx_list=conf_idx_list, not_conf_idx_list=not_conf_idx_list,
                    #                                         conf_keypoints=conf_keypoints, conf_kpt_labels=conf_kpt_labels, not_conf_keypoints=not_conf_keypoints, not_conf_kpt_labels=not_conf_kpt_labels,
                    #                                         conf_thresh=conf_thresh, show_bbox_border=show_bbox_border, bbox_label_mode=bbox_label_mode, index_offset=0, diameter=diameter)
                    # result=result
                    # printj.green(_bbox)
                    # printj.green(_bbox.to_int())
                    # printj.green(_bbox.to_int().to_list())
        printj.green.on_white(max_hook_score)
        if found_pole:
            result = draw_bool_mask(img=result, mask=pole_mask, color=[
                0, 255, 255],
                transparent=True
            )
            result = draw_bbox(img=result, bbox=pole_bbox,
                               text=pole_bbox_text, label_only=not show_bbox_border, label_orientation='top')
            result = draw_bbox(img=result, bbox=pole_bbox,
                               text=pole_bbox_text, label_only=not show_bbox_border, label_orientation='bottom')
        if found_hook:
            result = draw_bool_mask(img=result, mask=hook_mask, color=[
                255, 255, 0],
                transparent=True
            )
            result, len_ab = draw_inference_on_hook2(img=result, cleaned_keypoints=cleaned_keypoints, kpt_labels=kpt_labels, kpt_skeleton=kpt_skeleton,
                                                     score=hook_score, bbox=hook_bbox, vis_keypoints=hook_vis_keypoints, kpt_confidences=hook_kpt_confidences, conf_idx_list=conf_idx_list, not_conf_idx_list=not_conf_idx_list,
                                                     conf_keypoints=conf_keypoints, conf_kpt_labels=conf_kpt_labels, not_conf_keypoints=not_conf_keypoints, not_conf_kpt_labels=not_conf_kpt_labels,
                                                     conf_thresh=conf_thresh, show_bbox_border=show_bbox_border, bbox_label_mode=bbox_label_mode, index_offset=0, diameter=diameter)
        printj.purple(len_ab)
        if len_ab == 0:
            printj.green(keypoints)
        result = draw_info_box(result, len_ab, diameter)
        #                 img: np.ndarray, cleaned_keypoints, kpt_labels: List[str], kpt_skeleton: List[list],
        # score: float, bbox: BBox, vis_keypoints: list, kpt_confidences: list, conf_idx_list: list, not_conf_idx_list: list,
        # conf_keypoints, conf_kpt_labels, not_conf_keypoints, not_conf_kpt_labels,
        # conf_thresh: float = 0.3, show_bbox_border: bool = False, bbox_label_mode: str = 'euler', index_offset: int = 0, diameter=1

        # cv2.imshow('i', result)
        # # cv2.imwrite('i', result)
        # cv2.waitKey(10000)
        # quit_flag = cv_simple_image_viewer(img=result, preview_width=1000)
        # if quit_flag:
        #     break

        # cv2.imwrite(f"{infer_dump_dir}/good_seg/{img_filename}", result)
        cv2.imwrite(f"{infer_dump_dir}/infer_key_seg/{img_filename}", result)
        # cv2.imwrite(f"{infer_dump_dir}/good_seg/{img_filename}", result)
        # # img3, score_list, bbox_list, len_ab = inferer_key.infer_image(img=img2, draw_hm_collage=False, show_bbox_border=True, diameter=diameter)
        # if diameter<=0:
        #     length_ratio = np.inf
        # else:
        #     length_ratio = len_ab/diameter
        # printj.purple(length_ratio)
        # img4=img0
        # img4[ymin:ymax, xmin:xmax]=img3
        # font                   = cv2.FONT_HERSHEY_SIMPLEX
        # TopLeftCornerOfText = (10,50)
        # fontScale              = 1
        # fontColor              = (255,255,255)
        # lineType               = 2
        # cv2.rectangle(img4, (5,10 ), (280,180), (0,0,0), -1)
        # cv2.rectangle(img4, (5,10 ), (280,180), (200,200,0), 2)
        # cv2.putText(img4, f'Len-ab: {round(len_ab,2)}', (10,50), font, fontScale, fontColor, lineType)
        # cv2.putText(img4, f'Diameter: {round(diameter,2)}', (10,100), font, fontScale, fontColor, lineType)
        # cv2.putText(img4, str(round(length_ratio,2))+' D', (10,150), font, fontScale, fontColor, lineType)
        # printj.purple(f'img0.shape = {img0.shape}')
        # printj.purple(f'img.shape = {img.shape}')
        # printj.purple(f'img2.shape = {img2.shape}')
        # printj.purple(f'img3.shape = {img3.shape}')
        # printj.purple(f'img4.shape = {img4.shape}')
        # printj.purple(img.shape)
        # printj.purple(img2.shape)
        # printj.purple(img3.shape)
        # printj.purple(img4.shape)
        # quit_flag = cv_simple_image_viewer(img=img4, preview_width=1000)
        # if quit_flag:
        #     break
        # continue
        # if len(score_list) == 0:
    #     if all(score < thresh for score in score_list):
    #         count = count +1
    #         # printj.purple(img_path)
    #         printj.purple(score_list)
    #         printj.yellow.bold_on_black(f'Good count: {i+1-count}, Bad count: {count}, Total: {i+1}')
    #         dump_path = f"{infer_dump_dir}/bad/{img_filename}"
    #         # cv2.imwrite(dump_path, img)
    #     else:
    #     #     # printj.purple(score_list)
    #     #     pass
    #         dump_path = f"{infer_dump_dir}/good/{img_filename}"
    #         cv2.imwrite(f"{infer_dump_dir}/good_cropped/{img_filename}", img3)
    #         cv2.imwrite(f"{infer_dump_dir}/good_seg/{img_filename}", result)
    #     # dump_path = f"{infer_dump_dir}/{img_filename}"
    #     cv2.imwrite(dump_path, img4)
    #     printj.yellow(f"({i+1}/{len(img_pathlist)}): Wrote {dump_path}")
    #     for image in gt_dataset.images:
    #         if image.file_name == img_filename:
    #             image_id = image.id
    #     for ann in gt_dataset.annotations:
    #         if ann.image_id == image_id:
    #             keys = Keypoint2D_List.to_point_list(ann.keypoints)
    #     gt_diameter = keys[7].distance(keys[8])
    #     gt_len_ab = keys[0].distance(keys[1])
    #     # gt_ratio = round(gt_diameter/gt_len_ab, 2)
    #     if gt_diameter<=0:
    #         gt_ratio = np.inf
    #     else:
    #         gt_ratio = round(gt_len_ab/gt_diameter, 2)
    #     # correct_ratio = int((length_ratio>4)==(gt_ratio>4))
    #     # incorrect_ratio = int((length_ratio>4)!=(gt_ratio>4))
    #     correct_above4d_ratio = int((length_ratio>4)==(gt_ratio>4)==True)
    #     incorrect_below4d_ratio = int((length_ratio>4)==(gt_ratio<4)==True)
    #     correct_below4d_ratio = int((length_ratio<4)==(gt_ratio<4)==True)
    #     incorrect_above4d_ratio = int((length_ratio<4)==(gt_ratio>4)==True)

    #     if gt_diameter<=0:
    #         error_diameter = np.inf
    #     else:
    #         error_diameter = (gt_diameter-diameter)/gt_diameter*100
    #     if gt_len_ab<=0:
    #         error_len_ab = np.inf
    #     else:
    #         error_len_ab = (gt_len_ab-len_ab)/gt_len_ab*100
    #     # incorrect_below4d_ratio = int((length_ratio>4)==(gt_ratio<4)==True)
    #     # correct_below4d_ratio = int((length_ratio<4)==(gt_ratio<4)==True)
    #     # incorrect_above4d_ratio = int((length_ratio<4)==(gt_ratio>4)==True)
    #     row = {'gt_d': round(gt_diameter, 2), 'pred_d': diameter,
    #            'gt_ab': round(gt_len_ab, 2), 'pred_ab': len_ab,
    #            'error_diameter': error_diameter,
    #            'error_len_ab': error_len_ab,
    #            'gt_ratio': gt_ratio, 'pred_ratio': length_ratio,
    #            'gt_ratio>4': int(gt_ratio>4), 'pred_ratio>4': int(length_ratio>4),
    #            'correct_above4d_ratio': correct_above4d_ratio,
    #            'incorrect_above4d_ratio': incorrect_above4d_ratio,
    #            'correct_below4d_ratio': correct_below4d_ratio,
    #            'incorrect_below4d_ratio': incorrect_below4d_ratio,
    #            'image_path':img_path,
    #            }

    #     df = df.append(pd.DataFrame(row, index =[img_filename]) )
    #     if correct_above4d_ratio:
    #         cv2.imwrite(f"{infer_dump_dir}/G(>4D) P(>4D)/{img_filename}", img4)
    #     if incorrect_above4d_ratio:
    #         cv2.imwrite(f"{infer_dump_dir}/G(>4D) P(<4D)/{img_filename}", img4)
    #     if incorrect_below4d_ratio:
    #         cv2.imwrite(f"{infer_dump_dir}/G(<4D) P(>4D)/{img_filename}", img4)
    #     if correct_below4d_ratio:
    #         cv2.imwrite(f"{infer_dump_dir}/G(<4D) P(<4D)/{img_filename}", img4)
    # printj.blue(df)
    # # printj.cyan(df['correct_below4d_ratio'])
    # cm = pd.DataFrame(data=[],columns = ['p: more than 4D', 'p: less than 4D', 'Total'])
    # cm = cm.append(pd.DataFrame({'p: more than 4D':df['correct_above4d_ratio'].sum(),
    #                              'p: less than 4D':df['incorrect_above4d_ratio'].sum(),
    #                              'Total':df['correct_above4d_ratio'].sum()+df['incorrect_above4d_ratio'].sum()}, index =['g: more than 4D']) )
    # cm = cm.append(pd.DataFrame({'p: more than 4D':df['incorrect_below4d_ratio'].sum(),
    #                              'p: less than 4D':df['correct_below4d_ratio'].sum(),
    #                              'Total':df['incorrect_below4d_ratio'].sum()+df['correct_below4d_ratio'].sum()}, index =['g: less than 4D']) )
    # cm = cm.append(pd.DataFrame({'p: more than 4D':df['correct_above4d_ratio'].sum()+df['incorrect_below4d_ratio'].sum(),
    #                              'p: less than 4D':df['incorrect_above4d_ratio'].sum()+df['correct_below4d_ratio'].sum(),
    #                              'Total':df['correct_above4d_ratio'].sum()+df['incorrect_above4d_ratio'].sum()+df['incorrect_below4d_ratio'].sum()+df['correct_below4d_ratio'].sum()}, index =['Total']) )
    # printj.yellow(cm)
    # cm.to_excel(f"{os.path.abspath(f'{path}/..')}/cm_data.xlsx")
    # cm2 = pd.DataFrame(data=[],columns = ['correct', 'incorrect'])
    # cm2 = cm2.append(pd.DataFrame({'correct':df['correct_above4d_ratio'].sum(), 'incorrect':df['incorrect_above4d_ratio'].sum()}, index =['more than 4D']) )
    # cm2 = cm2.append(pd.DataFrame({'correct':df['correct_below4d_ratio'].sum(), 'incorrect':df['incorrect_below4d_ratio'].sum()}, index =['less than 4D']) )
    # printj.cyan(cm2)
    # df.to_excel(f"{os.path.abspath(f'{path}/..')}/test4d_data.xlsx") # pip install openpyx
    # cm.to_excel(f"{os.path.abspath(f'{path}/..')}/cm_data.xlsx") # pip install openpyx
    # total_time = datetime.now()-start
    # info = f'\nDetection count: {len(img_pathlist) - count}, Total: {len(img_pathlist)}'
    # info += f'\nNo detection count: {count}, Total: {len(img_pathlist)}'
    # # Starts # Write inference json
    # output_json_path = f"{infer_dump_dir}/infered_hook.json"

    # info += f'\nTotal inference time: {total_time} \nTime per image: {total_time/len(img_pathlist)}'
    # info += f'\n\nConfusion Matrix for ratio: \n{cm}'
    # printj.blue.bold_on_yellow(info)
    # text_file = f"{infer_dump_dir}/info.txt"
    # if os.path.exists(text_file):
    #     os.remove(text_file)
    # f= open(text_file,"w+")
    # f.write(info)
    # f.close()

    # printj.green.italic_on_black(infer_dump_dir)
    # from cocoeval_hook import run as evaluate
    # # evaluate(output_json_path)
    # os.system('spd-say "Folder Created"')


if __name__ == "__main__":
    now = datetime.now()
    dt_string3 = now.strftime("%Y_%m_%d_%H_%M_%S")
    dt_string3 = now.strftime("%m_%d_%H")

    TEST_PATH = '/home/jitesh/3d/data/coco_data/hook_test/level_01'
    # TEST_PATH = '/home/jitesh/sekisui/teamviewer/sampled_data/VID_20200107_142503/img'
    # TEST_PATH = '/home/jitesh/3d/data/coco_data/hook_real1/s_good'
    # TEST_PATH = '/home/jitesh/3d/data/coco_data/hlk1_100_coco-data/img'
    # TEST_PATH = '/home/jitesh/3d/data/coco_data/hlk2_200_coco-data/img'
    # GT_PATH = f'/home/jitesh/3d/data/coco_data/hook_test/json/hook.json'
    # GT_PATH = f'/home/jitesh/3d/data/coco_data/hook_test/json/hook4.json'
    # WEIGHT_PATH='/home/jitesh/3d/data/coco_data/hook_weights/seg_hook_pole/model_0049999.pth'
    WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hlk1_100_coco-data/weights/Keypoints_R_50_1x_aug_cm_seg_val_5/model_0004999.pth'
    WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_1/model_0019999.pth'
    # WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_2/model_0099999.pth'
    WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_2/model_0049999.pth'
    WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_101_3x_aug_key_seg_val_1/model_0099999.pth'
    # WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_3_hook-only/model_0049999.pth'
    # WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_sim_real_data8/weights/Keypoints_R_50_1x_aug_key_seg_val_2/model_0004999.pth'
    # KEY_WEIGHT_PATH = '/home/jitesh/3d/data/coco_data/hook_weights/seg_hook_pole/model_0049999.pth'
    iteration = WEIGHT_PATH.split('_')[-1].split('.')[0]
    training_data_name = WEIGHT_PATH.split('/')[-2].split('_')[0] + '_'\
        + WEIGHT_PATH.split('/')[6].split('_')[-2] + '_'\
        + WEIGHT_PATH.split('/')[6].split('_')[-1]
    # training_model_name = WEIGHT_PATH.split('/')[-2].split('_')[0]
    kpt_thresh = 0.1
    bbox_thresh = 0.5
    img_size = 1024
    # key = f's{img_size}'
    key = f'hookpole'
    # key = f'hook'
    class_names=['hook', 'pole']
    # class_names=['hook']
    output_dir_path = f'{TEST_PATH}_{dt_string3}_{training_data_name}_{key}_{iteration}_{bbox_thresh}_vis_infer_output_50_1x'

    infer(path=TEST_PATH,
          weights_path=WEIGHT_PATH,
          #   key='X'
          key='c',
          infer_dump_dir=output_dir_path,
          thresh=bbox_thresh,
        #   model='mask_rcnn_R_50_FPN_1x',
            model='mask_rcnn_R_101_FPN_3x',
          size=img_size,
          class_names=class_names,
          #   gt_path=GT_PATH,
          )
