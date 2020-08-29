import cv2
from pyjeasy.file_utils import (dir_exists, file_exists, delete_dir, 
                                make_dir, make_dir_if_not_exists)
from pyjeasy.math_utils import dist
import numpy as np


def draw_inference_on_hook2(
    img: np.ndarray, cleaned_keypoints, kpt_labels: List[str], kpt_skeleton: List[list],
    score: float, bbox: BBox, vis_keypoints: list, kpt_confidences: list, conf_idx_list: list, not_conf_idx_list: list,
    conf_keypoints, conf_kpt_labels, not_conf_keypoints, not_conf_kpt_labels,
    conf_thresh: float = 0.3, show_bbox_border: bool = False, bbox_label_mode: str = 'euler', index_offset: int = 0, diameter=1
):
    # printj.red(len(vis_keypoints))
    result = img.copy()
    # diameter = 10
    # printj.yellow(f'bbox = {bbox}')
    # printj.yellow(f'vis_keypoints = {vis_keypoints}')
    # printj.yellow(f'dist = {self.dist([0, 1], [1, 2])}')
    point_a = vis_keypoints[0]
    point_b = vis_keypoints[1]
    point_cb = vis_keypoints[2]
    point_c = vis_keypoints[3]
    point_cd = vis_keypoints[4]
    point_d = vis_keypoints[5]
    point_e = vis_keypoints[6]
    # point_dl = vis_keypoints[5]
    # point_dr = vis_keypoints[6]
    len_ab = dist(point_a, point_b)
    # printj.red(len_ab)
    if diameter <= 0:
        length_ratio = np.inf
    else:
        length_ratio = len_ab / diameter
    pass_condition = (length_ratio > 4)
    bbox_color = [0, 0, 255]  # fail: ab < 4*d_rl
    kpt_ab_color = [50, 255, 255]
    c_text = 'ab < 4D'
    if pass_condition:
        bbox_color = [0, 255, 0]  # pass: ab > 4*d_rl
        kpt_ab_color = [0, 255, 0]
        c_text = 'ab > 4D'

    # bbox_height = np.absolute(bbox.ymax - bbox.ymin)
    # bbox_width = np.absolute(bbox.xmax - bbox.xmin)
    # length_diff = np.absolute(bbox_height - bbox_width)
    # if bbox_height > bbox_width:
    #     bbox.xmin = bbox.xmin - int(length_diff/2)
    #     bbox.xmax = bbox.xmax + int(length_diff/2)
    # else:
    #     bbox.ymin = bbox.ymin - int(length_diff/2)
    #     bbox.ymax = bbox.ymax + int(length_diff/2)

    # printj.cyan(bbox)
    # printj.cyan(bbox.to_int())
    # printj.cyan(bbox.to_int().to_list())
    if bbox_label_mode == 'euler':
        # bbox_text = str(round(length_ratio, 2)) + 'D'
        bbox_text = f'h {score}'
        result = draw_bbox(img=result, color=bbox_color,  bbox=bbox, text=bbox_text,
                           label_only=not show_bbox_border, label_orientation='top')
        # result = draw_bbox(img=result, color=bbox_color,  bbox=bbox, text=c_text,
        #                    label_only=not show_bbox_border, label_orientation='bottom')
        result = draw_bbox(img=result, color=bbox_color,  bbox=bbox, text=str(score),
                           label_only=not show_bbox_border, label_orientation='bottom')
    result = draw_skeleton(
        img=result, keypoints=vis_keypoints, keypoint_skeleton=kpt_skeleton, index_offset=index_offset, thickness=2, color=[255, 0, 0],
        ignore_kpt_idx=[]
    )
    # ab
    result = draw_skeleton(
        img=result, keypoints=vis_keypoints, keypoint_skeleton=kpt_skeleton, index_offset=index_offset, thickness=2, color=kpt_ab_color,
        ignore_kpt_idx=[2, 3, 4, 5, 6]
    )
    # d_lr
    # result = draw_skeleton(
    #     img=result, keypoints=vis_keypoints, keypoint_skeleton=kpt_skeleton, index_offset=index_offset, thickness=2, color=[255, 255, 0],
    #     ignore_kpt_idx=[0, 1, 2, 3, 4]
    # )
    result = draw_keypoints(
        img=result, keypoints=vis_keypoints, radius=2, color=[0, 0, 255], keypoint_labels=kpt_labels, show_keypoints_labels=True, label_thickness=1,
        ignore_kpt_idx=conf_idx_list)
    if len(conf_keypoints) > 0:
        result = draw_keypoints(
            img=result, keypoints=vis_keypoints, radius=2, color=[0, 255, 0], keypoint_labels=kpt_labels, show_keypoints_labels=True, label_thickness=1,
            ignore_kpt_idx=not_conf_idx_list
        )
    return result, len_ab


def draw_info_box(image, len_ab, diameter):
    result_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    TopLeftCornerOfText = (10, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.rectangle(result_image, (5, 10), (280, 180), (0, 0, 0), -1)
    cv2.rectangle(result_image, (5, 10), (280, 180), (200, 200, 0), 2)
    cv2.putText(result_image, f'Len-ab: {round(len_ab,2)}',
                (10, 50), font, fontScale, fontColor, lineType)
    if diameter <= 0:
        cv2.putText(result_image, f'No Pole', (10, 100), font,
                    fontScale, fontColor, lineType)
        cv2.putText(result_image, 'No Diameter', (10, 150),
                    font, fontScale, fontColor, lineType)
    else:
        length_ratio = len_ab/diameter
        cv2.putText(result_image, f'Diameter: {round(diameter,2)}',
                    (10, 100), font, fontScale, fontColor, lineType)
        cv2.putText(result_image, str(round(length_ratio, 2))+' D',
                    (10, 150), font, fontScale, fontColor, lineType)

    return result_image

def confirm_folder(path, mode):
    # make_dir_if_not_exists(path)
    if mode == 'save':
        if dir_exists(path):
            delete_dir(path)
            make_dir(path)
        else:
            make_dir(path)
            