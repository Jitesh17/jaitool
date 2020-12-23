from __future__ import annotations

# try:
#     from detectron2.structures.boxes import BoxMode
# except ImportError:
#     printj.red(f'Refer to https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md')
#     raise ImportError
import json
import os
import sys
from typing import List

import cv2
import printj
from jaitool.structures.point import Point2D
from jaitool.structures.bbox import BBox
import functools
from pyjeasy.file_utils import make_dir_if_not_exists
# from jaitool.structures import BBox, Keypoint2D_List, Segmentation
# from pyjeasy.base import (BasicHandler, BasicLoadableHandler,
#                           BasicLoadableObject)
# from pyjeasy.check_utils import check_required_keys


# import numpy as np
# from tqdm import tqdm
import operator


class LabelmeLoader:
    def __init__(self, path=None):
        self.path = path
        with open(self.path) as json_file:
            self.l_data = json.load(json_file)
            
        
    def create_cropped_labelme(self, c_point1, c_point2, output_img, theshold: float=1.0):
        data = dict()
        data["version"] = self.l_data["version"]
        data["flags"] = dict()
        data["shapes"] = []
        useful_img = False
        frame_box = BBox.from_list(functools.reduce(operator.iconcat, [c_point1, c_point2], [])).to_int()
        for a in self.l_data["shapes"]:
            # p1, p2 = [Point2D.from_list(p) for p in a["points"]]
            bbox = BBox.from_list(functools.reduce(operator.iconcat, a["points"], []))
            fin_box = bbox.coord_in_cropped_frame(frame_box, theshold)
            # if all(p.inside_rectangle(c_point1, c_point2) for p in [p1, p2])  :
            if fin_box.xmax > 0:
                # p1 -= c_point1
                # p2 -= c_point1
                _to_append = dict()
                _to_append["label"] = a["label"]
                _to_append["points"] = [[fin_box.xmin, fin_box.ymin], [fin_box.xmax, fin_box.ymax]]
                _to_append["group_id"] = None
                _to_append["shape_type"] = "rectangle"
                _to_append["flags"] = {}
                data["shapes"].append(_to_append)
                useful_img = True
        data["imagePath"] = output_img
        data["imageData"] = None
        data["imageHeight"] = frame_box.height
        data["imageWidth"] = frame_box.width
        output_json = f'{output_img.split(".")[-2]}.json'
        if useful_img:
            with open(output_json, 'w') as outfile:
                json.dump(data, outfile, indent=2)
        return useful_img

def chop(
    input_img_path,
    output_path,
    c_sizes,
    c_strides,
    padding_color=[255, 255, 255],
    input_json_path=None, 
    theshold: float=1.0
): 
    lbm = None
    if input_json_path:
        lbm = LabelmeLoader(input_json_path)
    orig = cv2.imread(input_img_path)
    gray = cv2.cvtColor(orig,cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    img_h, img_w = gray.shape
    img = cv2.copyMakeBorder(
        img, top=0, bottom=max(c_sizes), left=0, right=max(c_sizes), 
        borderType=cv2.BORDER_CONSTANT, value=padding_color)
    make_dir_if_not_exists(os.path.abspath(f'{output_path}/..'))
    make_dir_if_not_exists(output_path)
    for c_size, c_stride in zip(c_sizes, c_strides):
        i = 0
        h = 0
        while h < img_h:
            sys.stdout.write(f"\rCreating image {i} if size {c_size}")
            sys.stdout.flush()
            w = 0
            while w < img_w:
                c_img = img[h:h+c_size, w:w+c_size, :]
                # if show_image(c_img, f"size: {c_size}, horizntal: {w}-{w+c_size}, vertical: {h}-{h+c_size}", 900):
                #     return
                output = os.path.join(output_path, f't_{c_size}_{i}.jpg')
                useful_img = lbm.create_cropped_labelme(
                    c_point1=[w, h], 
                    c_point2=[w+c_size, h+c_size], 
                    output_img=f'{output}',
                    theshold=theshold)
                if useful_img:
                    cv2.imwrite(f'{output}', c_img)
                i += 1
                w += c_size - c_stride
            h += c_size - c_stride
    sys.stdout.write(f"\rCreated images of sizes {c_sizes}. {' '*50}\n")

if __name__ == "__main__":
    test_img_num = 1
    input_img_path = f"/home/jitesh/3d/data/coco_data/ed/test_data/test_{test_img_num}.jpg"
    output_path = f"/home/jitesh/3d/data/coco_data/ed/test_data/img_{test_img_num}-3/img"
    
    c_sizes = [
        # 500,
        # 700, 
        800, 
        900, 
        1000, 
        1100, 
        1200, 
        1300, 
        1400, 
        1500, 
        1700, 
        1900, 
        2000, 
        2300]
    c_strides = [100]*len(c_sizes)
    chop(
        input_img_path,
        output_path,
        c_sizes,
        c_strides,
        padding_color=[255, 255, 255],
        input_json_path=f'/home/jitesh/3d/data/coco_data/ed/test_data/test_{test_img_num}.json',
        theshold = 0.8
         )
