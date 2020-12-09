import os

import cv2
import numpy as np
# import printj
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.common_types.segmentation import Segmentation
from tqdm import tqdm


def merge_categories(json_path :str, output_json_path :str, merge_from :list,
         merge_to :int = None):
    """# merge categories
    - json_path :str = input json file path, 
    - output_json_path :str = output json file path, 
    - merge_from :list = list of categories id to merge/combine into one category,
    - merge_to :int = category id to merge to, default is the first elemnt of 'merge_from'
    """
    if merge_to is None:
        merge_to = merge_from[0]
    coco_dataset = COCO_Dataset.load_from_path(
        json_path=json_path,
        check_paths=False,
        strict=False,
    )
    pbar = tqdm(coco_dataset.images, colour='#44aa44')
    for image in pbar:
        pbar.set_description("Combining Categories")
        pbar.set_postfix({'file_name': image.file_name})
        mask = np.zeros((image.width, image.height), np.uint8)
        for ann in coco_dataset.annotations:
            if ann.image_id == image.id:
                if ann.category_id in merge_from:
                    seg = ann.segmentation
                    contours = seg.to_contour()
                    mask0 = np.zeros((image.width, image.height), np.uint8)
                    cv2.drawContours(mask0, contours, -1, (255, 255, 255), -1)
                    mask = mask + mask0
                    # if show_image(mask):
                    #     return

        color_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg = Segmentation.from_contour(
            contour_list=color_contours, exclude_invalid_polygons=True)
        for ann in coco_dataset.annotations:
            if ann.image_id == image.id:
                if ann.category_id == merge_to:
                    ann.segmentation = seg
                    ann.bbox = seg.to_bbox()
    coco_dataset.categories.remove(
        [cat_id for cat_id in merge_from if cat_id is not merge_to])

    ouput_dir = os.path.abspath(f'{output_json_path}/..')
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    coco_dataset.save_to_path(output_json_path, overwrite=True)


if __name__ == "__main__":
    json_path = f'/home/jitesh/3d/data/coco_data/bolt/bp8_0/img/coco_annotations.json'

    output_json_path = os.path.abspath(
        f'{json_path}/../../json') + '/bolt.json'
    merge_from = [0, 1, 2, 3]
    merge_to = 0
    merge_categories(json_path, output_json_path, merge_from, merge_to)
