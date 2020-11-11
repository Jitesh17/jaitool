import os
import random
from datetime import datetime
from sys import exit as x

import albumentations as A
from albumentations.augmentations.functional import rotate
import cv2
import numpy as np
import printj
from jaitool.annotation.COCO import COCO_Dataset
from pyjeasy.file_utils import (dir_contents_path_list_with_extension,
                                make_dir_if_not_exists)
from pyjeasy.image_utils.edit import resize_img
# from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Dataset
# from annotation_utils.ndds.structs import NDDS_Dataset
# from logger import logger
from pyjeasy.image_utils.preview import show_image
from tqdm import tqdm


def aug_flip_and_rotate(load_path=None):
        if load_path:
            return A.load(load_path)
        else:
            aug_seq = A.Compose([
                A.Rotate(limit=(-90, 90), p=0.5),
                A.Flip(p=0.5),
            ])
            return aug_seq

def image_sequence(image_path_list):
    num = 0
    while True:
        yield cv2.imread(image_path_list[num])
        if num + 2 < len(image_path_list):
            num += 1
        else:
            num = 0
            random.shuffle(image_path_list)

def replace_bg_wrt_seg_ann(
    coco_data_dir: str, 
    json_filename: str, 
    bg_dirs: list,
    img_dir_name: str="img",
    output_img_dir_name: str="img_",
    aug_on: bool=False,
    aug_json: str=None,
    show_preview: bool=False,
    ): 
    coco_dataset = COCO_Dataset.load_from_path(json_path=f"{coco_data_dir}/json/{json_filename}.json", check_paths=False)
    # image_path_list = folder_list(folder1)
    image_path_list = []
    for bg_dir in bg_dirs:
        image_path_list += dir_contents_path_list_with_extension(
            dirpath=bg_dir,
            extension=['.jpg', '.jpeg', '.png'])
    bg_gen = image_sequence(image_path_list)
    pbar = tqdm(coco_dataset.images, colour='#44aa44')
    for image in pbar:
        pbar.set_description("Changing background")
        pbar.set_postfix({'file_name': image.file_name})
        image_path_split = image.coco_url.split("/")
        image_path_split[-2] = img_dir_name
        image_path = "/".join(image_path_split) 
        
        for ann in coco_dataset.annotations:
            if ann.image_id == image.id:
                seg = ann.segmentation
                
                background = next(bg_gen)
                if aug_on:
                    aug = aug_flip_and_rotate(aug_json)
                    background = aug(image=np.array(background))['image']
                orig_image = cv2.imread(image_path)
                assert orig_image.shape[1] == image.width
                assert orig_image.shape[0] == image.height
                mask = np.zeros((image.width, image.height), np.uint8)
                contours = seg.to_contour()
                cv2.drawContours(mask, contours, -1, (255,255,255),-1)
                final = replace_bg_wrt_mask(orig_image, background, mask)
                
                if show_preview:
                    show_image(final)
                else:
                    output = os.path.join(coco_data_dir, output_img_dir_name)
                    make_dir_if_not_exists(coco_data_dir)
                    make_dir_if_not_exists(output)
                    output_path = os.path.join(output, image.file_name)
                    
                    cv2.imwrite(output_path, final)

def replace_bg_wrt_mask(orig_image, background, mask):
    fg = cv2.bitwise_or(orig_image, orig_image, mask=mask)
    mask = cv2.bitwise_not(mask)
    background = resize_img(src=background, size=(orig_image.shape[0], orig_image.shape[1]))
    bg = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bg)
    return final
    


if __name__ == "__main__":
    now = datetime.now()

    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    dt_string2 = now.strftime("%Y-%m-%d")
    dt_string3 = now.strftime("%Y_%m_%d_%H_%M_%S")

    key='bolt'
    # folder_name = f'b8'
    # coco_data_dir = f'/home/jitesh/3d/data/coco_data/bolt/{folder_name}_coco-data'#_{dt_string3}_coco-data'
    folder_name = f'bolt_3-4'
    # folder_name = f'ram-bolt'
    coco_data_dir = f'/home/jitesh/3d/data/coco_data/bolt/{folder_name}'#_{dt_string3}_coco-data'
    bg_dirs=["/home/jitesh/3d/data/images_for_ndds_bg/solar_panel"]
    # bg_dirs.append("/home/jitesh/3d/data/images_for_ndds_bg/collaged_images_random-size")
    # bg_dirs.append("/home/jitesh/3d/data/images_for_ndds_bg/collaged_images_random-size-v")
    
    replace_bg_wrt_seg_ann(
        coco_data_dir=coco_data_dir, 
        json_filename=key, 
        bg_dirs=bg_dirs,
        img_dir_name="img0",
        aug_on=True)
