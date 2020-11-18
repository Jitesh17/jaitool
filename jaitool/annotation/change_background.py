import os
import random
from datetime import datetime
from sys import exit as x

# from jaitool.annotation.COCO import COCO_Datase
# from jaitool.annotation.NDDS import NDDS_Dataset
# from logger import logger
import albumentations as A
import cv2
import numpy as np
import printj
from pyjeasy.image_utils.edit import get_all_colors
from albumentations.augmentations.functional import rotate
from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.ndds.structs import NDDS_Dataset


from pyjeasy.file_utils import (dir_contents_path_list_with_extension,
                                make_dir_if_not_exists)
from pyjeasy.image_utils.edit import resize_img
from pyjeasy.image_utils.preview import show_image
from tqdm import tqdm


def aug_flip_and_rotate(load_path=None):
        if load_path:
            return A.load(load_path)
        else:
            aug_seq = A.Compose([
                A.Rotate(limit=(-90, 90), p=0.5),
                A.Flip(p=0.5),
                A.OpticalDistortion(
                    distort_limit=0.05, shift_limit=0.05, 
                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, 
                    value=None, mask_value=None, always_apply=False, 
                    p=0.5)
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

def create_mask_from_color(img, color):
    color = list(reversed(color))
    lower = np.array(color)-np.array([10]*3)  #, dtype="uint8")
    upper = np.array(color)+np.array([10]*3)  #, dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    return mask

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
    
def replace_bg_wrt_isimg(
    ndds_data_dir: str, 
    coco_data_dir: str, 
    bg_dirs: list,
    json_filename: str=None, 
    bg_iscolor: list=None,
    output_img_dir_name: str="img_",
    aug_on: bool=False,
    aug_json: str=None,
    show_preview: bool=False,
    ): 
    make_dir_if_not_exists(os.path.abspath(os.path.join(coco_data_dir, '../..')))
    make_dir_if_not_exists(os.path.abspath(os.path.join(coco_data_dir, '..')))
    make_dir_if_not_exists(coco_data_dir)
    # Load NDDS Dataset
    ndds_dataset = NDDS_Dataset.load_from_dir(
        json_dir=ndds_data_dir,
        show_pbar=True
    )
    coco_dataset = COCO_Dataset.load_from_path(json_path=f"{coco_data_dir}/json/{json_filename}.json", check_paths=False)
    
    image_path_list = []
    for bg_dir in bg_dirs:
        image_path_list += dir_contents_path_list_with_extension(
            dirpath=bg_dir,
            extension=['.jpg', '.jpeg', '.png'])
    bg_gen = image_sequence(image_path_list)
    # pbar = tqdm(ndds_dataset.frames, colour='#44aa44')
    pbar = tqdm(coco_dataset.images, colour='#44aa44')
    bg_gen = image_sequence(image_path_list)
    for image in pbar:
        pbar.set_description("Changing background")
        # pbar.set_postfix({'file_name': image.file_name})
        is_path = ndds_data_dir + '/' + image.file_name.split('.')[0]+'.is.'+image.file_name.split('.')[-1]
        # img_path = ndds_data_dir + 
        printj.green(image.coco_url)
        printj.green(is_path)
        img = cv2.imread(image.coco_url)
        is_img = cv2.imread(is_path)
        is_img2= is_img.copy()
        # from PIL import Image
        # img0 = Image.open(is_path)
        # colors = img0.convert('RGB').getcolors()
        # printj.red(colors)
        if bg_iscolor:
            mask = create_mask_from_color(is_img2, bg_iscolor)
        else:
            colors = get_all_colors(img_path=is_path)  # img.convert('RGB').getcolors()
            _bg_iscolor = [list(colors[0][-1])]
            printj.cyan(f"\nAll {len(colors)} colors in the image: {colors}")
            printj.yellow(f'Background color is {_bg_iscolor}')
            mask = create_mask_from_color(is_img2, _bg_iscolor)
            
        background = next(bg_gen)
        if aug_on:
            aug = aug_flip_and_rotate(aug_json)
            background = aug(image=np.array(background))['image']
        background = resize_img(src=background, size=(img.shape[1], img.shape[0]))
        bg = cv2.bitwise_or(background, background, mask=mask)
        mask = cv2.bitwise_not(mask)
        fg = cv2.bitwise_or(img, img, mask=mask)
        final = cv2.bitwise_or(fg, bg)
        output = os.path.join(coco_data_dir, output_img_dir_name)
        make_dir_if_not_exists(coco_data_dir)
        make_dir_if_not_exists(output)
        collaged_output = os.path.join(output,image.file_name)
        if show_preview:
            quit = show_image(final)
            if quit:
                break
        else:
            cv2.imwrite(collaged_output, final)

if __name__ == "__main__":
    now = datetime.now()

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
