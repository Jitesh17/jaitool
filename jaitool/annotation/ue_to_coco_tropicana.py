import os
from sys import exit as x
from datetime import datetime

import cv2
import numpy as np
import printj
from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Dataset
from annotation_utils.ndds.structs import NDDS_Dataset
from logger import logger

# dataset.display_preview(show_details=True)
# import disk2points_algo_fit_center as d2p
import vis_data
# from cook_data import run as cook


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def id_to_color(RGBint):
    pixel_b = RGBint & 255
    pixel_g = (RGBint >> 8) & 255
    pixel_r = (RGBint >> 16) & 255
    
    return [pixel_b, pixel_g, pixel_r]

def create_mask(img, color):
    lower = np.array(color)-np.array([1]*3)  #, dtype="uint8")
    upper = np.array(color)+np.array([1]*3)  #, dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    return mask
   
now = datetime.now()

dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
dt_string2 = now.strftime("%Y-%m-%d")
dt_string3 = now.strftime("%Y_%m_%d_%H_%M_%S")

# folder_name = f'h5_1000'
# folder_name = f'hc1_1000'
# folder_name = f'hr1_300'
# folder_name = f'hc11_400'
folder_name = f'tropi1'
# folder_name = f'hsk2_200'
# folder_name = f'hlk2_200'
ndds_path = f'/home/jitesh/3d/data/UE_training_results/{folder_name}'
coco_data_dir = f'/home/jitesh/3d/data/coco_data/{folder_name}_coco-data'#_{dt_string3}_coco-data'
make_dir_if_not_exists(os.path.abspath(os.path.join(coco_data_dir, '..')))
make_dir_if_not_exists(coco_data_dir)
# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir=ndds_path,
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    # printj.red(frame.img_path)
    # is_path = frame.img_path.split('.')[0]+'.is.'+frame.img_path.split('.')[-1]
    # # printj.red(is_path)
    # img = cv2.imread(is_path)
    # # printj.cyan(img)
    # img2= img.copy()
    # from PIL import Image
    # img0 = Image.open(is_path)
    # colors = img0.convert('RGB').getcolors()
    # printj.red(colors)
    # x()
    
    # short
    # change1_from = [36, 51, 243]  # red
    # change1_from = [240, 255, 255]  # white
    # # change2_from = id_to_color(15938340)  # pole1
    # change_to = id_to_color(7626000)  # pole0
    # # long
    # # change1_from = [240, 255, 255]  # white
    # # change2_from = [8, 93, 244]  # pole1
    # # change_to = [40, 186, 104]  # pole0
    # # change1_from = list(colors[0][1])[::-1]  # white
    # # change2_from = list(colors[1][1])[::-1]  # pole1
    # # change_to = list(colors[-1][1])[::-1]  # pole0
    # # printj.red(change1_from)
    # # printj.red(change2_from)
    # # printj.red(change_to)
    
    # mask1 = create_mask(img, change1_from)
    # img2[mask1==255]=change_to
    # mask2 = create_mask(img, change2_from)
    # img2[mask2==255]=change_to
    # cv2.imshow('img', img)
    # cv2.waitKey(111111)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(11111)
    # x()
    # cv2.imwrite(is_path, img2)
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        # printj.yellow.on_black(ann_obj)
        # all_keys = set().union(*(d for d in ann_obj))
        # printj.yellow.bold_on_black(all_keys)
        # raise Exception
        if ann_obj.class_name.startswith('tropicana'):
            obj_type, obj_name = 'seg', 'tropicana'
            instance_name = '0' #ann_obj.class_name #.replace('hook', '')
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        
            # printj.yellow( ann_obj.class_name)
        else:
            logger.error(f'ann_obj.class_name: {ann_obj.class_name}')
            # raise Exception
    
    # Delete Duplicate Objects
    frame.ndds_ann.objects.delete_duplicates(verbose=True, verbose_ref=frame.img_path)

# ndds_dataset.save_to_path(save_path=f'{coco_data_dir}/hook_fixed_ndds.json', overwrite=True)

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    # categories=COCO_Category_Handler.load_from_path(f'/home/jitesh/3d/data/categories/hook_7ckpt.json'),
    categories=COCO_Category_Handler.load_from_path(f'/home/jitesh/3d/data/categories/tropicana.json'),
    naming_rule='type_object_instance_contained',
    ignore_unspecified_categories=True,
    show_pbar=True,
    bbox_area_threshold=1,
    default_visibility_threshold=-1,
    allow_unfound_seg=True,
)
make_dir_if_not_exists(coco_data_dir)
img_path = f'{coco_data_dir}/img'
make_dir_if_not_exists(coco_data_dir)
ann_dir = f'{coco_data_dir}/json'
make_dir_if_not_exists(ann_dir)
dataset.move_images(
        dst_img_dir=img_path,
        preserve_filenames=False,
        update_img_paths=True,
        overwrite=True,
        show_pbar=True
    )
# if not os.path.exists(coco_data_dir):
#     os.makedirs(coco_data_dir)
key='tropicana'
dataset.save_to_path(f'{ann_dir}/{key}.json', overwrite=True)
# new_dataset = d2p.run(ann_dir)
# new_dataset.display_preview(show_details=True)

# new_dataset = cook(img_path, f'{ann_dir}/new-hook.json', key_num=11)
# new_dataset = cook(img_path, f'{ann_dir}/{key}.json', key_num=7)

vis_data.complete(
    # img_dir=f'{coco_data_dir}',
    img_dir=f'{img_path}',
    json_path=f'{ann_dir}/{key}.json',
    show_image=True,
    show_video=True,
    show_seg=True,
                      )
