import os
import sys

from common_utils.file_utils import (delete_all_files_in_dir, file_exists,
                                     get_dir_contents_len,
                                     make_dir_if_not_exists)
from common_utils.path_utils import get_dirnames_in_dir
from logger import logger

from pasonatron.det2.lib.train.seg import Detectron2SegmentationEvalTrainer
from pasonatron.det2.lib.train.kpt import Detectron2KeypointEvalTrainer, Detectron2KeypointSegmentationEvalTrainer
from pasonatron.det2.lib.trainer import COCO_Segmentation_EvalTrainer, COCO_Keypoint_EvalTrainer, COCO_KeypointSegmentation_EvalTrainer

# scale = 0.1
# key = 'cropped_hook_train'
# key = f'cropped_hook_{scale}_train'
# key = f'tropicana_train'
k_name = 'tropicana'
key = f'{k_name}_train'
# key = f'hook_train'
# path = "/home/jitesh/3d/data/coco_data/h11_600_coco-data"
path = "/home/jitesh/3d/data/coco_data/tropi1_coco-data"
# path = f"/home/jitesh/3d/data/coco_data/hook_sim_real_data7_{scale}"
# path = f"/home/jitesh/3d/data/coco_data/hlk1_100_coco-data"
img_path = f'{path}/img'
# coco_ann_path = os.path.join(path, "json/no_seg_hook.json")
# coco_ann_path = f'{path}/json/tested-hook.json'  # Segmentation
# coco_ann_path = f'{path}/json/{k_name}_train.json'  # Segmentation
# coco_ann_path = os.path.join(path, "json/bbox_resized.json")
# model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"mask_rcnn_R_101_FPN_3x
model_name = 'mask_rcnn_R_50_FPN_1x'
# model_name = 'mask_rcnn_R_101_FPN_3x'
# model = f"COCO-InstanceSegmentation/{model_name}.yaml"
# model_name = 'keypoint_rcnn_R_50_FPN_1x'
# model_name = 'keypoint_rcnn_R_101_FPN_3x'
model = f"COCO-Keypoints/{model_name}.yaml"
# model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"
# model = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

_model_name = model.split('/')[0].split('-')[1] + '_'\
            + model.split('/')[1].split('_')[2] + '_'\
            + model.split('/')[1].split('_')[3] + '_'\
            + model.split('/')[1].split('_')[5].split('.')[0] 
make_dir_if_not_exists(f'{path}/weights')
output_dir_path = f'{path}/weights/{_model_name}_aug_val_1'
# root_dir = '/home/doors/workspace/darwin/dataset_config/20200722_trainval'
# dataset_root_dir = f'{root_dir}/dataset_combined/all'
# root_dir=f'/home/jitesh/3d/data/coco_data/hook_sim_real_data6/output'
dataset_root_dir = path
# targets = get_dirnames_in_dir(dataset_root_dir)
# target_dirs = [f'{dataset_root_dir}/{target}' for target in targets]
# # target_train_dirs = [f'{target_dir}/train' for target_dir in target_dirs]
# target_train_dirs = [f'{target_dir}' for target_dir, target in zip(target_dirs, targets) if target.startswith('train')]
# target_val_dirs = [f'{target_dir}' for target_dir, target in zip(target_dirs, targets) if target.startswith('val')]
# # target_val_dirs = [f'{target_dir}/val' for target_dir in target_dirs]
# target_train_coco_ann_paths = [f'{target_train_dir}/coco/output.json' for target_train_dir in target_train_dirs]
# target_val_coco_ann_paths = [f'{target_val_dir}/coco/output.json' for target_val_dir in target_val_dirs]
# target_train_img_dirs = [f'{target_train_dir}/img' for target_train_dir in target_train_dirs]
# target_val_img_dirs = [f'{target_val_dir}/img' for target_val_dir in target_val_dirs]
output_dir = output_dir_path
target_train_coco_ann_paths = [f'{path}/json/{key}.json']
target_train_img_dirs = [f'{img_path}']
target_val_coco_ann_paths = []
target_val_img_dirs = []
val_paths = [
        f'/home/jitesh/3d/data/coco_data/tropi1_coco-data',
        # f'/home/jitesh/3d/data/coco_data/hook_real_training_data4',
        # f'/home/jitesh/3d/data/coco_data/hsk2_200_coco-data',
        # f'/home/jitesh/3d/data/coco_data/hmk2_200_coco-data',
        # f'/home/jitesh/3d/data/coco_data/hlk2_200_coco-data',
        # f'/home/jitesh/3d/data/coco_data/hook_test',
    ]
    
# val_paths = [f'/home/jitesh/3d/data/coco_data/hlk1_100_coco-data']
    
all_datasets = []
# val_json_name='cropped_hook_test'
# val_json_name=f'cropped_hook_{scale}_test'
val_json_name=f'{k_name}_test'
# val_json_name=f'hook_test'
for path in val_paths:
    target_val_coco_ann_paths.append(f'{path}/json/{val_json_name}.json')
    # target_val_img_dirs.append(f'{path}/cropped_hook_img')
    # target_val_img_dirs.append(f'{path}/cropped_hook_{scale}_img')
    # target_val_img_dirs.append(f'{path}/img')
    target_val_img_dirs.append(f'{path}/img')
# train_instance_name = [{'Train'}]
train_instance_name = ['Train']
# val_instance_name = [{'val_real'}, {'Val_hook_medium'}, {'Val_hook_long'}] 
val_instance_name = [
    'Val_sim',
    # 'Val_real',
    # 'Val_hook_short',
    # 'Val_hook_medium',
    # 'Val_hook_long',
    # 'Val_Test149'
    ] 
# val_instance_name = [{'val0'}, {'val1'}, {'val3'}]  
# val_instance_name = ['Val_hook_long']  

logger.yellow(f'target_train_coco_ann_paths = {target_train_coco_ann_paths}')
logger.yellow(f'target_train_img_dirs = {target_train_img_dirs}')
logger.yellow(f'target_val_coco_ann_paths = {target_val_coco_ann_paths}')
logger.yellow(f'target_val_img_dirs = {target_val_img_dirs}')
# targets = ['val1', 'train', 'val0']
# logger.yellow(f'targets = {targets}')
# logger.yellow(f'train_instance_name = {[{target} for target in targets if target.startswith("train")]}')
# logger.yellow(f'val_instance_name = {[{target} for target in targets if target.startswith("val")]}')
# sys.exit()
# trainer = Detectron2KeypointEvalTrainer(
# trainer = Detectron2KeypointSegmentationEvalTrainer(
trainer = Detectron2SegmentationEvalTrainer(
    train_coco_ann_path=target_train_coco_ann_paths,
    train_img_dir=target_train_img_dirs,
    # train_instance_name=[f'{target}' for target in targets if target.startswith('train')],
    train_instance_name=train_instance_name,
    val_coco_ann_path=target_val_coco_ann_paths,
    val_img_dir=target_val_img_dirs,
    # val_instance_name=[f'{target}' for target in targets if target.startswith('val')],
    val_instance_name=val_instance_name,
    output_dir=output_dir,
    # trainer_constructor=COCO_KeypointSegmentation_EvalTrainer,
    trainer_constructor=COCO_Segmentation_EvalTrainer,
    images_per_batch=1,
    # kpt_idx_offset=-1,
    model_name=model_name,
    # batch_size_per_image=1024,
    batch_size_per_image=512,
    # min_size_train=1024, max_size_train=1024,
    min_size_train=1024, max_size_train=1024,
    max_iter=20000,
    checkpoint_period=1000,
    eval_period=250,
    # oks_sigmas=[0.3]*7,
    save_vis=True,
    # seg_train=True,
)
trainer.train(resume=True)
# trainer.train(resume=False)
