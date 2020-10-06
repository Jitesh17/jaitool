
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.common_types.segmentation import Segmentation
from common_utils.common_types.keypoint import Keypoint2D_List
from datetime import datetime
from logger import logger
import json

now = datetime.now()

dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
dt_string2 = now.strftime("%Y-%m-%d")
dt_string3 = now.strftime("%Y_%m_%d_%H_%M_%S")
def run(path: str,
        key: str='measure',
        show_image: bool=False,
        show_video: bool=True,
        ):
    dataset = COCO_Dataset.load_from_path(
        json_path=f'{path}/{key}-coco.json',
        img_dir=path,
        strict=False
    )
    # for i, coco_ann in enumerate(dataset.annotations):
    #     if i % 2 == 0:
    #         coco_ann.segmentation = Segmentation()
    #     if i % 3 == 0:
    #         coco_ann.keypoints = Keypoint2D_List()
    #         coco_ann.num_keypoints = 0
    for coco_ann in dataset.annotations:
        coco_ann.segmentation = Segmentation()
        coco_ann.keypoints = Keypoint2D_List()
        coco_ann.num_keypoints = 0
        coco_ann.keypoints_3d = None
        coco_ann.camera = None
    for coco_cat in dataset.categories:
        coco_cat.keypoints = []
        coco_cat.skeleton = []
    dataset.save_to_path('non_strict_dataset.json', overwrite=True, strict=False)
    dataset0 = COCO_Dataset.load_from_path('non_strict_dataset.json', strict=False)
    dataset0.images.sort(attr_name='file_name')
    
    if show_image:
        dataset0.save_visualization(
            save_dir=f'{path}_{dt_string3}_visualize',
            show_preview=True,
            kpt_idx_offset=-1,
            overwrite=True,
            show_details=True
        )
    if show_video:
        dataset0.save_video(
            save_path=f'{path}_{dt_string3}.mp4',
            # draw_order=['screw'],
            show_details=True,
            show_preview=True,
            kpt_idx_offset=-1,
            overwrite=True,
            fps=5,
            show_seg=True
        )
    logger.green('Visualisation complete')
    
    
def complete(img_dir: str,
        json_path: str,
        show_preview: bool=False,
        show_image: bool=False,
        show_video: bool=False,
        show_seg: bool=False,
        kpt_idx_offset: int=0,
        ):
    logger.yellow("Visualisation starts")
    # dataset = COCO_Dataset.load_from_path(
    #     json_path=json_path,
    #     img_dir=img_dir,
    #     strict=False
    # )
    strict=False
    json_dict = json.load(open(json_path, 'r'))
    dataset = COCO_Dataset.from_dict(json_dict, strict=strict)
    # for i, coco_ann in enumerate(dataset.annotations):
    #     if i % 2 == 0:
    #         coco_ann.segmentation = Segmentation()
    #     if i % 3 == 0:
    #         coco_ann.keypoints = Keypoint2D_List()
    #         coco_ann.num_keypoints = 0
    # for coco_ann in dataset.annotations:
    #     coco_ann.segmentation = Segmentation()
    #     coco_ann.keypoints = Keypoint2D_List()
    #     coco_ann.num_keypoints = 0
    #     coco_ann.keypoints_3d = None
    #     coco_ann.camera = None
    # for coco_cat in dataset.categories:
    #     coco_cat.keypoints = []
    #     coco_cat.skeleton = []
    dataset.save_to_path('non_strict_dataset.json', overwrite=True, strict=False)
    dataset0 = COCO_Dataset.load_from_path('non_strict_dataset.json', strict=False)
    dataset0.images.sort(attr_name='file_name')
    #Show image
    # if show_preview==True:
    #     cv2.imshow('compare images', compare_images)
    #     # cv2.waitKey(10000)
    #     key = cv2.waitKey(0) & 0xFF
    #     printj.blue(key)
    #     if key == ord('q'):
    #         quit_flag = True
    #         cv2.destroyAllWindows()
    #         # break
    if show_image:
        dataset0.save_visualization(
            save_dir=f'{img_dir}_{dt_string3}_visualize',
            show_preview=show_preview,
            kpt_idx_offset=kpt_idx_offset,
            overwrite=True,
            show_details=True,
            show_seg=show_seg
            
        )
    if show_video:
        dataset0.save_video(
            save_path=f'{img_dir}_{dt_string3}.mp4',
            # draw_order=['screw'],
            show_details=True,
            show_preview=show_preview,
            kpt_idx_offset=kpt_idx_offset,
            overwrite=True,
            fps=5,
            show_seg=show_seg
        )
    logger.green('Visualisation complete')

if __name__ == "__main__":

    # run(path="/home/jitesh/3d/data/coco_data/screw1_500_coco-data",
    #     show_image=False,
    #     show_video=True,)
    # run(path="/home/jitesh/3d/data/coco_data/screw2_500_coco-data",
    #     show_image=False,
    #     show_video=True,)
    # run(path="/home/jitesh/3d/data/coco_data/mp_v_500_07_05_2020_15_43_15_coco-data",
    #     show_image=False,
    #     show_video=True,)
    coco_data_dir='/home/jitesh/3d/data/coco_data/mp_v_500_07_05_2020_15_43_15_coco-data'
    # complete(img_dir=coco_data_dir,
    #                   json_path=f'{coco_data_dir}/full-measure-coco.json',
    #                   show_image=False,
    #                   show_video=True,
    #                   )
    # path = '/home/jitesh/3d/data/my_measure/real_measure_ann_data'
    # complete(img_dir=f'{path}/img',
    #                   json_path=f'{path}/measure-only.json',
    #                   show_image=False,
    #                   show_video=True,
    #                   )
    # path = '/home/jitesh/3d/data/coco_data/hc10_400_coco-data'
    # path = '/home/jitesh/3d/data/coco_data/hook_real_training_data4'
    path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data9'
    # path = '/home/jitesh/3d/data/coco_data/hook_sim_real_data'
    # path = '/home/jitesh/3d/data/coco_data/hp15_300_coco-data'
    # # path = '/home/jitesh/3d/data/coco_data/hmk1_100_coco-data/json/hook_cropped_hook.json'
    # path = '/home/jitesh/3d/data/coco_data/hmk1_100_coco-data'
    # path = '/home/jitesh/3d/data/coco_data/h8_500_coco-data'
    complete(img_dir=f'{path}/img',
                    # json_path=f'{path}/json/pole.json',
                    # json_path=f'{path}/json/hook_cropped_hook.json',
                    # json_path=f'{path}/json/hook.json',
                    # json_path=f'{path}/json/hookpole_train.json',
                    json_path=f'{path}/json/hookpole_no-kpt_train.json',
                    # json_path=f'{path}/json/no_seg_hook.json',
                    # json_path=f'{path}/json/new-hook.json',
                    # show_preview=True,
                    show_image=True,
                    # show_video=True,
                    show_seg=True,
                    kpt_idx_offset=0,
                      )
    # complete(img_dir=f'/home/jitesh/Downloads/orig_cam_data/VID_20200107_142213/img',
    #                 # json_path=f'{path}/json/pole.json',
    #                 # json_path=f'{path}/json/hook_cropped_hook.json',
    #                 json_path=f'/home/jitesh/Downloads/orig_cam_data/VID_20200107_142213/hook6.json',
    #                 # json_path=f'{path}/json/no_seg_hook.json',
    #                 # json_path=f'{path}/json/new-hook.json',
    #                 # show_preview=True,
    #                 show_image=True,
    #                 # show_video=True,
    #                 show_seg=True,
    #                 kpt_idx_offset=0,
    #                   )
    # path = '/home/jitesh/3d/data/coco_data/hc10_400_coco-data'
    # complete(img_dir=f'/home/jitesh/sekisui/teamviewer/sampled_data/VID_20200107_142213/img',
    #                 # json_path=f'{path}/json/hook.json',
    #                 json_path=f'/home/jitesh/sekisui/teamviewer/sampled_data/VID_20200107_142213/hooks.json',
    #                 # json_path=f'{path}/json/new-hook.json',
    #                 show_preview=False,
    #                 show_image=True,
    #                 show_video=True,
    #                 show_seg=True,
    #                 kpt_idx_offset=0,
    #                   )
# from annotation_utils.coco.refactored.structs import COCO_Dataset

# def run(path):
#     dataset = COCO_Dataset.load_from_path(
#         json_path=path+'/screw-coco.json',
#         img_dir=path
#     )
#     dataset.images.sort(attr_name='file_name')
#     dataset.save_visualization(
#         save_dir=f'{path}/viz_dump',
#         show_preview=True,
#         kpt_idx_offset=-1,
#         overwrite=True,
#     )
#     # dataset.save_video(
#     #     save_path=path+'.mp4',
#     #     # draw_order=['screw'],
#     #     show_preview=True,
#     #     kpt_idx_offset=-1,
#     #     overwrite=True,
#     #     fps=5
    # )
