
import copy
# aug_visualizer = AugVisualizer(
#         aug_vis_save_path="aug.png",
#         wait=None
#     )
# import os
from typing import Tuple
# from numba import jit
import cv2
# import imgaug as ia
import numpy as np
# from numpy.lib.function_base import iterable
import printj
import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T
# from fvcore.common.file_io import PathManager
# from imgaug import augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBoxesOnImage
# from imgaug.augmentables.kps import KeypointsOnImage
# from imgaug.augmentables.polys import PolygonsOnImage
# from ..dataset_parser import Detectron2_Annotation_Dict
from jaitool.annotation.DET2 import Detectron2_Annotation_Dict
# from common_utils.common_types.bbox import BBox
# from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
# from common_utils.common_types.segmentation import Polygon, Segmentation
# from common_utils.cv_drawing_utils import (cv_simple_image_viewer, draw_bbox,
#                                            draw_keypoints, draw_segmentation)
from jaitool.draw.drawing_utils import (draw_bbox, draw_keypoints,
                                        draw_mask_contour)
from jaitool.structures import (BBox, Keypoint2D_List, Polygon,
                                Segmentation)
# from PIL import Image
# from common_utils.utils import unflatten_list
# from pyjeasy.data_utils import unflatten_list
from pyjeasy.file_utils import dir_contents_path_list_with_extension

# from pasonatron.det2.util.augmentation.aug_utils import do_aug, custom_aug, smart_aug_getter, sometimes, bbox_based_aug_getter
from .aug_visualizer import AugVisualizer

# , aug_visualizer

def flatten(t):
    return [item for sublist in t for item in sublist]


# def rarely(x): return iaa.Sometimes(0.10, x)
# def occasionally(x): return iaa.Sometimes(0.25, x)
# def sometimes(x): return iaa.Sometimes(0.5, x)
# def often(x): return iaa.Sometimes(0.75, x)
# def almost_always(x): return iaa.Sometimes(0.9, x)
# def always(x): return iaa.Sometimes(1.0, x)


class AugmentedLoader(DatasetMapper):
    def __init__(self, cfg, train_type: str = 'kpt', aug=None,
                 aug_vis_save_path: bool = None, show_aug_seg: bool = True,
                 aug_n_rows: int = 3, aug_n_cols: int = 5,
                 aug_save_dims: Tuple[int] = (3 * 500, 5 * 500),
                 bg_dirs: str = None,
                 ):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_type = train_type
        self.aug = aug
        self.aug_vis_save_path = aug_vis_save_path
        self.show_aug_seg = show_aug_seg
        self.aug_visualizer = AugVisualizer(
            aug_vis_save_path=self.aug_vis_save_path,
            n_rows=aug_n_rows,
            n_cols=aug_n_cols,
            save_dims=aug_save_dims,
            wait=None
        )
        self.bg_dirs = bg_dirs
        if self.bg_dirs is not None:
            self.background_images = []
            for bg_dir in self.bg_dirs:
                self.background_images += dir_contents_path_list_with_extension(
                    dirpath=bg_dir,
                    extension=['.jpg', '.jpeg', '.png'])
            self.num_background_images = len(self.background_images)
            self.background_image_dict = dict()

    # def get_bbox_list(self, ann_dict: Detectron2_Annotation_Dict) -> List[BBox]:
    #     return [ann.bbox for ann in ann_dict.annotations]

    def mapper(self, dataset_dict, train_type: str = 'kpt'):
        if train_type != 'kpt':
            for item in dataset_dict["annotations"]:
                if 'keypoints' in item:
                    del item['keypoints']
        # image = cv2.imread(dataset_dict["file_name"], flags=cv2.IMREAD_UNCHANGED)
        
        ann_dict = Detectron2_Annotation_Dict.from_dict(dataset_dict)
        
        if train_type == 'seg':
            seg, bbox, image = self.augment_image_mask(dataset_dict["file_name"])

            for ann in ann_dict.annotations:
                ann.segmentation = Segmentation.from_list(seg)
                ann.bbox = bbox
        if train_type == 'bbox':
            # image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            # utils.check_image_size(dataset_dict, image)
            image = cv2.imread(dataset_dict["file_name"])
            bboxes = []
            class_labels = []
            for ann in dataset_dict["annotations"]:
                bboxes.append(ann['bbox'])
                class_labels.append(ann['category_id'])
            transforms = self.aug(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transforms['image']
            transformed_bboxes = transforms['bboxes']
            transformed_class_labels = transforms['class_labels']
            # print(len(dataset_dict["annotations"]))
            # print(len(transformed_bboxes))
            # print((dataset_dict["annotations"]))
            # print((transformed_bboxes))
            # print()
            for i, ann in enumerate(dataset_dict["annotations"]):
                # print(dataset_dict["annotations"][i]['bbox'])
                # print(transformed_bboxes[i])
                dataset_dict["annotations"][i]['bbox'] = transformed_bboxes[i]
                dataset_dict["annotations"][i]['category_id'] = transformed_class_labels[i]


        # printj.red(ann_dict)
        # bbox_list = [ann.bbox for ann in ann_dict.annotations]
        # if train_type == 'seg':
        #     printj.purple(len(ann_dict.annotations))
        #     for ann in ann_dict.annotations:
        #         seg = ann.segmentation
        #     mask = seg.to_mask()
        #     tranformed = self.aug(mask=mask)
        #     mask = tranformed['mask']
        #     image = tranformed['image']

        # else:
        # if train_type == 'seg':
        # if True:


        """
        # image2 = image.copy()
        # cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), [222, 111, 222], 2)
        # for xi, yi in zip(x, y):
        #     image2 = cv2.circle(image2, (xi, yi), radius=1,
        #                         color=(0, 0, 255), thickness=-1)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # m2 = cv2.hconcat([image2, mask])
        # m0 = cv2.vconcat([m1, m2])
        """
        # from pyjeasy.image_utils import show_image
        # show_image(m0, "a", window_width=1100)
        # cv2.fillPoly(image, pts=contours, color=(11, 255, 11))
        # show_image(image)
        # i=0
        # debug_save = "debug/"
        # import os
        # while os.path.exists(debug_save+f"{i}.png"):
        #     i += 1
        # print(debug_save+f"{i}.png")
        # # cv2.imwrite(debug_save+f"{i}.png", m0)

        # for si, segi in enumerate(seg):
        #     print(i, si, end=" ")
        #     printj.cyan(f"{len(segi)=}")
        #     if len(segi)<8:
        #         printj.red.bold_on_white(f"{len(segi)=}")
        #         print(f"{segi}")
        # else:
        #     tranform = self.aug(image=np.array(image))
        #     image = tranform['image']


        dataset_dict = ann_dict.to_dict()

        # image, transforms = T.apply_transform_gens([], image)

        num_kpts = 0
        annots = []
        for item in dataset_dict["annotations"]:
            if 'keypoints' in item and num_kpts == 0:
                del item['keypoints']
            elif 'keypoints' in item:
                item['keypoints'] = np.array(
                    item['keypoints']).reshape(-1, 3).tolist()
            annots.append(item)
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annots, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(
            instances, by_box=True, by_mask=False)
        return dataset_dict

    def get_bg_img(self, image):
        fw, fh = image.shape[:2]
        bg_idx = np.random.randint(0, self.num_background_images)
        if bg_idx in self.background_image_dict:
            bg_img = self.background_image_dict[bg_idx]
            printj.cyan("old background from hash table")
        else:
            bg_img = cv2.imread(self.background_images[bg_idx])
            bw, bh = bg_img.shape[:2]
            if fw < bw and fh < bh:
                bg_img = bg_img[0:fw, 0:fh]
            else:
                bg_img = cv2.resize(bg_img, (fw, fh))
            self.background_image_dict[bg_idx] = bg_img.copy()
            # printj.yellow.bold_on_white("new background into hash table")
        return bg_img

    def augment_image_mask(self, file_name):
        if self.bg_dirs is None:
        # """
            image = cv2.imread(file_name)
            path_split = file_name.split("/")
            path_split_mask = path_split[:-2] + \
                [f"img_mask", path_split[-1]]
            # printj.cyan(f'{path_split_mask=}')
            mask_path = "/".join(path_split_mask)
            # print(f"{mask_path=}")
            # # mask = utils.read_image(mask_path, format="BGR")
            # mask3 = cv2.imread(mask_path)
            mask = cv2.imread(mask_path, 0)
        # """
        else:
            # """ input rgba and replace bg
            image = cv2.imread(file_name, flags=cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4
            mask = image[:, :, 3]
            image = image[:, :, :3]
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            bg_img = self.get_bg_img(image)
            image = mask3//255*image + (1-mask3//255)*bg_img
            # """
        # m1 = cv2.hconcat([image, mask3])
        # mask = mask3[:, :, 0]
        # image = utils.read_image(dataset_dict["file_name"], format="BGR")
        # img_h, img_w = image.shape[:2]
        # num_pixels = img_w * img_h

        polygon_length = 0
        while polygon_length < 6:
            tranform = self.aug(image=np.array(image), mask=mask)
            image = tranform['image']
            mask = tranform['mask']

            ret, thresh = cv2.threshold(np.array(mask), 127, 255, 0)
            contours = np.asarray(cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
            x = [c[0][0] for c in flatten(contours)]
            y = [c[0][1] for c in flatten(contours)]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
            bbox = BBox(xmin, ymin, xmax, ymax)

            seg = [flatten(flatten(c)) for c in contours]
            polygon_length = min([len(segi) for segi in seg])
        return seg, bbox, image

    def visualize_aug(self, dataset_dict: dict):
        image = dataset_dict["image"].cpu().numpy(
        ).transpose(1, 2, 0).astype('uint8')
        vis_img = image.copy()
        bbox_list = [BBox.from_list(
            vals) for vals in dataset_dict["instances"].gt_boxes.tensor.numpy().tolist()]
        seg_list = [Segmentation([Polygon.from_list(poly.tolist(), demarcation=False) for poly in seg_polys])
                    for seg_polys in dataset_dict["instances"].gt_masks.polygons] if hasattr(dataset_dict['instances'], 'gt_masks') else []
        kpts_list = [Keypoint2D_List.from_numpy(arr, demarcation=True) for arr in dataset_dict["instances"].gt_keypoints.tensor.numpy(
        )] if hasattr(dataset_dict["instances"], 'gt_keypoints') else []
        if self.show_aug_seg:
            for seg in seg_list:
                vis_img = draw_mask_contour(
                    img=vis_img, mask_bool=seg, transparent=True, alpha=0.3)
        for bbox in bbox_list:
            vis_img = draw_bbox(img=vis_img, bbox=bbox)
        for kpts in kpts_list:
            vis_img = draw_keypoints(img=vis_img, keypoints=kpts.to_numpy(
                demarcation=True)[:, :2].tolist(), radius=6)
        self.aug_visualizer.step(vis_img)
    def __call__(self, dataset_dict) -> dict:
        result = copy.deepcopy(dataset_dict)
        result = self.mapper(result, train_type=self.train_type)

        self.visualize_aug(result)
        return result
