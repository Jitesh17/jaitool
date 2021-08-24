
import copy
# aug_visualizer = AugVisualizer(
#         aug_vis_save_path="aug.png",
#         wait=None
#     )
import os
from typing import List, Tuple

import cv2
import imgaug as ia
import numpy as np
import printj
import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from fvcore.common.file_io import PathManager
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import PolygonsOnImage
# from ..dataset_parser import Detectron2_Annotation_Dict
from jaitool.annotation.DET2 import Detectron2_Annotation_Dict
# from common_utils.common_types.bbox import BBox
# from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
# from common_utils.common_types.segmentation import Polygon, Segmentation
# from common_utils.cv_drawing_utils import (cv_simple_image_viewer, draw_bbox,
#                                            draw_keypoints, draw_segmentation)
from jaitool.draw.drawing_utils import (draw_bbox, draw_keypoints,
                                        draw_mask_contour)
from jaitool.structures import (BBox, Keypoint2D, Keypoint2D_List, Polygon,
                                Segmentation)
from PIL import Image
# from common_utils.utils import unflatten_list
from pyjeasy.data_utils import unflatten_list
from pyjeasy.file_utils import dir_contents_path_list_with_extension

# from pasonatron.det2.util.augmentation.aug_utils import do_aug, custom_aug, smart_aug_getter, sometimes, bbox_based_aug_getter
from .aug_visualizer import AugVisualizer

# , aug_visualizer


def flatten(t):
    return [item for sublist in t for item in sublist]


def rarely(x): return iaa.Sometimes(0.10, x)
def occasionally(x): return iaa.Sometimes(0.25, x)
def sometimes(x): return iaa.Sometimes(0.5, x)
def often(x): return iaa.Sometimes(0.75, x)
def almost_always(x): return iaa.Sometimes(0.9, x)
def always(x): return iaa.Sometimes(1.0, x)


class AugmentedLoader(DatasetMapper):
    def __init__(self, cfg, train_type: str = 'kpt', aug=None,
                 aug_vis_save_path: bool = None, show_aug_seg: bool = True,
                 aug_n_rows: int = 3, aug_n_cols: int = 5,
                 aug_save_dims: Tuple[int] = (3 * 500, 5 * 500),
                 background_images: str = None,
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
        self.background_images = []
        for bg_dir in background_images:
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
        img_path = os.path.join(
            dataset_dict["coco_url"], dataset_dict["file_name"])
        image = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
        fw, fh = image.shape[:2]
        bg_idx = np.random.randint(0, self.num_background_images)
        if bg_idx in self.background_image_dict:
            bg_img = self.background_image_dict[bg_idx]
            # printj.blue.bold_on_white("background from hash table")
        else:
            bg_img = cv2.imread(self.background_images[bg_idx])
            bw, bh = bg_img.shape[:2]
            if fw < bw and fh < bh:
                bg_img = bg_img[0:fw, 0:fh]
            else:
                bg_img = cv2.resize(bg_img, (fw, fh))
            self.background_image_dict[bg_idx] = bg_img

        # path_split = dataset_dict["file_name"].split("/")
        # path_split_mask = path_split[:-2] + \
        #     ["coco_data_mask", path_split[-1]]
        # # printj.cyan(f'{path_split_mask=}')
        # mask_path = "/".join(path_split_mask)
        # # print(f"{mask_path=}")
        # # mask = utils.read_image(mask_path, format="BGR")
        # mask = cv2.imread(mask_path)
        mask = image[:, :, 3]
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = mask3//255*image + (1-mask3//255)*bg_img
        m1 = cv2.hconcat([image, mask3])

        # image = utils.read_image(dataset_dict["file_name"], format="BGR")
        # img_h, img_w = image.shape[:2]
        # num_pixels = img_w * img_h

        ann_dict = Detectron2_Annotation_Dict.from_dict(dataset_dict)
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

        polygon_length = 0
        while polygon_length < 6:
            tranform = self.aug(image=np.array(image), mask=mask)
            image = tranform['image']
            mask = tranform['mask']

            ret, thresh = cv2.threshold(np.array(mask), 127, 255, 0)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x = [c[0][0] for c in flatten(contours)]
            y = [c[0][1] for c in flatten(contours)]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
            bbox = BBox(xmin, ymin, xmax, ymax)

            seg = [flatten(flatten(c)) for c in contours]
            polygon_length = min([len(segi) for segi in seg])

        for ann in ann_dict.annotations:
            ann.segmentation = Segmentation.from_list(seg)
            ann.bbox = bbox

        image2 = image.copy()
        cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), [222, 111, 222], 2)
        for xi, yi in zip(x, y):
            image2 = cv2.circle(image2, (xi, yi), radius=1,
                                color=(0, 0, 255), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        m2 = cv2.hconcat([image2, mask])
        m0 = cv2.vconcat([m1, m2])

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
        '''
        seq_aug_for_no_seg = almost_always(iaa.Sequential(
            [
                # iaa.Rot90(ia.ALL, keep_size=False)
            ]
        ))
        seq_aug_for_seg = sometimes(iaa.Sequential(
            [
                iaa.Rot90(ia.ALL, keep_size=False),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-180, 180),
                    order=[0, 1],
                    # cval=(0, 255),
                    cval=255,
                    mode=ia.ALL
                )
            ]
        ))
        imgaug_kpts = KeypointsOnImage(keypoints=[], shape=image.shape)
        imgaug_bboxes = BoundingBoxesOnImage(
            bounding_boxes=[], shape=image.shape)
        imgaug_polys = PolygonsOnImage(polygons=[], shape=image.shape)

        num_ann = len(ann_dict.annotations)
        num_kpts = None
        seg_len_list = []
        for ann in ann_dict.annotations:
            if num_kpts is None:
                num_kpts = len(ann.keypoints)
            if len(ann.keypoints.to_imgaug(img_shape=image.shape).keypoints) != len(ann_dict.annotations[0].keypoints):
                printj.red(
                    f'len(ann.keypoints.to_imgaug(img_shape=image.shape).keypoints) == {len(ann.keypoints.to_imgaug(img_shape=image.shape).keypoints)} != {len(ann_dict.annotations[0].keypoints)} == len(ann_dict.annotations[0].keypoints)')
                raise Exception
            imgaug_kpts.keypoints.extend(
                ann.keypoints.to_imgaug(img_shape=image.shape).keypoints)
            if ann.bbox.to_imgaug() is None:
                printj.red(f'ann.bbox.to_imgaug() is None')
                printj.red(f'ann.bbox: {ann.bbox}')
                raise Exception
            imgaug_bboxes.bounding_boxes.append(ann.bbox.to_imgaug())
            if ann.segmentation.to_imgaug(img_shape=image.shape).polygons is None:
                printj.red(
                    f'ann.segmentation.to_imgaug(img_shape=image.shape).polygons is None')
                printj.red(f'ann.segmentation:\n{ann.segmentation}')
                raise Exception
            seg_len_list.append(len(ann.segmentation))

            imgaug_polys.polygons.extend(
                ann.segmentation.to_imgaug(img_shape=image.shape).polygons)
        if len(imgaug_polys.polygons) > 0:
            if num_kpts > 0:
                image, imgaug_kpts_aug, imgaug_polys_aug = seq_aug_for_seg(
                    image=image, keypoints=imgaug_kpts, polygons=imgaug_polys)
            else:
                image, imgaug_polys_aug = seq_aug_for_seg(
                    image=image, polygons=imgaug_polys)
                imgaug_kpts_aug = None
            imgaug_bboxes_aug = None
        else:
            if num_kpts > 0:
                image, imgaug_kpts_aug, imgaug_bboxes_aug = seq_aug_for_no_seg(
                    image=image, keypoints=imgaug_kpts, bounding_boxes=imgaug_bboxes)
            else:
                image, imgaug_bboxes_aug = seq_aug_for_no_seg(
                    image=image, bounding_boxes=imgaug_bboxes)
                imgaug_kpts_aug = None
            imgaug_polys_aug = None

        kpts_aug0 = Keypoint2D_List.from_imgaug(
            imgaug_kpts=imgaug_kpts_aug) if num_kpts > 0 else Keypoint2D_List()
        kpts_aug_list = kpts_aug0.to_numpy(demarcation=True)[:, :2].reshape(
            num_ann, num_kpts, 2) if num_kpts > 0 else []
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug]
                         for kpts_aug in kpts_aug_list]
        kpts_aug_list = [Keypoint2D_List.from_list(
            kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]

        if imgaug_polys_aug is not None and imgaug_bboxes_aug is None:
            import printj
            print()
            print()
            printj.yellow(imgaug_polys_aug)
            print(len(imgaug_polys_aug))
            print(imgaug_polys_aug.polygons)
            printj.cyan(imgaug_polys_aug.polygons[0])
            printj.cyan.on_green(Polygon.from_imgaug(imgaug_polys_aug.polygons[0]))
            poly_aug_list = [Polygon.from_imgaug(
                imgaug_polygon) for imgaug_polygon in imgaug_polys_aug.polygons]
            poly_aug_list_list = unflatten_list(
                poly_aug_list, part_sizes=seg_len_list)
            printj.red.on_cyan(poly_aug_list_list)
            seg_aug_list = [Segmentation(poly_aug_list)
                            for poly_aug_list in poly_aug_list_list]
            printj.red(seg_aug_list)
            bbox_aug_list = [seg_aug.to_bbox() for seg_aug in seg_aug_list]
            # Adjust BBoxes when Segmentation BBox does not contain all keypoints
            for i in range(len(bbox_aug_list)):
                kpt_points_aug = [
                    kpt_aug.point for kpt_aug in kpts_aug_list[i]] if num_kpts > 0 else []
                kpt_points_aug_contained = [kpt_point_aug.within(
                    bbox_aug_list[i]) for kpt_point_aug in kpt_points_aug]
                if len(kpt_points_aug) > 0:
                    if not np.any(np.array(kpt_points_aug_contained)):
                        printj.red(
                            f"Keypoints not contained in corresponding bbox.")
                    elif not np.all(np.array(kpt_points_aug_contained)):
                        pass
                    else:
                        break
        elif imgaug_polys_aug is None and imgaug_bboxes_aug is not None:
            bbox_aug_list = [BBox.from_imgaug(
                bbox_aug) for bbox_aug in imgaug_bboxes_aug.bounding_boxes]
            seg_aug_list = [None] * len(bbox_aug_list)
        else:
            printj.red(f'Unexpected error')
            raise Exception
        
        if num_kpts > 0:
            for ann, kpts_aug, bbox_aug, seg_aug in zip(ann_dict.annotations, kpts_aug_list, bbox_aug_list, seg_aug_list):
                ann.keypoints = kpts_aug
                ann.bbox = bbox_aug
                ann.segmentation = seg_aug if seg_aug is not None else Segmentation.from_list([
                ])
        else:
            for ann, bbox_aug, seg_aug in zip(ann_dict.annotations, bbox_aug_list, seg_aug_list):
                ann.keypoints = Keypoint2D_List()
                ann.bbox = bbox_aug
                ann.segmentation = seg_aug if seg_aug is not None else Segmentation.from_list([
                ])
        '''
        num_kpts = 0

        dataset_dict = ann_dict.to_dict()

        image, transforms = T.apply_transform_gens([], image)

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

        # if True:
        #     vis_img = image.copy()
        #     bbox_list = [BBox.from_list(vals) for vals in dataset_dict["instances"].gt_boxes.tensor.numpy().tolist()]
        #     seg_list = [Segmentation([Polygon.from_list(poly.tolist(), demarcation=False) for poly in seg_polys]) for seg_polys in dataset_dict["instances"].gt_masks.polygons]
        #     kpts_list = [Keypoint2D_List.from_numpy(arr, demarcation=True) for arr in dataset_dict["instances"].gt_keypoints.tensor.numpy()] if hasattr(dataset_dict["instances"], 'gt_keypoints') else []
        #     for seg in seg_list:
        #         vis_img = draw_segmentation(img=vis_img, segmentation=seg, transparent=True)
        #     for bbox in bbox_list:
        #         vis_img = draw_bbox(img=vis_img, bbox=bbox)
        #     for kpts in kpts_list:
        #         vis_img = draw_keypoints(img=vis_img, keypoints=kpts.to_numpy(demarcation=True)[:, :2].tolist(), radius=6)
        #     aug_visualizer.step(vis_img)

        return dataset_dict

    # def kpt_dataset_mapper(self, dataset_dict):
    #     return self.mapper(dataset_dict, train_type='kpt')

    # def seg_dataset_mapper(self, dataset_dict):
    #     return self.mapper(dataset_dict, train_type='seg')

    # def bbox_dataset_mapper(self, dataset_dict):
    #     return self.mapper(dataset_dict, train_type='bbox')

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
                # vis_img = draw_segmentation(img=vis_img, segmentation=seg, transparent=True)
                vis_img = draw_mask_contour(
                    img=vis_img, mask_bool=seg, transparent=True, alpha=0.3)
        for bbox in bbox_list:
            vis_img = draw_bbox(img=vis_img, bbox=bbox)
        for kpts in kpts_list:
            vis_img = draw_keypoints(img=vis_img, keypoints=kpts.to_numpy(
                demarcation=True)[:, :2].tolist(), radius=6)
        # aug_visualizer = AugVisualizer(
        #     aug_vis_save_path=self.aug_vis_save_path,
        #     wait=None
        # )
        self.aug_visualizer.step(vis_img)
        # printj.red.bold_on_black("vis _______________________________________")

    def check_image_size(self, dataset_dict, image):
        """
        Raise an error if the image does not match the size specified in the dict.
        """
        if "width" in dataset_dict or "height" in dataset_dict:
            image_wh = (image.shape[1], image.shape[0])
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            if not image_wh == expected_wh:
                # raise SizeMismatchError(
                #     "Mismatched (W,H){}, got {}, expect {}".format(
                #         " for image " + dataset_dict["file_name"]
                #         if "file_name" in dataset_dict
                #         else "",
                #         image_wh,
                #         expected_wh,
                #     )
                # )
                dataset_dict["width"] = image_wh[0]
                dataset_dict["height"] = image_wh[1]

        # To ensure bbox always remap to original image size
        if "width" not in dataset_dict:
            dataset_dict["width"] = image.shape[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.shape[0]

    def det2_default_mapper(self, dataset_dict) -> dict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # USER: Write your own image loading if it's not from a file
        if 'image' in dataset_dict:
            image = dataset_dict["image"].cpu().numpy(
            ).transpose(1, 2, 0).astype('uint8')
        else:
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
        self.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop(
                "sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    def __call__(self, dataset_dict) -> dict:

        result = copy.deepcopy(dataset_dict)
        # print(result)
        result = self.mapper(result, train_type=self.train_type)
        # result = self.det2_default_mapper(result)

        self.visualize_aug(result)
        return result
