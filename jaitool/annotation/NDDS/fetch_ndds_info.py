from __future__ import annotations

import json
import os
from sys import exit as x
from typing import List

import printj

import numpy as np
import pyjeasy.file_utils as f
from pyjeasy.check_utils import check_required_keys, check_dir_exists, check_type, check_type_from_list
from pyjeasy.file_utils import (dir_exists, file_exists,
                                get_all_filenames_of_extension,
                                get_all_filepaths_of_extension,
                                get_filename_from_path, get_rootname_from_path)
from pyjeasy.base import BasicLoadableObject, BasicLoadableHandler, BasicHandler

class NDDS_Dataset:
    def __init__(self, camera_config, obj_config, frames):
        self.camera_config = camera_config
        self.obj_config = obj_config
        self.frames = frames  # if frames is not None else NDDS_Frame_Handler()

    @classmethod
    def load_from_dir(
        cls, json_dir: str,
        img_dir: str = None, camera_config_path: str = None, obj_config_path: str = None, show_pbar: bool = False
    ) -> NDDS_Dataset:
        """Loads NDDS_Dataset object from a directory path.

        Arguments:
            json_dir {str} -- [Path to directory with all of the NDDS annotation json files.]

        Keyword Arguments:
            img_dir {str} -- [Path to directory with all of the NDDS image files.] (default: json_dir)
            camera_config_path {str} -- [Path to the camera configuration json file.] (default: f'{json_dir}/_camera_settings.json')
            obj_config_path {str} -- [Path to the object configuration json file.] (default: f'{json_dir}/_object_settings.json')
            show_pbar {bool} -- [Show the progress bar.] (default: {False})

        Returns:
            NDDS_Dataset -- [NDDS_Dataset object]
        """
        check_dir_exists(json_dir)
        if img_dir is None:
            img_dir = json_dir
        else:
            check_dir_exists(img_dir)
        camera_config_path = camera_config_path if camera_config_path is not None else f'{json_dir}/_camera_settings.json'
        check_dir_exists(camera_config_path)
        obj_config_path = obj_config_path if obj_config_path is not None else f'{json_dir}/_object_settings.json'
        check_dir_exists(obj_config_path)

        return NDDS_Dataset(
            camera_config=CameraConfig.load_from_path(camera_config_path),
            obj_config=ObjectSettings.load_from_path(obj_config_path),
            frames=NDDS_Frame_Handler.load_from_dir(
                img_dir=img_dir,
                json_dir=json_dir,
                show_pbar=show_pbar
            )
        )


class NDDS_Frame_Handler:
    def __init__(self, frames: List[NDDS_Frame] = None):
        self.frames = frames

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> NDDS_Frame_Handler:
        return NDDS_Frame_Handler([NDDS_Frame.from_dict(item_dict) for item_dict in dict_list])

    @staticmethod
    def load_from_dir(json_dir: str) -> List[NDDS_Frame]:
        all_frames = []
        files_list = f.dir_files_list(dir_path=json_dir)
        json_path_list = get_all_filepaths_of_extension(
            dirpath=json_dir, extension='json', except_condition=get_filename_from_path(json_dir).startswith('_'))
        for json_path in json_path_list:
            img_path = f"{json_path.split('.')[0]}.png"
            is_img_path = f"{json_path.split('.')[0]}.is.png"
            # cs_img_path = f"{json_path.split('.')[0]}.cs.png"
            # depth_img_path = f"{json_path.split('.')[0]}.depth.png"
            if not (file_exists(img_path) and file_exists(is_img_path)):
                printj.red(f"{img_path} or {is_img_path} not exists.")
                raise FileExistsError
            frame = NDDS_Frame(
                img_path=img_path, ndds_ann=NDDS_Annotation.load_from_path(json_path), is_img_path=is_img_path,
                # cs_img_path=cs_img_path, depth_img_path=depth_img_path
            )
            all_frames.append(frame)
        return all_frames

    # @staticmethod
    # def get_camera_settings(camera_settings_path):
    #     camera_settings = json.load(open(camera_settings_path, 'r'))

    # @staticmethod
    # def get_object_settings(object_settings_path):
    #     camera_config = json.load(open(object_settings_path, 'r'))


class NDDS_Annotation_Object:
    def __init__(
        self,
        class_name: str, instance_id: int, visibility: float, location: Point3D, quaternion_xyzw: Quaternion,
        pose_transform: np.ndarray, cuboid_centroid: Point3D, projected_cuboid_centroid: Point2D,
        bounding_box: BBox, cuboid: Cuboid3D, projected_cuboid: Cuboid2D
    ):
        super().__init__()
        self.class_name = class_name
        self.instance_id = instance_id
        if visibility < 0 or visibility > 1:
            printj.red(f'visibility must be between 0 and 1')
            printj.red(f'visibility: {visibility}')
            raise Exception
        self.visibility = visibility
        self.location = location
        self.quaternion_xyzw = quaternion_xyzw
        self.pose_transform = pose_transform
        self.cuboid_centroid = cuboid_centroid
        self.projected_cuboid_centroid = projected_cuboid_centroid
        self.bounding_box = bounding_box
        self.cuboid = cuboid
        self.projected_cuboid = projected_cuboid

    @classmethod
    def from_dict(cls, object_dict: dict) -> NDDS_Annotation_Object:
        check_required_keys(
            object_dict,
            required_keys=[
                'class', 'instance_id', 'visibility',
                'location', 'quaternion_xyzw', 'pose_transform',
                'cuboid_centroid', 'projected_cuboid_centroid', 'bounding_box',
                'cuboid', 'projected_cuboid'
            ]
        )
        check_required_keys(
            object_dict['bounding_box'],
            required_keys=['top_left', 'bottom_right']
        )
        return NDDS_Annotation_Object(
            class_name=object_dict['class'],
            instance_id=object_dict['instance_id'],
            visibility=object_dict['visibility'],
            location=Point3D.from_list(object_dict['location']),
            quaternion_xyzw=Quaternion.from_list(
                object_dict['quaternion_xyzw']),
            pose_transform=np.array(object_dict['pose_transform']),
            cuboid_centroid=Point3D.from_list(object_dict['cuboid_centroid']),
            projected_cuboid_centroid=Point2D.from_list(
                object_dict['projected_cuboid_centroid']),
            bounding_box=BBox.from_list(object_dict['bounding_box']['top_left'][::-1] +
                                        object_dict['bounding_box']['bottom_right'][::-1], input_format='pminpmax'),
            cuboid=Cuboid3D.from_list(object_dict['cuboid'], demarcation=True),
            projected_cuboid=Cuboid2D.from_list(
                object_dict['projected_cuboid'], demarcation=True)
        )


class NDDS_Annotation_Object_Handler:
    def __init__(self, ndds_obj_list: List[NDDS_Annotation_Object] = None):
        self.ndds_obj_list = ndds_obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> NDDS_Annotation_Object_Handler:
        return NDDS_Annotation_Object_Handler(
            ndds_obj_list=[NDDS_Annotation_Object.from_dict(
                obj_dict) for obj_dict in dict_list]
        )


class NDDS_Annotation:
    def __init__(
        self, camera_data: CameraData, objects: NDDS_Annotation_Object_Handler = None
    ):
        self.camera_data = camera_data
        self.objects = objects if objects is not None else NDDS_Annotation_Object_Handler()

    @classmethod
    def from_dict(cls, ann_dict: dict) -> NDDS_Annotation:
        check_required_keys(ann_dict, required_keys=['camera_data', 'objects'])
        return NDDS_Annotation(
            camera_data=CameraData.from_dict(ann_dict['camera_data']),
            objects=NDDS_Annotation_Object_Handler.from_dict_list(
                ann_dict['objects'])
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> NDDS_Annotation:
        ann_dict = json.load(open(json_path, 'r'))
        return NDDS_Annotation(
            camera_data=CameraData.from_dict(ann_dict['camera_data']),
            objects=NDDS_Annotation_Object_Handler.from_dict_list(
                ann_dict['objects'])
        )


class NDDS_Frame:
    def __init__(
        self, img_path: str, ndds_ann: NDDS_Annotation,
        cs_img_path: str = None, depth_img_path: str = None, is_img_path: str = None
    ):
        super().__init__()
        self.img_path = img_path
        self.ndds_ann = ndds_ann
        self.cs_img_path = cs_img_path
        self.depth_img_path = depth_img_path
        self.is_img_path = is_img_path

    @classmethod
    def from_dict(cls, item_dict: dict) -> NDDS_Frame:
        check_required_keys(item_dict, required_keys=['img_path', 'ndds_ann']
                            )
        return NDDS_Frame(
            img_path=item_dict['img_path'],
            ndds_ann=NDDS_Annotation.from_dict(item_dict['ndds_ann']),
            cs_img_path=item_dict['cs_img_path'] if 'cs_img_path' in item_dict else None,
            depth_img_path=item_dict['depth_img_path'] if 'depth_img_path' in item_dict else None,
            is_img_path=item_dict['is_img_path'] if 'is_img_path' in item_dict else None
        )


class CameraData:
    def __init__(self, location_worldframe: Point3D, quaternion_xyzw_worldframe: Quaternion):
        super().__init__()
        self.location_worldframe = location_worldframe
        self.quaternion_xyzw_worldframe = quaternion_xyzw_worldframe

    def to_dict(self) -> dict:
        return {
            'location_worldframe': self.location_worldframe.to_list(),
            'quaternion_xyzw_worldframe': self.quaternion_xyzw_worldframe.to_list()
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraData:
        check_required_keys(
            item_dict,
            required_keys=['location_worldframe', 'quaternion_xyzw_worldframe']
        )
        return CameraData(
            location_worldframe=Point3D.from_list(
                coords=item_dict['location_worldframe']),
            quaternion_xyzw_worldframe=Quaternion.from_list(
                coords=item_dict['quaternion_xyzw_worldframe'])
        )


class ObjectSettings(BasicLoadableObject['ObjectSettings']):
    def __init__(self, exported_object_classes: List[str] = None, exported_objects: ExportedObjectHandler = None):
        super().__init__()
        self.exported_object_classes = exported_object_classes if exported_object_classes is not None else []
        self.exported_objects = exported_objects if exported_objects is not None else ExportedObjectHandler()
        check_type_from_list(self.exported_object_classes,
                             valid_type_list=[str])
        check_type(self.exported_objects, valid_type_list=[
                   ExportedObjectHandler])

    @classmethod
    def from_dict(cls, item_dict: dict) -> ObjectSettings:
        check_required_keys(
            item_dict,
            required_keys=['exported_object_classes', 'exported_objects']
        )
        return ObjectSettings(
            exported_object_classes=item_dict['exported_object_classes'],
            exported_objects=ExportedObjectHandler.from_dict_list(
                item_dict['exported_objects'])
        )


class ExportedObject(BasicLoadableObject['ExportedObject']):
    def __init__(
        self, class_name: str, segmentation_class_id: int, segmentation_instance_id: int,
        fixed_model_transform: np.ndarray, cuboid_dimensions: list
    ):
        super().__init__()
        check_type(class_name, valid_type_list=[str])
        self.class_name = class_name
        check_type(segmentation_class_id, valid_type_list=[int])
        self.segmentation_class_id = segmentation_class_id
        check_type(segmentation_instance_id, valid_type_list=[int])
        self.segmentation_instance_id = segmentation_instance_id
        check_type(fixed_model_transform, valid_type_list=[np.ndarray])
        if fixed_model_transform.shape != (4, 4):
            printj.red(
                f'fixed_model_transform.shape == {fixed_model_transform.shape} != (4, 4)')
            raise Exception
        self.fixed_model_transform = fixed_model_transform
        check_type(cuboid_dimensions, valid_type_list=[list])
        self.cuboid_dimensions = cuboid_dimensions

    def to_dict(self) -> dict:
        return {
            'class': self.class_name,
            'segmentation_class_id': self.segmentation_class_id,
            'segmentation_instance_id': self.segmentation_instance_id,
            'fixed_model_transform': self.fixed_model_transform.tolist(),
            'cuboid_dimensions': self.cuboid_dimensions
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> ExportedObject:
        check_required_keys(
            item_dict,
            required_keys=[
                'class', 'segmentation_class_id',
                'segmentation_instance_id', 'fixed_model_transform',
                'cuboid_dimensions'
            ]
        )
        return ExportedObject(
            class_name=item_dict['class'],
            segmentation_class_id=item_dict['segmentation_class_id'],
            segmentation_instance_id=item_dict['segmentation_instance_id'],
            fixed_model_transform=np.array(item_dict['fixed_model_transform']),
            cuboid_dimensions=item_dict['cuboid_dimensions']
        )


class ExportedObjectHandler(
    BasicLoadableHandler['ExportedObjectHandler', 'ExportedObject'],
    BasicHandler['ExportedObjectHandler', 'ExportedObject']
):
    def __init__(self, exported_obj_list: List[ExportedObject] = None):
        super().__init__(obj_type=ExportedObject, obj_list=exported_obj_list)
        self.exported_obj_list = self.obj_list
        check_type_from_list(self.exported_obj_list,
                             valid_type_list=[ExportedObject])

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> ExportedObjectHandler:
        return ExportedObjectHandler(exported_obj_list=[ExportedObject.from_dict(item_dict) for item_dict in dict_list])


class CameraConfig(BasicLoadableObject['CameraConfig']):
    def __init__(self, camera_settings: CameraSettingsHandler = None):
        super().__init__()
        self.camera_settings = camera_settings if camera_settings is not None else CameraSettingsHandler()
        check_type(self.camera_settings, valid_type_list=[
                   CameraSettingsHandler])

    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraConfig:
        check_required_keys(item_dict, required_keys=['camera_settings'])
        return CameraConfig(
            camera_settings=CameraSettingsHandler.from_dict_list(
                item_dict['camera_settings'])
        )


class CameraSettingsHandler(BasicLoadableHandler['CameraSettingsHandler', 'CameraSettings']):
    def __init__(self, settings_list: List[CameraSettings] = None):
        super().__init__(obj_type=CameraSettings, obj_list=settings_list)
        self.settings_list = self.obj_list
        check_type(self.settings_list, valid_type_list=[CameraSettings])

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> CameraSettingsHandler:
        return CameraSettingsHandler(settings_list=[CameraSettings.from_dict(item_dict) for item_dict in dict_list])


class CameraSettings(BasicLoadableObject['CameraSettings']):
    def __init__(self, name: str, horizontal_fov: int, intrinsic_settings: IntrinsicSettings, captured_image_size: CapturedImageSize):
        super().__init__()
        check_type(name, valid_type_list=[str])
        check_type(horizontal_fov, valid_type_list=[int])
        check_type(intrinsic_settings, valid_type_list=[IntrinsicSettings])
        check_type(captured_image_size, valid_type_list=[CapturedImageSize])
        self.name = name
        self.horizontal_fov = horizontal_fov
        self.intrinsic_settings = intrinsic_settings
        self.captured_image_size = captured_image_size

    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraSettings:
        check_required_keys(
            item_dict,
            required_keys=['name', 'horizontal_fov', 'intrinsic_settings', 'captured_image_size'])
        return CameraSettings(
            name=item_dict['name'],
            horizontal_fov=item_dict['horizontal_fov'],
            intrinsic_settings=IntrinsicSettings.from_dict(
                item_dict['intrinsic_settings']),
            captured_image_size=CapturedImageSize.from_dict(
                item_dict['captured_image_size'])
        )


class IntrinsicSettings(BasicLoadableObject['IntrinsicSettings']):
    def __init__(self, resX: int, resY: int, fx: float, fy: float, cx: float, cy: float, s: int):
        super().__init__()
        check_type(resX, valid_type_list=[int])
        check_type(resY, valid_type_list=[int])
        check_type(fx, valid_type_list=[float, int])
        check_type(fy, valid_type_list=[float, int])
        check_type(cx, valid_type_list=[float, int])
        check_type(cy, valid_type_list=[float, int])
        check_type(s, valid_type_list=[int])
        self.resX = resX
        self.resY = resY
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.s = s


class CapturedImageSize(BasicLoadableObject['CapturedImageSize']):
    def __init__(self, width: int, height: int):
        super().__init__()
        check_type(width, valid_type_list=[int])
        check_type(height, valid_type_list=[int])
        self.width = width
        self.height = height

    def shape(self) -> Tuple[int]:
        """Returns (self.height, self.width)

        Returns:
            Tuple[int] -- [Image shape]
        """
        return (self.height, self.width, 3)
