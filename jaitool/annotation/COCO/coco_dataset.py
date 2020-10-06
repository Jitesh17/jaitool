from __future__ import annotations
import os
from sys import exit as x
import printj
import pyjeasy.file_utils as f
from pyjeasy.file_utils import get_filename_from_path, get_rootname_from_path, get_all_filenames_of_extension, get_all_filepaths_of_extension, file_exists, make_dir_if_not_exists
from pyjeasy.check_utils import check_required_keys, check_path_exists
import json
from tqdm import tqdm

from typing import List

class COCO_Dataset:
    def __init__(
        self, info: COCO_Info, licenses: COCO_License_Handler, images: COCO_Image_Handler,
        annotations: COCO_Annotation_Handler, categories: COCO_Category_Handler
    ):
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories

    @classmethod
    def buffer(cls, coco_dataset: COCO_Dataset) -> COCO_Dataset:
        """
        A buffer that will return the same value, but mark the object as a COCO_Dataset object.
        This can be useful if your IDE doesn't recognize the type of your coco dataset object.
        
        coco_dataset: The object that you would like to send through the buffer.
        """
        return coco_dataset

    def copy(self) -> COCO_Dataset:
        """
        Copies the entirety of the COCO Dataset to a new object, which is located at a different
        location in memory.
        """
        return COCO_Dataset(
            info=self.info.copy(),
            licenses=self.licenses.copy(),
            images=self.images.copy(),
            annotations=self.annotations.copy(),
            categories=self.categories.copy()
        )

    @classmethod
    def new(cls, description: str=None) -> COCO_Dataset:
        """
        Create an empty COCO Dataset.
        description (optional): A description of the new dataset that you are creating.
        """
        coco_info = COCO_Info(description=description) if description is not None else COCO_Info()
        return COCO_Dataset(
            info=coco_info,
            licenses=COCO_License_Handler(),
            images=COCO_Image_Handler(),
            annotations=COCO_Annotation_Handler(),
            categories=COCO_Category_Handler()
        )

    def to_dict(self, strict: bool=True) -> dict:
        """
        Converts the COCO_Dataset object to a dictionary format, which is the standard format of COCO datasets.
        """
        return {
            'info': self.info.to_dict(),
            'licenses': self.licenses.to_dict_list(),
            'images': self.images.to_dict_list(),
            'annotations': self.annotations.to_dict_list(strict=strict),
            'categories': self.categories.to_dict_list(strict=strict)
        }

    @classmethod
    def from_dict(cls, dataset_dict: dict, strict: bool=True) -> COCO_Dataset:
        """
        Converts a coco dataset dictionary (the standard COCO format) to a COCO_Dataset class object.
        """
        check_required_keys(
            dataset_dict,
            required_keys=[
                'info', 'licenses', 'images',
                'annotations', 'categories'
            ]
        )
        return COCO_Dataset(
            info=COCO_Info.from_dict(dataset_dict['info']),
            licenses=COCO_License_Handler.from_dict_list(dataset_dict['licenses']),
            images=COCO_Image_Handler.from_dict_list(dataset_dict['images']),
            annotations=COCO_Annotation_Handler.from_dict_list(dataset_dict['annotations'], strict=strict),
            categories=COCO_Category_Handler.from_dict_list(dataset_dict['categories'], strict=strict)
        )
    def move_images(
        self, dst_img_dir: str,
        preserve_filenames: bool=False, overwrite_duplicates: bool=False, update_img_paths: bool=True, overwrite: bool=False,
        show_pbar: bool=True
    ):
        """
        Combines all image directories specified in the coco_url of each coco image in self.images
        to a single image directory.

        dst_img_dir: The directory where you would like to save the combined image set.
        preserve_filenames: If False, unique filenames will be generated so as to not create a filename conflict.
        overwrite_duplicates: Only applicable when preserve_filenames=True.
                              In the event that two images with the same filename are moved to dst_img_dir from
                              two different source folders, an error will be raised if overwrite_duplicates=False.
                              If overwrite_duplicates=True, the second copy will be overwrite the first copy.
        update_img_paths: If True, all coco_url paths specified in self.images will be updated to reflect the new
                          combined image directory.
        overwrite: If True, all files in dst_img_dir will be deleted before copying images into the folder.
        """
        used_img_dir_list = []
        for coco_image in self.images:
            used_img_dir = get_dirpath_from_filepath(coco_image.coco_url)
            if used_img_dir not in used_img_dir_list:
                check_dir_exists(used_img_dir)
                used_img_dir_list.append(used_img_dir)

        if len(used_img_dir_list) == 0:
            logger.error(f"Couldn't parse used_img_dir_list.")
            logger.error(f"Are the coco_url paths in your dataset's image dictionary correct?")
            raise Exception

        make_dir_if_not_exists(dst_img_dir)
        if get_dir_contents_len(dst_img_dir) > 0:
            if overwrite:
                delete_all_files_in_dir(dst_img_dir, ask_permission=False)
            else:
                logger.error(f'dst_img_dir={dst_img_dir} is not empty.')
                logger.error('Please use overwrite=True if you would like to delete the contents before proceeding.')
                raise Exception

        pbar = tqdm(total=len(self.images), unit='image(s)') if show_pbar else None
        pbar.set_description(f'Moving Images...')
        for coco_image in self.images:
            if not preserve_filenames:
                img_extension = get_extension_from_path(coco_image.coco_url)
                dst_img_path = get_next_dump_path(
                    dump_dir=dst_img_dir,
                    file_extension=img_extension
                )
                dst_img_path = rel_to_abs_path(dst_img_path)
            else:
                img_filename = get_filename(coco_image.coco_url)
                dst_img_path = f'{dst_img_dir}/{img_filename}'
                if file_exists(dst_img_path) and not overwrite_duplicates:
                    logger.error(f'Failed to copy {coco_image.coco_url} to {dst_img_dir}')
                    logger.error(f'{img_filename} already exists in destination directory.')
                    logger.error(f'Hint: In order to use preserve_filenames=True, all filenames in the dataset must be unique.')
                    logger.error(
                        f'Suggestion: Either update the filenames to be unique or use preserve_filenames=False' + \
                        f' in order to automatically assign the destination filename.'
                    )
                    raise Exception
            copy_file(src_path=coco_image.coco_url, dest_path=dst_img_path, silent=True)
            if update_img_paths:
                coco_image.coco_url = dst_img_path
                coco_image.file_name = get_filename(dst_img_path)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    def save_to_path(self, save_path: str, overwrite: bool=False, strict: bool=True):
        """
        Save this COCO_Dataset object to a json file in the standard COCO format.
        
        save_path: Path of where you would like to save the dataset.
        overwrite: If True, any existing file that exists at save_path will be overwritten.
        """
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict(strict=strict)
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)


    def to_ndds(self) -> NDDS_Frame_Handler:
        raise NotImplementedError

    @classmethod
    def from_ndds(
        cls, ndds_dataset: NDDS_Dataset, categories: COCO_Category_Handler,
        naming_rule: str='type_object_instance_contained', delimiter: str='_',
        license_url: str='https://github.com/cm107/annotation_utils/blob/master/LICENSE',
        license_name: str='MIT License',
        ignore_unspecified_categories: bool=False,
        bbox_area_threshold: float=10,
        default_visibility_threshold: float=0.10,
        visibility_threshold_dict: Dict[str, float]={},
        min_visibile_kpts: int=None,
        color_interval: int=1,
        camera_idx: int=0,
        exclude_invalid_polygons: bool=True,
        allow_unfound_seg: bool=False,
        class_merge_map: Dict[str, str]=None,
        show_pbar: bool=False
    ) -> COCO_Dataset:
        """Creates a COCO_Dataset object from an NDDS_Dataset object.
        The conversion is based on the naming convention of the labels in the NDDS Dataset, so it is important
        to fix the labels in the NDDS_Dataset object before conversion when necessary.
        Note that it is also necessary to define the categories that you want to use in your COCO_Dataset by
        providing a COCO_Category_Handler object. Refer to the COCO_Category_Handler class for usage information.

        Arguments:
            ndds_dataset {NDDS_Dataset} -- [NDDS_Dataset object]
            categories {COCO_Category_Handler} -- [Category Handler that you would like to use for your converted COCO dataset.]

        Keyword Arguments:
            naming_rule {str} -- [
                The naming rule that you would like when converting the NDDS Dataset to a COCO Dataset.
                The category name is separated from the instance name and other strings included in the NDDS annotation label
                based on the naming rule, so it is important that you choose the correct naming rule for your use case.
                Right now only the 'type_object_instance_contained' pattern is available.
            ] (default: {'type_object_instance_contained'})
            delimiter {str} -- [
                The delimiter string that you would like to use when parsing information from the NDDS annotation label.
                Example: If you use delimiter='_', the NDDS annotation label should look something like 'objtype_objname_instancename'
            ] (default: {'_'})
            license_url {str} -- [
                The license url that you would like to associate with all of the images in your converted COCO dataset.
            ] (default: {'https://github.com/cm107/annotation_utils/blob/master/LICENSE'})
            license_name {str} -- [The technical name of your dataset's images' license.] (default: {'MIT License'})
            ignore_unspecified_categories {bool} -- [
                If True, all of the object names in your NDDS dataset (after parsing from the label) that do not match up with
                what is defined in the provided COCO_Category_Handler object will be ignored.
                Otherwise, an error will be thrown if an undefined object name is encountered.
            ] (default: {False})
            bbox_area_threshold {float} -- [
                The threshold that determines when to exclude a segmentation/bbox annotation from the dataset conversion.
                Example: bbox_area_threshold=10 means that any bbox annotation that has an area less than 10 pixels will be excluded.
            ] (default: {10})
            default_visibility_threshold {float} -- [
                The default threshold that determines when to exclude an object that is partially covered by another object.
                This visibility refers to the percentage of the object that is visible to the camera.
                Use visibility_threshold_dict instead to specify the visibility threshold for specific objects.
            ] (default: {0.10})
            visibility_threshold_dict {Dict[str, float]} -- [
                This is a visibility threshold dictionary that can be used to specify the visibility threshold for specific object names.
                If not specified here, unspecified objects will use the default_visibility_threshold.
            ] (default: {{}})
            min_visibile_kpts {int} -- [
                The threshold that determines when to exclude an annotation from a keypoint dataset conversion.
                Example: min_visible_kpts=3 means that any bbox/segmentation annotation that contains less than 3 keypoints will
                         be excluded from the conversion.
            ] (default: {None})
            color_interval {int} -- [
                The color interval that is used when calculating the segmentations from the mask images saved in the NDDS dataset directory.
                The a unique bgr color is assigned to each object instance in the frame based on instance_id, and each mask image represents that relationship.
                Unless there is something wrong with the mask images, the default color_interval=1 should always work.
                Change this value only when debugging.
            ] (default: {1})
            camera_idx {int} -- [
                There is a json file in the NDDS dataset directory called _camera_settings.json.
                camera_idx is the index of the camera in _camera_settings.json that you used when making your NDDS dataset.
                Since there is usually only one camera defined, the default camera_idx=0 should usually work.
            ] (default: {0})
            exclude_invalid_polygons {bool} -- [
                If True, polygons that are composed of less than 3 points will be ignored.
                This can be useful in order to get rid of polygons that result from image artifacts,
                but it can also result in the masks of small objects being ignored unintentionally.
                Change this to False if there are valid small objects being ignored.
            ] (default: {True})
            allow_unfound_seg {bool} -- [
                There may be times when the segmentation can't be parsed from the mask because the object's mask is too thin to create a valid polygon.
                If True, these cases will be skipped without raising an error.
            ] (default: {False})
            class_merge_map {Dict[str, str]} -- [TODO] (default: None)
            show_pbar {bool} -- [Whether or not you would like to display a progress bar in your terminal during conversion.] (default: {False})

        Returns:
            COCO_Dataset -- [The converted COCO Dataset object.]

        Usage:
            ```python
            from logger import logger
            from annotation_utils.ndds.structs import NDDS_Dataset
            from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler

            # Load NDDS Dataset
            ndds_dataset = NDDS_Dataset.load_from_dir(
                json_dir='/path/to/ndds/dir',
                show_pbar=True
            )

            # Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
            for frame in ndds_dataset.frames:
                # Fix Naming Convention
                for ann_obj in frame.ndds_ann.objects:
                    if ann_obj.class_name == 'objname1':
                        obj_type, obj_name, instance_name = 'seg', 'objname', '1'
                        ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
                    elif ann_obj.class_name.startswith('point'):
                        obj_type, obj_name = 'kpt', 'objname'
                        temp = ann_obj.class_name.replace('point', '')
                        instance_name, contained_name = temp[1], temp[0]
                        ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}'
                    elif ...:
                        ...
                    else:
                        logger.error(f'ann_obj.class_name: {ann_obj.class_name}')
                        raise Exception
                
                # Delete Duplicate Objects
                frame.ndds_ann.objects.delete_duplicates(verbose=True, verbose_ref=frame.img_path)

            # Convert To COCO Dataset
            dataset = COCO_Dataset.from_ndds(
                ndds_dataset=ndds_dataset,
                categories=COCO_Category_Handler.load_from_path('/path/to/categories.json'),
                naming_rule='type_object_instance_contained',
                show_pbar=True,
                bbox_area_threshold=50
            )

            dataset.save_to_path('ndds2coco_test.json', overwrite=True)
            dataset.display_preview(show_details=True)
            ```
        """
        # Start constructing COCO Dataset
        dataset = COCO_Dataset.new(description='COCO_Dataset converted from NDDS_Dataset')
        dataset.categories = categories.copy()
        cat_names = [cat.name for cat in dataset.categories]
        cat_keypoints_list = [cat.keypoints for cat in dataset.categories]


        # Get Camera's Settings
        camera_settings = ndds_dataset.camera_config.camera_settings[camera_idx]

        # Add a license to COCO Dataset
        dataset.licenses.append(
            COCO_License(
                url=license_url,
                name=license_name,
                id=0
            )
        )

        if show_pbar:
            frame_pbar = tqdm(total=len(ndds_dataset.frames), unit='frame(s)', leave=True)
            frame_pbar.set_description('Converting Frames')
        for frame in ndds_dataset.frames:
            # Define Camera
            camera = Camera(
                f=[camera_settings.intrinsic_settings.fx, camera_settings.intrinsic_settings.fy],
                c=[camera_settings.intrinsic_settings.cx, camera_settings.intrinsic_settings.cy],
                T=frame.ndds_ann.camera_data.location_worldframe.to_list()
            )
            
            # Load Image Handler
            check_file_exists(frame.img_path)
            img = cv2.imread(frame.img_path)
            img_h, img_w = img.shape[:2]
            if img.shape != camera_settings.captured_image_size.shape():
                logger.error(f'img.shape == {img.shape} != {camera_settings.captured_image_size.shape()} == camera_settings.captured_image_size.shape()')
                logger.error(f'frame.img_path: {frame.img_path}')
                raise Exception
            image_id = len(dataset.images)
            dataset.images.append(
                COCO_Image(
                    license_id=0,
                    file_name=get_filename(frame.img_path),
                    coco_url=frame.img_path,
                    height=img_h,
                    width=img_w,
                    date_captured=get_ctime(frame.img_path),
                    flickr_url=None,
                    id=image_id
                )
            )

            # Load Instance Image
            if class_merge_map is None:
                check_file_exists(frame.is_img_path)
                instance_img = cv2.imread(frame.is_img_path)
                exclude_classes = []
            else:
                instance_img = frame.get_merged_is_img(class_merge_map=class_merge_map)
                exclude_classes = list(class_merge_map.keys())

            organized_handler = frame.to_labeled_obj_handler(naming_rule=naming_rule, delimiter=delimiter, exclude_classes=exclude_classes, show_pbar=show_pbar)
            for labeled_obj in organized_handler:
                specified_category_names = [cat.name for cat in categories]
                if labeled_obj.obj_name not in specified_category_names:
                    if ignore_unspecified_categories:
                        continue
                    else:
                        logger.error(f'Found an NDDS Object name ({labeled_obj.obj_name}) that does not exist in the specified categories.')
                        logger.error(f'specified_category_names: {specified_category_names}')
                        logger.error(f'frame.img_path: {frame.img_path}')
                        logger.error(f'Hint: Use ignore_unspecified_categories=True to bypass this check.')
                        raise Exception
                coco_cat = categories.get_unique_category_from_name(labeled_obj.obj_name)

                partitioned_coco_instances = {}
                for instance in labeled_obj.instances:
                    # Get Segmentation, BBox, and Keypoints
                    if labeled_obj.obj_name in visibility_threshold_dict.keys():
                        if instance.ndds_ann_obj.visibility < visibility_threshold_dict[labeled_obj.obj_name]:
                            continue
                    else:
                        if instance.ndds_ann_obj.visibility < default_visibility_threshold:
                            continue
                    
                    if instance.instance_type == 'seg':
                        seg = instance.get_segmentation(
                            instance_img=instance_img, color_interval=color_interval,
                            is_img_path=frame.is_img_path,
                            exclude_invalid_polygons=exclude_invalid_polygons,
                            allow_unfound_seg=allow_unfound_seg
                        )
                        if len(seg) == 0:
                            continue
                        bbox = seg.to_bbox()
                    elif instance.instance_type == 'bbox':
                        seg = Segmentation()
                        bbox = instance.ndds_ann_obj.bounding_box.copy()
                        bbox = bbox.clip_at_bounds(frame_shape=img.shape[:2])
                        bbox.check_bbox_in_frame(frame_shape=img.shape[:2])
                    elif instance.instance_type == 'kpt':
                        logger.error(f"'kpt' can only be used as a contained instance and not as a container instance")
                        logger.error(f'instance:\n{instance}')
                        raise Exception
                    else:
                        logger.error(f'Invalid instance.instance_type: {instance.instance_type}')
                        logger.error(f'instance:\n{instance}')
                        raise Exception

                    if bbox.area() < bbox_area_threshold:
                        continue

                    kpts_2d, kpts_3d = instance.get_keypoints(kpt_labels=coco_cat.keypoints)
                    visible_kpt_count = sum([kpt.visibility == 2 for kpt in kpts_2d])
                    if min_visibile_kpts is not None and visible_kpt_count < min_visibile_kpts:
                        continue

                    # Construct COCO Annotation
                    coco_ann = COCO_Annotation(
                        id=len(dataset.annotations),
                        category_id=coco_cat.id,
                        image_id=image_id,
                        segmentation=seg,
                        bbox=bbox,
                        area=bbox.area(),
                        keypoints=kpts_2d,
                        num_keypoints=len(kpts_2d),
                        iscrowd=0,
                        keypoints_3d=kpts_3d,
                        camera=camera
                    )
                    if instance.part_num is None:
                        dataset.annotations.append(coco_ann)
                    else:
                        if instance.instance_name not in partitioned_coco_instances:
                            partitioned_coco_instances[instance.instance_name] = [{'coco_ann': coco_ann, 'part_num': instance.part_num}]
                        else:
                            existing_part_numbers = [item['part_num'] for item in partitioned_coco_instances[instance.instance_name]]
                            if instance.part_num not in existing_part_numbers:
                                partitioned_coco_instances[instance.instance_name].append({'coco_ann': coco_ann, 'part_num': instance.part_num})
                            else:
                                logger.error(f'instance.part_num already exists in existing_part_numbers for instance.instance_name={instance.instance_name}')
                                logger.error(f'instance.part_num: {instance.part_num}')
                                logger.error(f'existing_part_numbers: {existing_part_numbers}')
                                logger.error(f"Please check your NDDS annotation json to make sure that you don't have any duplicate part_num!=None instances.")
                                raise Exception

                for instance_name, partitioned_items in partitioned_coco_instances.items():
                    working_seg = Segmentation()
                    working_bbox = None
                    first_coco_ann = partitioned_items[0]['coco_ann']
                    first_coco_ann = COCO_Annotation.buffer(first_coco_ann)
                    for partitioned_item in partitioned_items:
                        coco_ann = partitioned_item['coco_ann']
                        coco_ann = COCO_Annotation.buffer(coco_ann)
                        working_seg = working_seg + coco_ann.segmentation
                        if working_bbox is None:
                            working_bbox = coco_ann.bbox
                        else:
                            working_bbox = working_bbox + coco_ann.bbox
                    dataset.annotations.append(
                        COCO_Annotation(
                            id=len(dataset.annotations),
                            category_id=coco_cat.id,
                            image_id=image_id,
                            segmentation=working_seg,
                            bbox=working_bbox,
                            area=working_bbox.area(),
                            keypoints=first_coco_ann.keypoints,
                            num_keypoints=first_coco_ann.num_keypoints,
                            iscrowd=first_coco_ann.iscrowd,
                            keypoints_3d=first_coco_ann.keypoints_3d,
                            camera=first_coco_ann.camera
                        )
                    )
            if show_pbar:
                frame_pbar.update()
        return dataset