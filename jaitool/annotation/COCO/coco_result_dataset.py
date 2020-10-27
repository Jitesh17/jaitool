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

class COCO_Result_Dataset:
    def __init__(
        self, 
        image_id, category_id, bbox, score
        # info: COCO_Info, licenses: COCO_License_Handler, images: COCO_Image_Handler,
        # annotations: COCO_Annotation_Handler, categories: COCO_Category_Handler
    ):
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.score = score

    @classmethod
    def buffer(cls, coco_dataset: COCO_Result_Dataset) -> COCO_Result_Dataset:
        """
        A buffer that will return the same value, but mark the object as a COCO_Dataset object.
        This can be useful if your IDE doesn't recognize the type of your coco dataset object.
        
        coco_dataset: The object that you would like to send through the buffer.
        """
        return coco_dataset

    # def copy(self) -> COCO_Result_Dataset:
    #     """
    #     Copies the entirety of the COCO Dataset to a new object, which is located at a different
    #     location in memory.
    #     """
    #     return COCO_Result_Dataset(
    #         info=self.info.copy(),
    #         licenses=self.licenses.copy(),
    #         images=self.images.copy(),
    #         annotations=self.annotations.copy(),
    #         categories=self.categories.copy()
    #     )

    # @classmethod
    # def new(cls, description: str=None) -> COCO_Result_Dataset:
    #     """
    #     Create an empty COCO Dataset.
    #     description (optional): A description of the new dataset that you are creating.
    #     """
    #     coco_info = COCO_Info(description=description) if description is not None else COCO_Info()
    #     return COCO_Dataset(
    #         info=coco_info,
    #         licenses=COCO_License_Handler(),
    #         images=COCO_Image_Handler(),
    #         annotations=COCO_Annotation_Handler(),
    #         categories=COCO_Category_Handler()
    #     )

    def to_dict(self, strict: bool=True) -> dict:
        """
        Converts the COCO_Dataset object to a dictionary format, which is the standard format of COCO datasets.
        """
        # return {
        #     'info': self.info.to_dict(),
        #     'licenses': self.licenses.to_dict_list(),
        #     'images': self.images.to_dict_list(),
        #     'annotations': self.annotations.to_dict_list(strict=strict),
        #     'categories': self.categories.to_dict_list(strict=strict)
        # }
        return {
            'image_id': self.image_id,
            'category_id': self.category_id,
            'bbox': self.bbox,
            'score': self.score,
        }