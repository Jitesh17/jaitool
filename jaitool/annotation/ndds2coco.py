from __future__ import annotations

import os
from datetime import datetime
from sys import exit as x
from typing import Union

import printj

import pyjeasy.file_utils as f
from jaitool.annotation.COCO import COCO_Dataset
from jaitool.annotation.NDDS import NDDS_Dataset
from pyjeasy.check_utils import check_required_keys, check_path_exists
from pyjeasy.file_utils import (file_exists, get_all_filenames_of_extension,
                                get_all_filepaths_of_extension,
                                get_filename_from_path, make_dir_if_not_exists, delete_dir_if_exists, make_dir, path_exists)


class NDDS2COCO:
    def __init__(self, ndds_dir: str, category: Union[dict, str], output_dir: str): #, write_vis_images: bool, write_vis_video: bool, show_seg: bool, seg_transpare):
        self.ndds_dir = ndds_dir
        self.category = category
        self.output_dir = output_dir
        
    def run(self):
        if not path_exists(self.ndds_dir):
            printj.red(f"Input ndds_dir path does not exist./n{self.ndds_dir}")
        make_dir_if_not_exists(dir_path=self.output_dir)
        coco_data_dir = os.path.join(self.output_dir, f"{get_filename_from_path(self.ndds_dir)}_coco_data")
        delete_dir_if_exists(dir_path=coco_data_dir)
        make_dir(dir_path=coco_data_dir)
        
        ndds_dataset = NDDS_Dataset.load_from_dir(
            json_dir=self.ndds_dir)
        
    # @classmethod    
    # def run(cls):
    #     pass
