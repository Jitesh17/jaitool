from __future__ import annotations

from typing import List, Tuple

import numpy as np
import printj
from imgaug.augmentables.bbs import BoundingBox as ImgAugBBox
# from imgaug.augmentables.bbs import BoundingBoxesOnImage as ImgAugBBoxes
from numpy.ma import ceil, floor
from pandas.conftest import cls

from .keypoint import Keypoint2D
from .point import Point2D  # , Point2D_List

# from imgaug.imgaug import flatten


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        # self.xmin = xmin
        # self.ymin = ymin
        # self.xmax = xmax
        # self.ymax = ymax
        self.xmin = min(xmin, xmax)
        self.ymin = min(ymin, ymax)
        self.xmax = max(xmin, xmax)
        self.ymax = max(ymin, ymax)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.area = self.width*self.height
        # self.center = ((self.ymax+self.ymin)//2, (self.xmax + self.xmin)//2)
        self.center = (np.mean([self.xmin, self.xmax], dtype=np.int), np.mean(
            [self.ymin, self.ymax], dtype=np.int))

    def __str__(self):
        class_string = str(type(self)).replace(
            "<class '", "").replace("'>", "").split('.')[-1]
        return f"{class_string}: (xmin, ymin, xmax, ymax)=({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: BBox) -> BBox:
        if isinstance(other, BBox):
            xmin = min(self.xmin, other.xmin)
            ymin = min(self.ymin, other.ymin)
            xmax = max(self.xmax, other.xmax)
            ymax = max(self.ymax, other.ymax)
            return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).to_float()
        elif isinstance(other, (int, float)):
            return BBox(xmin=self.xmin + other, ymin=self.ymin + other, xmax=self.xmax + other, ymax=self.ymax + other)
        elif isinstance(other, Point2D):
            return BBox(xmin=self.xmin + other.x, ymin=self.ymin + other.y, xmax=self.xmax + other.x,
                        ymax=self.ymax + other.y)
        elif isinstance(other, Keypoint2D):
            return BBox(xmin=self.xmin + other.point.x, ymin=self.ymin + other.point.y, xmax=self.xmax + other.point.x,
                        ymax=self.ymax + other.point.y)
        else:
            # printj.red(f'Cannot add {type(other)} to BBox')
            raise TypeError

    def __sub__(self, other) -> BBox:
        if isinstance(other, BBox):
            return BBox(xmin=self.xmin-other.xmin,
                        ymin=self.ymin-other.ymin,
                        xmax=self.xmax-other.xmin,
                        ymax=self.ymax-other.ymin)
            # raise NotImplementedError
        elif isinstance(other, (int, float)):
            return BBox(xmin=self.xmin-other, ymin=self.ymin-other, xmax=self.xmax-other,
                        ymax=self.ymax-other)
        elif isinstance(other, Point2D):
            return BBox(xmin=self.xmin-other.x, ymin=self.ymin-other.y, xmax=self.xmax-other.x, ymax=self.ymax-other.y)
        elif isinstance(other, Keypoint2D):
            return BBox(xmin=self.xmin-other.point.x, ymin=self.ymin-other.point.y, xmax=self.xmax-other.point.x,
                        ymax=self.ymax-other.point.y)
        else:
            printj.red(f'Cannot subtract {type(other)} from BBox')
            raise TypeError

    # def __mul__(self, other) -> BBox:
    #     if isinstance(other, (int, float)):
    #         return BBox(xmin=self.xmin*other, ymin=self.ymin*other, xmax=self.xmax*other, ymax=self.ymax*other)
    #     else:
    #         printj.red(f'Cannot multiply {type(other)} with BBox')
    #         raise TypeError
    #
    #
    # def __truediv__(self, other) -> BBox:
    #     if isinstance(other, (int, float)):
    #         return BBox(xmin=self.xmin/other, ymin=self.ymin/other, xmax=self.xmax/other, ymax=self.ymax/other)
    #     else:
    #         printj.red(f'Cannot divide {type(other)} from BBox')
    #         raise TypeError
    #
    # def __eq__(self, other: BBox) -> bool: if isinstance(other, BBox): return self.xmin == other.xmin and self.ymin
    # == other.ymin and self.xmax == other.xmax and self.ymax == other.ymax else: return NotImplemented

    @classmethod
    def buffer(cls, bbox: cls) -> cls:
        return bbox

    def copy(self) -> cls:
        return BBox(
            xmin=self.xmin,
            ymin=self.ymin,
            xmax=self.xmax,
            ymax=self.ymax
        )

    def to_int(self) -> BBox:
        return BBox(
            xmin=int(self.xmin),
            ymin=int(self.ymin),
            xmax=int(self.xmax),
            ymax=int(self.ymax)
        )

    def to_rounded_int(self, special: bool = False) -> BBox:
        """Rounds BBox object to have integer coordinates.

        Keyword Arguments: special {bool} -- [Round xmin and ymin down using floor, and round xmax and ymax using
        ceil.] (default: {False})

        Returns:
            BBox -- [description]
        """
        if not special:
            return BBox(
                xmin=round(self.xmin),
                ymin=round(self.ymin),
                xmax=round(self.xmax),
                ymax=round(self.ymax)
            )
        else:
            return BBox(
                xmin=floor(self.xmin),
                ymin=floor(self.ymin),
                xmax=ceil(self.xmax),
                ymax=ceil(self.ymax)
            )

    def to_float(self) -> BBox:
        return BBox(
            xmin=float(self.xmin),
            ymin=float(self.ymin),
            xmax=float(self.xmax),
            ymax=float(self.ymax)
        )

    def shape(self) -> list:
        """return [height, width]"""
        return [self.height, self.width]

    # def height(self) -> list:
    #     """return height"""
    #     return self.ymax - self.ymin

    # def width(self) -> list:
    #     """return width"""
    #     return self.xmax - self.xmin

    def to_point_list(self) -> list:
        """
        output_format: [[xmin, ymin], [xmax, ymax]]
        """
        return [[self.xmin, self.ymin], [self.xmax, self.ymax]]

    def to_list(self, output_format: str = 'pminpmax') -> list:
        """
        output_format options:
            'pminpmax': [xmin, ymin, xmax, ymax]
            'pminsize': [xmin, ymin, width, height]
        """
        # check_value(output_format, valid_value_list=['pminpmax', 'pminsize'])
        if output_format == 'pminpmax':
            return [self.xmin, self.ymin, self.xmax, self.ymax]
        elif output_format == 'pminsize':
            return [self.xmin, self.ymin, self.width, self.height]
        else:
            raise Exception

    @classmethod
    def from_list(cls, bbox: list, input_format: str = 'pminpmax') -> BBox:
        """
        input_format options:
            'pminpmax': [xmin, ymin, xmax, ymax]
            'pminsize': [xmin, ymin, width, height]
        """
        # check_value(input_format, valid_value_list=['pminpmax', 'pminsize'])
        if input_format == 'pminpmax':
            if len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
            else:
                xmin, ymin, xmax, ymax = flatten(bbox)

            return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        elif input_format == 'pminsize':
            if len(bbox) == 4:
                xmin, ymin, bbox_w, bbox_h = bbox
            else:
                xmin, ymin, bbox_w, bbox_h = flatten(bbox)
            xmax, ymax = xmin + bbox_w, ymin + bbox_h
            return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        else:
            raise Exception

    def to_imgaug(self) -> ImgAugBBox:
        return ImgAugBBox(x1=self.xmin, y1=self.ymin, x2=self.xmax, y2=self.ymax)

    @classmethod
    def from_imgaug(cls, imgaug_bbox: ImgAugBBox) -> BBox:
        return BBox(
            xmin=imgaug_bbox.x1,
            ymin=imgaug_bbox.y1,
            xmax=imgaug_bbox.x2,
            ymax=imgaug_bbox.y2
        )

    def is_inside_other_bbox(self, other: BBox) -> bool:
        if other.xmin < self.xmin and self.xmax < other.xmax and other.ymin < self.ymin and self.ymax < other.ymax:
            return True
        else:
            return False

    def pad(self, pad_left: int = 0, pad_right: int = 0, pad_top: int = 0, pad_bottom: int = 0, img_width: int = np.inf, img_height: int = np.inf) -> BBox:
        return BBox(xmin=max(self.xmin - pad_left, 0),
                    ymin=max(self.ymin - pad_top, 0),
                    xmax=min(self.xmax + pad_right, img_width),
                    ymax=min(self.ymax + pad_bottom, img_width))

    def coord_in_cropped_frame(self, cropped_frame: BBox, theshold: float = 1) -> BBox:
        result = BBox(xmin=self.xmin-cropped_frame.xmin,
                      ymin=self.ymin-cropped_frame.ymin,
                      xmax=self.xmax-cropped_frame.xmin,
                      ymax=self.ymax-cropped_frame.ymin)
        _thes_w = theshold*self.width
        _thes_h = theshold*self.height
        if self.xmax - _thes_w < cropped_frame.xmin \
            or self.xmin + _thes_w > cropped_frame.xmax \
                or self.ymax - _thes_h < cropped_frame.ymin \
            or self.ymin + _thes_h > cropped_frame.ymax:
            return BBox(0, 0, 0, 0)
        result.xmin = max(0, result.xmin)
        result.ymin = max(0, result.ymin)
        result.xmax = min(cropped_frame.width, result.xmax)
        result.ymax = min(cropped_frame.height, result.ymax)
        return result.to_int()

    @staticmethod
    def iou(bbox1: BBox, bbox2: BBox):
        xA = max(bbox1.xmin, bbox2.xmin)
        yA = max(bbox1.ymin, bbox2.ymin)
        xB = min(bbox1.xmax, bbox2.xmax)
        yB = min(bbox1.ymax, bbox2.ymax)
        interW = xB - xA
        interH = yB - yA
        if interW <= 0 or interH <= 0:
            return 0
        interArea = interW * interH
        unionArea = bbox1.area + bbox2.area - interArea
        if unionArea:
            iou = interArea / unionArea
        else:
            iou = np.inf
        return iou
