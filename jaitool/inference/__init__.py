"""
D2Inferer
=========

Parameters:
------
weights_path: str 
class_names: List[str] = None, num_classes: int = None,
keypoint_names: List[str] = None, num_keypoints: int = None,
model: str = "mask_rcnn_R_50_FPN_1x",
confidence_threshold: float = 0.5,
size_min: int = None,
size_max: int = None,
key_seg_together: bool = False,
detectron2_dir_path: str = "/home/jitesh/detectron/detectron2"
"""

from .d2_infer import D2Inferer