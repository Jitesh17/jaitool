import jaitool
import cv2
import numpy as np
import json
from pyjeasy.image_utils import show_image
from jaitool.draw import draw_mask_image, draw_bbox
from jaitool.structures import BBox
import printj

def decodeSeg(mask, segmentations):
    """
    Draw segmentation
    """
    pts = [
        # np.array([int(a) for a in ann]).reshape(-1, 2)
        np
            .array(polygon)
            .reshape(-1, 2)
            .round()
            .astype(int)
        for polygon in segmentations
    ]
    mask = cv2.fillPoly(mask, pts, 255)

    return mask

def decodeRl(mask, rle):
    """
    Run-length encoded object decode
    """
    mask = mask.reshape(-1, order='F')

    last = 0
    val = True
    for count in rle['counts']:
        val = not val
        mask[last:(last+count)] |= val
        last += count

    mask = mask.reshape(rle['size'], order='F')
    return mask

def annotation2binarymask(segmentations, h, w):
    mask = np.zeros((h, w), np.uint8)
    if isinstance(segmentations, list): # segmentation
        mask = decodeSeg(mask, segmentations)
    else:                               # run-length
        mask = decodeRl(mask, segmentations)
    # show_image(mask)
    return mask


def visualize_coco_ann(
    path, json_path, 
    show_image_preview=True,
    write_image=False):
    with open(json_path) as json_file:
        data = json.load(json_file)
    for image in data["images"]:
        img_id = image["id"]
        img = cv2.imread(f'{path}/{image["file_name"].split("/")[-1]}')
        for ann in data["annotations"]:
            if ann["image_id"]==img_id:
                segmentations = ann['segmentation']
                mask = annotation2binarymask(segmentations, image["height"], image["width"])
                # show_image(mask)
                # show_image(img)
                img = draw_mask_image(img, mask, )

                bbox = BBox.from_list(ann["bbox"], 'pminsize')
                img = draw_bbox(img, bbox, )
                for cat in data["categories"]:
                    if ann["category_id"] == cat["id"]:
                        img = cv2.putText(img, cat["name"], (bbox.xmin, bbox.ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA, )
        if show_image_preview:
            show_image(img)
                
if __name__ == "__main__":

    path = '/home/jitesh/prj/belt-hook/data/training_data/8/coco_data'
    
    visualize_coco_ann(path=path, json_path=f'{path}/coco_annotations.json')