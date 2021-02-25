import argparse
import json
import os

import funcy
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
# parser.add_argument('annotations', metavar='coco_annotations', type=str,
#                     help='Path to COCO annotations file.')
# parser.add_argument('train', type=str, help='Where to store COCO training annotations')
# parser.add_argument('test', type=str, help='Where to store COCO test annotations')
# parser.add_argument('-s', dest='split', type=float, required=True,
#                     help="A percentage of a split; a number in (0, 1)")
# parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
#                     help='Ignore all images without annotations. Keep only these with at least one annotation')

# args = parser.parse_args()


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def split_coco(input_json, json_split_1, json_split_2, split_ratio=0.9, having_annotations: bool = True):
    with open(input_json, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        # images_with_annotations = funcy.lmap(
        #     lambda a: int(a['image_id']), annotations)

        # if having_annotations:
        #     images = funcy.lremove(
        #         lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split_ratio)

        save_coco(json_split_1, info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco(json_split_2, info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(x), json_split_1, len(y), json_split_2))


def main(path, split_ratio=0.9, key='hook_train'):
    input_json = f"{path}/json/{key}.json"
    json_split_1 = f"{path}/json/{key}_train.json"
    json_split_2 = f"{path}/json/{key}_test.json"
    # split_ratio = 0.9
    having_annotations = True
    split_coco(input_json, json_split_1, json_split_2, split_ratio, having_annotations)
    # main(args)
    # python split_coco.py


if __name__ == "__main__":
    paths = []
    input_sub_dir_path = [
        "/home/jitesh/prj/ed_analyser/data/mark_data/1/train",
    ]
    print(input_sub_dir_path)

    for path in input_sub_dir_path:
        if os.path.exists(f'{path}/json'):
            paths.append(f'{path}')
    print(paths)

    for path in paths:
        main(path,
             split_ratio=0.99,
             #  key='hook_mask')
             key='coco')
        #  key='cropped_hook_0.1')
