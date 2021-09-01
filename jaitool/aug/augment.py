
import albumentations as A
import printj, cv2

# from pyjeasy.image_utils import show_image


# def flatten(t): return [item for sublist in t for item in sublist]


# def get_ann(img, mask):
#     ret, thresh = cv2.threshold(mask, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(
#         thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     seg = [flatten(flatten(c)) for c in contours]
#     x = [c[0][0] for c in flatten(contours)]
#     y = [c[0][1] for c in flatten(contours)]
#     xmin = min(x)
#     ymin = min(y)
#     xmax = max(x)
#     ymax = max(y)
#     bbox = [xmin, ymin, xmax, ymax]

#     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [222, 111, 222], 2)
#     for xi, yi in zip(x, y):
#         img = cv2.circle(img, (xi, yi), radius=1,
#                          color=(0, 0, 255), thickness=-1)
#     cv2.fillPoly(img, pts=contours, color=(11, 255, 11))

#     return img, seg, bbox

def get_augmentation(save_path=None, load_path=None):
        if load_path is not None:
            aug_seq=A.load(load_path)

            # printj.red.bold_on_green("00000000000000000000000000000000")
            # img1 = cv2.imread(
            #     "/home/jitesh/prj/belt-hook/data/training_data/8/coco_data_out/rgb_0010.png")
            # mask1 = cv2.imread(
            #     "/home/jitesh/prj/belt-hook/data/training_data/8/coco_data_mask/rgb_0010.png", 0)

            # print(img1.shape)
            # print(mask1.shape)

            # oo=cv2.hconcat([img1, cv2.cvtColor(mask1.copy(),cv2.COLOR_GRAY2RGB)])

            # get_ann(img1.copy(), mask1.copy())
            # for i in range(50):
            #     a = aug_seq(image=img1.copy(), mask=mask1.copy())
            #     img2 = a["image"]
            #     mask2 = a["mask"]
            #     img, seg, bbox = get_ann(img2, mask2)
            #     ooo=cv2.hconcat([img2, cv2.cvtColor(mask2.copy(),cv2.COLOR_GRAY2RGB)])
            #     o=cv2.vconcat([oo, ooo])
            #     show_image(o, "", 1111)
            return aug_seq
        else:
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            return
            aug_seq1 = A.OneOf([
                A.Rotate(limit=(-90, 90), p=1.0),
                A.Flip(p=1.0),
                A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.3, 0.3), 
                                    shift_limit=(-0.05, 0.05), interpolation=3, 
                                    border_mode=3, value=(0, 0, 0), mask_value=None),
            ], p=1.0)
            aug_seq2 = A.OneOf([
                # A.ChannelDropout(always_apply=False, p=1.0, channel_drop_range=(1, 1), fill_value=0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=1.0),
                A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(
                    -0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True)
            ], p=1.0)
            aug_seq3 = A.OneOf([
                A.GaussNoise(always_apply=False, p=1.0, var_limit=(10, 50)),
                A.ISONoise(always_apply=False, p=1.0, intensity=(
                    0.1, 1.0), color_shift=(0.01, 0.3)),
                A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(
                    0.8, 1.6), per_channel=True, elementwise=True),
            ], p=1.0)
            aug_seq4 = A.OneOf([
                A.Equalize(always_apply=False, p=1.0,
                           mode='pil', by_channels=True),
                A.InvertImg(always_apply=False, p=1.0),
                A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                A.RandomFog(always_apply=False, p=1.0, 
                            fog_coef_lower=0.01, fog_coef_upper=0.2, alpha_coef=0.2)
            ], p=1.0)
            aug_seq = A.Compose([
                # A.Resize(self.img_size, self.img_size),
                aug_seq1,
                aug_seq2,
                aug_seq3,
                aug_seq4,
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            # aug_path = '/home/jitesh/prj/classification/test/bolt/aug/aug_seq.json'
            if save_path:
                A.save(aug_seq, save_path)
            # loaded_transform = A.load(aug_path)
            return aug_seq
        
if __name__ == "__main__":
    get_augmentation(
        save_path="/home/jitesh/prj/SekisuiProjects/test/gosar/bolt/training_scripts/aug_seq2.json", 
        load_path=None)