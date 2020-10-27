
import albumentations as A


def get_augmentation(save_path=None, load_path=None):
        if load_path:
            return A.load(load_path)
        else:
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
                # aug_seq1,
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