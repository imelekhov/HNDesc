import albumentations.core.composition as AC
import albumentations.augmentations.transforms as AT
from albumentations.pytorch import ToTensorV2


def get_imagenet_mean_std():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std


def test_img_augmentations():
    mean, std = get_imagenet_mean_std()
    augs = AC.Compose([AT.Normalize(mean, std), ToTensorV2()], p=1)
    return augs


def augs_for_visualization():
    augs = AC.Compose([ToTensorV2()], p=1)
    return augs


def train_img_augmentations():
    mean, std = get_imagenet_mean_std()
    augs = AC.Compose(
        [
            AC.OneOf(
                [
                    AT.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.2, p=0.7
                    ),
                    AT.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=30,
                        val_shift_limit=0,
                        p=0.7,
                    ),
                    AT.CLAHE(p=0.5),
                    AT.ToGray(p=0.5),
                    AT.ChannelShuffle(p=0.1),
                ],
                p=0.6,
            ),
            AC.OneOf(
                [AT.GaussianBlur(p=0.5), AT.Blur(p=0.5)],
                p=0.3,
            ),
            AC.OneOf([AT.GaussNoise(p=0.5), AT.GaussNoise(p=0.5)], p=0.1),
            AT.Normalize(mean, std),
            ToTensorV2(),
        ],
        p=1,
    )
    return augs


def get_img_augmentations():
    train_augs = train_img_augmentations()
    test_augs = test_img_augmentations()
    return train_augs, test_augs
