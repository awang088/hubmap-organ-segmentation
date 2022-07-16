import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp, RGBShift, ColorJitter, MotionBlur, RandomFog)
from albumentations.pytorch import ToTensorV2 as ToTensor
import cv2
from get_config import get_config
config = get_config()

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_train():
    transform_train = Compose([
        #Basic
        RandomRotate90(p=1),
        HorizontalFlip(p=0.5),
        
        #Morphology
        ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.5),

        # Noise Transforms
        OneOf(
            [
                GaussNoise(var_limit=(0.0, 50.0), always_apply=True),
                RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.25, always_apply=True),
            ],
            p=0.5,
        ),

        # Blur Transforms
        OneOf(
            [
                MotionBlur(blur_limit=5, always_apply=True),
                GaussianBlur(blur_limit=(3,7), always_apply=True),
            ],
            p=0.5
        ),
        
        #Color Transforms
        OneOf(
            [
                Compose(
                    [
                        RandomGamma(gamma_limit=(80, 120), p=1),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,  # 0.3
                            contrast_limit=0.1,  # 0.3
                            p=1,
                        ),
                    ]
                ),
                RGBShift(
                    r_shift_limit=30,
                    g_shift_limit=0,
                    b_shift_limit=30,
                    p=1,
                ),
                HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=30,
                    val_shift_limit=30,
                    p=1,
                ),
                ColorJitter(
                    brightness=0.3,  # 0.3
                    contrast=0.3,  # 0.3
                    saturation=0.3,
                    hue=0.05,
                    p=1,
                ),
            ],
            p=0.5,
        ),

        OneOf(
            [
                ElasticTransform(
                    alpha=1,
                    sigma=25,
                    alpha_affine=25,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    always_apply=True,
                ),
                GridDistortion(always_apply=True),
                OpticalDistortion(distort_limit=1, shift_limit=0.2, always_apply=True),
            ],
            p=0.5,
        ),
        
        CoarseDropout(max_holes=2, 
                      max_height=config['input_resolution'][0]//4, max_width=config['input_resolution'][1]//4, 
                      min_holes=1,
                      min_height=config['input_resolution'][0]//16, min_width=config['input_resolution'][1]//16, 
                      fill_value=0, mask_fill_value=0, p=0.5),
        
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensor(),
    ])
    return transform_train


def get_transforms_valid():
    transform_valid = Compose([
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensor(),
    ] )
    return transform_valid


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean