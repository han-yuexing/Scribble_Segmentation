import numpy as np
import cv2


""" AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and improvements and
does not include any of the search code.
AA and RA Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data - https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection - https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty - https://arxiv.org/abs/1912.02781
Hacked together by / Copyright 2020 Ross Wightman
"""
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _MAX_LEVEL) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * .9
    level = 1.0 + _randomly_negate(level)
    return level,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


def auto_augment_policy_v0(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, cannot find a paper reference.
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],  # This results in black image with Tpu posterize
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0r(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, with variation of Posterize used
    # in Google research implementation (number of bits discarded increases with magnitude)
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('PosterizeIncreasing', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_original(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501
    policy = [
        [('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_originalr(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501 with research posterize variation
    policy = [
        [('PosterizeIncreasing', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeIncreasing', 0.6, 7), ('PosterizeIncreasing', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeIncreasing', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name='v0', hparams=None):
    hparams = hparams or _HPARAMS_DEFAULT
    if name == 'original':
        return auto_augment_policy_original(hparams)
    elif name == 'originalr':
        return auto_augment_policy_originalr(hparams)
    elif name == 'v0':
        return auto_augment_policy_v0(hparams)
    elif name == 'v0r':
        return auto_augment_policy_v0r(hparams)
    else:
        assert False, 'Unknown AA policy (%s)' % name


class AutoAugment:

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img


def auto_augment_transform(config_str, hparams):
    """
    Create a AutoAugment transform
    :param config_str: String defining configuration of auto augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the AutoAugment policy (one of 'v0', 'v0r', 'original', 'originalr').
    The remaining sections, not order sepecific determine
        'mstd' -  float std deviation of magnitude noise applied
    Ex 'original-mstd0.5' results in AutoAugment with original policy, magnitude_std 0.5
    :param hparams: Other hparams (kwargs) for the AutoAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    config = config_str.split('-')
    policy_name = config[0]
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        else:
            assert False, 'Unknown AutoAugment config section'
    aa_policy = auto_augment_policy(policy_name, hparams=hparams)
    return AutoAugment(aa_policy)


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    # 'Rotate',  # 如果使用旋转需要对应标柱图同样进行旋转
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    # 'ShearX',  # 如果使用旋转需要对应标柱图同样进行错切
    # 'ShearY',  # 如果使用旋转需要对应标柱图同样进行错切
    # 'TranslateXRel',  # 如果使用旋转需要对应标柱图同样进行处理(平移?)
    # 'TranslateYRel',  # 如果使用旋转需要对应标柱图同样进行处理(平移?)
    #'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]



# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'Posterize': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'inc':
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


_AUGMIX_TRANSFORMS = [
    'AutoContrast',
    'ColorIncreasing',  # not in paper
    'ContrastIncreasing',  # not in paper
    'BrightnessIncreasing',  # not in paper
    'SharpnessIncreasing',  # not in paper
    'Equalize',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]


def augmix_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _AUGMIX_TRANSFORMS
    return [AugmentOp(
        name, prob=1.0, magnitude=magnitude, hparams=hparams) for name in transforms]


class AugMixAugment:
    """ AugMix Transform
    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, ops, alpha=1., width=3, depth=-1, blended=False):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth
        self.blended = blended  # blended mode is faster but not well tested

    def _calc_blended_weights(self, ws, m):
        ws = ws * m
        cump = 1.
        rws = []
        for w in ws[::-1]:
            alpha = w / cump
            cump *= (1 - alpha)
            rws.append(alpha)
        return np.array(rws[::-1], dtype=np.float32)

    def _apply_blended(self, img, mixing_weights, m):
        # This is my first crack and implementing a slightly faster mixed augmentation. Instead
        # of accumulating the mix for each chain in a Numpy array and then blending with original,
        # it recomputes the blending coefficients and applies one PIL image blend per chain.
        # todo the results appear in the right ballpark but they differ by more than rounding.
        img_orig = img.copy()
        ws = self._calc_blended_weights(mixing_weights, m)
        for w in ws:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img_orig  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            img = Image.blend(img, img_aug, w)
        return img

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def __call__(self, img):
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        if self.blended:
            mixed = self._apply_blended(img, mixing_weights, m)
        else:
            mixed = self._apply_basic(img, mixing_weights, m)
        return mixed


def augment_and_mix_transform(config_str, hparams):
    """ Create AugMix PyTorch transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude (severity) of augmentation mix (default: 3)
        'w' - integer width of augmentation chain (default: 3)
        'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1)
        'b' - integer (bool), blend each branch of chain into end result without a final blend, less CPU (default: 0)
        'mstd' -  float std deviation of magnitude noise applied (default: 0)
    Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2
    :param hparams: Other hparams (kwargs) for the Augmentation transforms
    :return: A PyTorch compatible Transform
    """
    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.
    blended = False
    hparams['magnitude_std'] = float('inf')
    config = config_str.split('-')
    assert config[0] == 'augmix'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'w':
            width = int(val)
        elif key == 'd':
            depth = int(val)
        elif key == 'a':
            alpha = float(val)
        elif key == 'b':
            blended = bool(val)
        else:
            assert False, 'Unknown AugMix config section'
    ops = augmix_ops(magnitude=magnitude, hparams=hparams)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth, blended=blended)


def random_mirror(image, image1, label, activate=True):
    is_flip = np.random.uniform(0, 1.0) >= 0.5
    is_flip = activate and is_flip
    if is_flip:
        image = image[:, ::-1, ...].copy()
        image1 = image1[:, ::-1, ...].copy()
        label = label[:, ::-1, ...].copy()
    return image, image1, label

def flip(image, label):
    image = image[:, ::-1, ...].copy()
    label = label[:, ::-1, ...].copy()
    return image, label

def resize(image, image1, label, ratio, image_method='bilinear', label_method='nearest'):
    """Rescale image and label to the same size by the specified ratio.
    The aspect ratio is remained the same after rescaling.
    Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    ratio: A float/integer indicates the scaling ratio.
    image_method: Image resizing method. bilinear/nearest.
    label_method: Image resizing method. bilinear/nearest.
    Return:
    Two tensors of shape `[new_height, new_width, channels]`.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(ratio * h), int(ratio * w)

    inter_image = (cv2.INTER_LINEAR if image_method == 'bilinear'
                    else cv2.INTER_NEAREST)
    new_image = cv2.resize(image, (new_w, new_h), interpolation=inter_image)
    new_image1 = cv2.resize(image1, (new_w, new_h), interpolation=inter_image)

    inter_label = (cv2.INTER_LINEAR if label_method == 'bilinear'
                    else cv2.INTER_NEAREST)
    new_label = cv2.resize(label, (new_w, new_h), interpolation=inter_label)

    return new_image, new_image1, new_label

def random_scale(image, image1, label):  # 这里的实现和用RandomScaleCrop的实现不大一样,RandomScaleCrop是先根据base_size计算随机比例边长,然后把短边进行缩放;如果把base_size设成513,感觉就是大部分都是放大了,放大和缩小的比例可能有问题
    scale_min = 0.5  # 0.5
    scale_max = 1  # 1.5,感觉2更合理,缩小一倍对应放大一倍
    image_method = 'bilinear'
    label_method = 'nearest'
    assert(scale_max >= scale_min)
    ratio = np.random.uniform(scale_min, scale_max)
    return resize(image, image1, label, ratio, image_method, label_method)

def fixed_scale(image, label, ratio):  # 这里的实现和用RandomScaleCrop的实现不大一样,RandomScaleCrop是先根据base_size计算随机比例边长,然后把短边进行缩放;如果把base_size设成513,感觉就是大部分都是放大了,放大和缩小的比例可能有问题
    image_method = 'bilinear'
    label_method = 'nearest'
    new_image, new_image1, new_label = resize(image, image, label, ratio, image_method, label_method)
    return new_image, new_label

def resize_with_pad(image, size, image_pad_value=0, pad_mode='left_top'):
    """Upscale image by pad to the width and height.
    Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    size: A tuple of integers indicates the target size.
    image_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.
    Return:
    A tensor of shape `[new_height, new_width, channels]`.
    """
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] = h if h > size[0] else size[0]
    new_shape[1] = w if w > size[1] else size[1]
    pad_image = np.zeros(new_shape, dtype=image.dtype)

    if isinstance(image_pad_value, int) or isinstance(image_pad_value, float):
        pad_image.fill(image_pad_value)
    else:
        for ind_ch, val in enumerate(image_pad_value):
            pad_image[:, :, ind_ch].fill(val)

    if pad_mode == 'center':
        s_y = (new_shape[0] - h) // 2
        s_x = (new_shape[1] - w) // 2
        pad_image[s_y:s_y+h, s_x:s_x+w, ...] = image
    elif pad_mode == 'left_top':
        pad_image[:h, :w, ...] = image
    else:
        raise ValueError('Unsupported padding mode')

    return pad_image

def random_crop_with_pad(image, image1, label, crop_size, image_pad_value=0, label_pad_value=255, pad_mode='left_top', return_bbox=False):
    """Randomly crop image and label, and pad them before cropping
    if the size is smaller than `crop_size`.
    Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    crop_size: A tuple of integers indicates the cropped size.
    image_pad_value: An integer indicates the padding value.
    label_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.
    Return:
    Two tensors of shape `[new_height, new_width, channels]`.
    """
    image = resize_with_pad(image, crop_size, image_pad_value, pad_mode)
    image1 = resize_with_pad(image1, crop_size, image_pad_value, pad_mode)
    label = resize_with_pad(label, crop_size, 0, pad_mode)

    h, w = image.shape[:2]
    start_h = int(np.floor(np.random.uniform(0, h - crop_size[0])))
    start_w = int(np.floor(np.random.uniform(0, w - crop_size[1])))
    end_h = start_h + crop_size[0]
    end_w = start_w + crop_size[1]

    crop_image = image[start_h:end_h, start_w:end_w, ...]
    crop_image1 = image1[start_h:end_h, start_w:end_w, ...]
    crop_label = label[start_h:end_h, start_w:end_w, ...]

    if return_bbox:
        bbox = [start_w, start_h, end_w, end_h]
        return crop_image, crop_image1, crop_label, bbox
    else:
        return crop_image, crop_image1, crop_label

def random_crop(image, image1, label):
    img_mean = np.array([0.485, 0.456, 0.406])
    # img_mean = 0
    size = [256, 256]  # 300  # 513和512都看到过,不过513好像对align_corners更友好
    image, image1, label = random_crop_with_pad(image, image1, label, size, img_mean, 255)
    return image, image1, label

def random_rotate(image, image1, label):
    seed = np.random.rand()
    # ----- rotate 4 -----
    if seed >= 0.25 and seed < 0.5:
        time = 1
    elif seed >= 0.5 and seed < 0.75:
        time = 2
    elif seed >= 0.75:
        time = 3
    else:
        time = 0
    # ----- rotate 4 -----
    # ----- rotate 2 -----
    # if seed >= 0.5:
    #     time = 2
    # else:
    #     time = 0
    # ----- rotate 2 -----
    if time != 0:
        image = np.rot90(image, time).copy()
        image1 = np.rot90(image1, time).copy()
        label = np.rot90(label, time).copy()
    return image, image1, label

import torchvision.transforms as transforms

def random_rotate_custom(image, image1, label, frequency, margin_expand):
    mask = np.zeros_like(label)
    # ----- 从opencv到PIL预处理 -----
    image = image * 255
    image1 = image1 * 255
    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2RGB))
    image1 = Image.fromarray(cv2.cvtColor(image1.astype(np.uint8),cv2.COLOR_BGR2RGB))
    label = Image.fromarray(label.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))
    # ----- 从opencv到PIL预处理 -----

    # seed = random.randint(-18,18)
    # angle = seed * 10
    # seed = random.randint(-4,4)
    # angle = seed * 45
    # angle = 0  # no rotate
    # seed = random.randint(-1,1)
    # angle = seed * 180
    # seed = random.randint(-2,2)
    # angle = seed * 90

    # 修改为frequency传入旋转密度
    if frequency != 0:
        seed = random.randint(-frequency, frequency)
        angle = seed * 180 / frequency
    else:
        angle = 0

    if angle != 0:
        image = transforms.functional.rotate(image, angle, expand=margin_expand, fill=0)  # TODO:这里的expand可以保证全图都在里面,但是大小会不一样,所以只能batchsize是1,另外伪标柱图里面的空白位置也要处理下,显存也不大够
        image1 = transforms.functional.rotate(image1, angle, expand=margin_expand, fill=0)
        label = transforms.functional.rotate(label, angle, expand=margin_expand, fill=255)
        mask = transforms.functional.rotate(mask, angle, expand=margin_expand, fill=255)
        # image = transforms.functional.rotate(image, angle, fill=0)
        # image1 = transforms.functional.rotate(image1, angle, fill=0)
        # label = transforms.functional.rotate(label, angle, fill=255)
        # mask = transforms.functional.rotate(mask, angle, fill=255)

    # ----- 从PIL到opencv预处理 -----
    image = np.asarray(image)[:,:,(2,1,0)]
    image1 = np.asarray(image1)[:,:,(2,1,0)]
    label = np.asarray(label)
    mask = np.asarray(mask)
    image = image / 255
    image1 = image1 / 255
    # ----- 从PIL到opencv预处理 -----
    return image, image1, label, mask

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

def fourierAug(img,img1):
    try:
        assert img.shape[2] == 3
        assert img1.shape[2] == 3
    except (IndexError, AssertionError):
        print("The channel of fourier augment images should be 3!")
        exit(0)
    try:
        assert img.shape == img1.shape
    except AssertionError:
        print("Fourier augment images have different shape!")
        exit(0)
    channel_list = []
    img = np.transpose(img,(2,0,1))  # 把通道拉到第一位方便拆分
    img1 = np.transpose(img1,(2,0,1))
    rows, cols = img.shape[1], img.shape[2]
    for img_channel,img_add in zip(img,img1):
        fft_img = np.fft.fft2(img_channel)
        fft_img1 = np.fft.fft2(img_add)
        real = np.real(fft_img)
        real1 = np.real(fft_img1)
        imaginary = np.imag(fft_img)
        imaginary1 = np.imag(fft_img1)
        amplitude = np.sqrt(np.power(real,2.0) + np.power(imaginary,2.0))
        amplitude1 = np.sqrt(np.power(real1,2.0) + np.power(imaginary1,2.0))
        factor = (np.random.rand(1) + 1) / 2  # mix的系数为0.5~1,以原图为主(需要后续实验)
        amplitude = factor * amplitude + (1 - factor) * amplitude1
        phase = np.arctan2(imaginary, real)
        fourier_reverse = amplitude*np.exp(1j*phase)
        transform = np.abs(np.fft.ifft2(fourier_reverse))
        transform = np.clip(transform,0,255).astype(np.uint8)
        # -----visualize-----
        amplitude_norm = cv2.normalize(amplitude,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 这边这个minmax归一化应该是不大合理,不过只是弄成灰度值可视化一下也无所谓了
        amplitude_norm *= 255
        # -----visualize-----
        channel_list.append(transform)
    augImg = np.dstack((channel_list[0], channel_list[1], channel_list[2]))
    return augImg


transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
    
    def __call__(self, img):
        out = img 
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        
        return out

def random_color_jitter(img):
    # https://zhuanlan.zhihu.com/p/84883838?from_voters_page=true
    # _transform_dict = {'brightness':0.1026, 'contrast':0.0935, 'sharpness':0.8386, 'color':0.1592}
    _transform_dict = {'brightness':0.2, 'contrast':0.1, 'sharpness':0, 'color':0.2}  # 飞机(地面):0.2,0.1,0,0.2;
    _color_jitter = ColorJitter(_transform_dict)

    image = img * 255
    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2RGB))

    image = _color_jitter(image)
    image = np.asarray(image)[:,:,(2,1,0)]
    image = image / 255
    return image
