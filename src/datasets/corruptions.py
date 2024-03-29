import collections
import os.path
from abc import ABC, abstractmethod

import PIL.Image
import torch
from PIL import Image, ImageFile
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO

from torchvision.transforms import Compose
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates

from ood_detection.config import Config as OodConfig

# https://github.com/hendrycks/robustness/blob/master/ImageNet- C/imagenet_c/imagenet_c/corruptions.py

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    img = img[trim_top:trim_top + h, trim_top:trim_top + h]
    return img


# /////////////// End Distortion Helpers ///////////////
class OodTransform(ABC):
    def __init__(self, severity=1):
        self.severity = severity
        self.sample_shape = (224, 224, 3)

    def __call__(self, sample):
        if isinstance(sample, (ImageFile.ImageFile, Image.Image)):
            sample = np.array(sample)
        assert sample.shape == self.sample_shape, f"Wrong image shape: {sample.shape} instead of {self.sample_shape}"

        return self.transform(sample).astype(np.uint8)

    @abstractmethod
    def transform(self, sample):
        pass


class GaussianNoiseTransform(OodTransform):
    def transform(self, sample):
        c = [.08, .12, 0.18, 0.26, 0.38][self.severity - 1]
        sample = np.array(sample) / 255.
        sample = np.clip(sample + np.random.normal(size=sample.shape, scale=c), 0, 1) * 255
        return sample


class ShotNoiseTransform(OodTransform):
    def transform(self, sample):
        c = [60, 25, 12, 5, 3][self.severity - 1]
        sample = np.array(sample) / 255.
        return np.clip(np.random.poisson(sample * c) / float(c), 0, 1) * 255


class ImpulseNoiseTransform(OodTransform):
    def transform(self, sample):
        c = [.03, .06, .09, 0.17, 0.27][self.severity - 1]
        sample = sk.util.random_noise(np.array(sample) / 255., mode='s&p', amount=c)
        return np.clip(sample, 0, 1) * 255


class SpeckleNoiseTransform(OodTransform):
    def transform(self, sample):
        c = [.15, .2, 0.35, 0.45, 0.6][self.severity - 1]
        sample = np.array(sample) / 255.
        return np.clip(sample + sample * np.random.normal(size=sample.shape, scale=c), 0, 1) * 255


class GaussianBlurTransform(OodTransform):
    def transform(self, sample):
        c = [1, 2, 3, 4, 6][self.severity - 1]
        sample = gaussian(np.array(sample) / 255., sigma=c, channel_axis=2)
        return np.clip(sample, 0, 1) * 255


class GlassBlurTransform(OodTransform):
    def transform(self, sample):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][self.severity - 1]

        sample = np.uint8(gaussian(np.array(sample) / 255., sigma=c[0], channel_axis=True) * 255)
        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(224 - c[1], c[1], -1):
                for w in range(224 - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    sample[h, w], sample[h_prime, w_prime] = sample[h_prime, w_prime], sample[h, w]

        return np.clip(gaussian(sample / 255., sigma=c[0], channel_axis=True), 0, 1) * 255


class DefocusBlurTransform(OodTransform):
    def transform(self, sample):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][self.severity - 1]

        sample = np.array(sample) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(sample[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return np.clip(channels, 0, 1) * 255


class MotionBlurTransform(OodTransform):
    def transform(self, sample):
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][self.severity - 1]
        if isinstance(sample, np.ndarray):
            sample = Image.fromarray(sample)
            to_store = sample.copy()
        with BytesIO() as output:
            sample.save(output, format='PNG')
            output.seek(0)
            sample = MotionImage(blob=output.getvalue())

        # sample.save(output, format='PNG')
        # sample = MotionImage(blob=output.getvalue())
        sample.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        sample = cv2.imdecode(np.frombuffer(sample.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # print(sample[:5, :5, :])
        if sample.shape != self.sample_shape:
            to_store.save(os.path.join(OodConfig.DATAPATH, "fail_image.png"))
            return sample
        # print(sample.max())
        return sample.astype(float)


class ZoomBlurTransform(OodTransform):
    def transform(self, sample):
        if isinstance(sample, (PIL.Image.Image, PIL.ImageFile.ImageFile)):
            sample = np.asarray(sample)
        if sample.shape != self.sample_shape:
            if sample.shape == (3, 224, 224):
                sample = sample.transpose(1, 2, 0)
            else:
                raise IndexError(f"Wrong shape. Expected CHW or HWC, got:{sample.shape}")
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][self.severity - 1]

        sample = (np.array(sample) / 255.).astype(np.float32)

        out = np.zeros_like(sample)
        for zoom_factor in c:
            out += clipped_zoom(sample, zoom_factor)

        sample = (sample + out) / (len(c) + 1)
        return np.clip(sample, 0, 1) * 255


class FogTransform(OodTransform):
    def transform(self, sample):
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][self.severity - 1]

        sample = np.array(sample) / 255.
        max_val = sample.max()
        sample += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        return np.clip(sample * max_val / (max_val + c[0]), 0, 1) * 255


class FrostTransform(OodTransform):
    def __init__(self, severity=1):
        super(FrostTransform, self).__init__(severity)
        self.frostis = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg']
        self.frostis = [os.path.join(OodConfig.DATAPATH, 'frost', frost) for frost in self.frostis]

    def transform(self, sample):
        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][self.severity - 1]
        idx = np.random.randint(5)

        filename = self.frostis[idx]
        frost = cv2.imread(filename)
        # randomly crop and convert to rgb
        x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
        frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

        return np.clip(c[0] * np.array(sample) + c[1] * frost, 0, 255)


class SnowTransform(OodTransform):
    def transform(self, sample):
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][self.severity - 1]

        sample = np.array(sample, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=sample.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        sample = c[6] * sample + (1 - c[6]) * \
                 np.maximum(sample,
                            cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
        return np.clip(sample + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


class SpatterTransfrom(OodTransform):
    def transform(self, sample):

        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][self.severity - 1]
        sample = np.array(sample, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=sample.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
            #     ker -= np.mean(ker)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(sample + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0
            #         m = np.abs(m) ** (1/c[4])

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(sample[..., :1]),
                                    42 / 255. * np.ones_like(sample[..., :1]),
                                    20 / 255. * np.ones_like(sample[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            sample *= (1 - m[..., np.newaxis])

            return np.clip(sample + color, 0, 1) * 255


class ContrastTransform(OodTransform):
    def transform(self, sample):
        c = [0.4, .3, .2, .1, .05][self.severity - 1]

        sample = np.array(sample) / 255.
        means = np.mean(sample, axis=(0, 1), keepdims=True)
        return np.clip((sample - means) * c + means, 0, 1) * 255


class BrightnessTransform(OodTransform):
    def transform(self, sample):
        c = [.1, .2, .3, .4, .5][self.severity - 1]

        sample = np.array(sample) / 255.
        sample = sk.color.rgb2hsv(sample)
        sample[:, :, 2] = np.clip(sample[:, :, 2] + c, 0, 1)
        sample = sk.color.hsv2rgb(sample)

        return np.clip(sample, 0, 1) * 255


class SaturateTransform(OodTransform):
    def transform(self, sample):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][self.severity - 1]

        sample = np.array(sample) / 255.
        sample = sk.color.rgb2hsv(sample)
        sample[:, :, 1] = np.clip(sample[:, :, 1] * c[0] + c[1], 0, 1)
        sample = sk.color.hsv2rgb(sample)

        return np.clip(sample, 0, 1) * 255


class JpegCompressionTransform(OodTransform):
    def transform(self, sample):
        c = [25, 18, 15, 10, 7][self.severity - 1]
        if isinstance(sample, np.ndarray):
            sample = Image.fromarray(sample)
        with BytesIO() as f:
            sample.save(f, format='JPEG', quality=c)
            f.seek(0)
            sample = Image.open(f)
            sample.load()
        sample = np.array(sample).astype(float)
        return sample


class Pixelate:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            sample = PIL.Image.fromarray(sample, mode='RGB')
        return self.transform(sample)

    def transform(self, sample):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][self.severity - 1]

        sample = sample.resize((int(224 * c), int(224 * c)), Image.BOX)
        sample = sample.resize((224, 224), Image.BOX)
        return np.array(sample).astype(np.float32) / 255


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
class ElasticTransform(OodTransform):
    def transform(self, sample):
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][self.severity - 1]

        sample = np.array(sample, dtype=np.float32) / 255.
        shape = sample.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        sample = cv2.warpAffine(sample, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(sample, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def get_corruption_transform(clip_transform, corr, severity):
    assert severity in range(1, 6)
    corruption = corr(severity=severity)
    transform_list = clip_transform.transforms[:-2]
    transform_list.append(corruption)
    transform_list.extend(clip_transform.transforms[-2:])
    transform = Compose(transform_list)
    return transform


def store_corruptions_feature_dict(feature_dict, corruption, dataset, severity):
    datafolder = os.path.join(OodConfig.DATAPATH, 'corruptions', dataset)
    os.makedirs(datafolder, exist_ok=True)
    prefix = "_".join([corruption, str(severity)])
    full_prefix = os.path.join(datafolder, prefix)
    for key, value in feature_dict.items():
        torch.save(value, full_prefix + f"-{key}.pt")


def load_corruptions_feature_dict(keys, corruption, dataset, severity):
    datafolder = os.path.join(OodConfig.DATAPATH, 'corruptions', dataset)
    prefix = "_".join([corruption, str(severity)])
    full_prefix = os.path.join(datafolder, prefix)
    feature_dict = {}
    for key in keys:
        feature_dict[key] = torch.load(full_prefix + f"-{key}.pt", map_location=OodConfig.DEVICE)

    return feature_dict

Corruptions = collections.OrderedDict()
Corruptions['Gaussian Noise'] = GaussianNoiseTransform
Corruptions['Shot Noise'] = ShotNoiseTransform
Corruptions['Impulse Noise'] = ImpulseNoiseTransform
Corruptions['Defocus Blur'] = DefocusBlurTransform
Corruptions['Motion Blur'] = MotionBlurTransform
Corruptions['Zoom Blur'] = ZoomBlurTransform
Corruptions['Snow'] = SnowTransform
Corruptions['Frost'] = FrostTransform
Corruptions['Fog'] = FogTransform
Corruptions['Brightness'] = BrightnessTransform
Corruptions['Contrast'] = ContrastTransform
Corruptions['Elastic'] = ElasticTransform
Corruptions['Pixelate'] = Pixelate
Corruptions['JPEG'] = JpegCompressionTransform
Corruptions['Glass Blur'] = GlassBlurTransform  # speed

# Corruptions['Speckle Noise'] = SpeckleNoiseTransform
# Corruptions['Gaussian Blur'] = GaussianBlurTransform
# Corruptions['Spatter'] = SpatterTransfrom
# Corruptions['Saturate'] = SaturateTransform

THESIS_CORRUPTIONS = collections.OrderedDict()
THESIS_CORRUPTIONS['Gaussian'] = GaussianNoiseTransform
THESIS_CORRUPTIONS['Brightness'] = BrightnessTransform
THESIS_CORRUPTIONS['Snow'] = SnowTransform
