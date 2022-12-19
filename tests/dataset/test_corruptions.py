import pytest

import PIL
import numpy as np
from datasets.corruptions import Corruptions
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor


def test_int_range_after_corruption():
    corruptions_dict = Corruptions

    test_image_int = PIL.Image.fromarray(np.random.randint(0, 255, (123, 421, 3)), mode='RGB')
    for name, corr in corruptions_dict.items():
        print(name)
        clip_transforms_start = [Resize(224), CenterCrop(224)]
        compose_list = clip_transforms_start
        compose_list.append(corr())
        comp = Compose(compose_list)


        # for correct images in 0 - 255
        image_trans = comp(test_image_int)
        assert isinstance(image_trans, np.ndarray)
        # assert not only values between 0 and 1
        assert not np.allclose(image_trans, .5, atol=.5)
        assert np.allclose(image_trans, 127.5, atol=127.5, rtol=0)
        assert not np.allclose(image_trans, 255)


    test_image_float = PIL.Image.fromarray(np.random.randn(low=0.001, high=0.009, size = (300,300,3)))
