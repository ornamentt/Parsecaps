
import torch
import numpy as np
import random
import math
import numpy as np
from icecream import ic
from scipy.ndimage import gaussian_filter, interpolation


class Mixup:
    def __init__(self, alpha=1.0):
        """
        :param alpha: The alpha value for the beta distribution from which lambda is sampled.
        """
        self.alpha = alpha

    def __call__(self, data, target):
        """
        Apply mixup to the given batch.

        :param data: Input data (a batch of images).
        :param target: Labels corresponding to the data.
        :return: Mixed input data and corresponding mixed target.
        """
        # Check if single image; if so, just return the original data and target
        if len(data) < 2:
            return data, target

        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Randomly shuffle data and targets
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_target = target[indices]

        # Mix the data and target
        data = lam * data + (1.0 - lam) * shuffled_data
        target = lam * target + (1.0 - lam) * shuffled_target

        return data, target


class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # Convert mask to tensor
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
 
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
 
    def __call__(self, img):
 
        if random.uniform(0, 1) > self.probability:
            return img
 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
 
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
 
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
 
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
 
        return img


def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images.
    """
    random_state = np.random.RandomState(None)
    shape = image.shape                         # (512, 512)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha          # (512, 512)
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1], dtype=np.float64), np.arange(shape[0], dtype=np.float64))            # (512, 512)
    indices = np.stack([np.reshape(y+dy, (-1,)), np.reshape(x+dx, (-1,))])                                      # (2, 262144)
    
    distorted_image = interpolation.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    return distorted_image.astype(np.float32)

