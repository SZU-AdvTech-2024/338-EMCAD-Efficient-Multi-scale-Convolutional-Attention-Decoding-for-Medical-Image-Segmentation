import random
from tkinter import Image

import cv2
import numpy as np

class RandomGaussianBlurCV:
    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.1, 2.0), p=0.5):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(range(*self.kernel_size_range))
            sigma = random.uniform(*self.sigma_range)
            kernel_size = (kernel_size, kernel_size)
            # OpenCV expects BGR format, ensure correct handling if necessary
            img = np.asarray(img)
            img_blurred = cv2.GaussianBlur(img, kernel_size, sigma)
            return Image.fromarray(img_blurred)
        return img
