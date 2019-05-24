#-*- coding: UTF-8 -*-
import os
import scipy.misc
import numpy as np
from glob import glob

class Avatar:

    def __init__(self):
        self.data_name = 'train'
        self.source_shape = (256, 256, 3)
        self.img_shape = self.source_shape
        self.img_list = self._get_img_list()
        self.batch_size = 6
        self.batch_shape = (self.batch_size, ) + self.img_shape
        self.chunk_size = len(self.img_list) // self.batch_size

    def _get_img_list(self):
        path = os.path.join(os.getcwd(), self.data_name, '*.png')
        return glob(path)

    def _get_img(self, name):
        img = scipy.misc.imread(name).astype(np.float32)
        return np.array(img) / 127.5 - 1.

    def batches(self):
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            name_list = self.img_list[start:end]
            imgs = [self._get_img(name) for name in name_list]
            batches = np.zeros(self.batch_shape)
            batches[::] = imgs
            np.random.shuffle(batches)
            yield batches
            start += self.batch_size
            end += self.batch_size
