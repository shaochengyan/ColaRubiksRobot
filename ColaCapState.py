import cv2
import numpy as np
from threading import Thread
import os
import sys


class ColaCapState:
    def __init__(self):
        self.sx = 10  # width of start
        self.sy = 10  # height of start
        self.width = 100
        self.rotation = 90
        self.filebase = "./cola_store/Data_ColaCapState_{}.npy"
        self.idx_cap = -1

    
    def init2(self, sx, sy, width, rotation, idx_cap):
        self.sx = sx
        self.sy = sy
        self.width = width
        self.rotation = rotation
        self.idx_cap = idx_cap


    def init(self, idx_cap):
        self.idx_cap = idx_cap
        filename = self.filebase.format(idx_cap)
        if os.path.isfile(filename):
            data = np.loadtxt(filename)
            if len(data) == 0:
                return
            self.sx, self.sy, self.width, self.rotation = tuple(data)
    
    
    def get_image_range(self, img):
        width_max = 480 - self.sx
        if self.rotation == 90 or self.rotation == 270:
            width_max = 640 - self.sy
        width = min(self.width, width_max)
        range_x = np.arange(self.sx, self.sx + width).astype(np.int32)
        range_y = np.arange(self.sy, self.sy + width).astype(np.int32)
        return np.meshgrid(range_x, range_y)

    
    def __del__(self):
        if self.idx_cap == -1:
            return
        else:
            fname = self.filebase.format(self.idx_cap)
            np.savetxt(fname, np.asarray([self.sx, self.sy, self.width, self.rotation]))
