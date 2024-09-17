import os
import cv2
import numpy as np

def normalize(data):
    data = data/data.max()
    data = data * 65535
    data = data.astype(np.uint16)
    return data

def resize(img, dr, orig):
    space = np.arange(0,img.shape[0], dr)
    img = img[space]
    img = cv2.resize(img, (orig.shape[1], orig.shape[0]))
    return img

if __name__ == '__main__':
    pass