#! /usr/bin/env python3

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

"""
simple, in-line implementation of image FFT, and FFT-shifted filter application, and image return. Image not included

ENV:
PKG  Version
Pillow	7.2.0
argparse	1.4.0
common	0.1.2   
cycler	0.10.0
graphviz	0.14.1
imutils	0.5.3
kiwisolver	1.2.0
matplotlib	3.3.0
numpy	1.19.1
objgraph	3.4.1
opencv-python	4.4.0.40
pip	20.2.2
pyparsing	2.4.7
python-dateutil	2.8.1
scipy	1.5.2
setuptools	39.1.0
six	1.15.0


in: some image

out: 1) image original, 2) fast fourier transform of image (freq domain), 3)
4) rectilinear mask inv transforrm, 5) gaussian blur, 6) GB transformed returned image

"""


def gaussian_transform(org_img, img_in):
    rows, cols = org_img.shape
    gmask = np.zeros((rows, cols, 2), np.float32)
    gaussianr = cv.getGaussianKernel(rows, rows // 5)
    gaussianc = cv.getGaussianKernel(cols, cols // 5)
    gaussian = gaussianr * gaussianc.T * 255
    gmask[:, :, 0] += gaussian
    gmask[:, :, 1] += gaussian
    gshift = img_in * gmask
    g_ishift = np.fft.ifftshift(gshift)
    gauss_back = cv.idft(g_ishift)
    return g_ishift, gauss_back


def visual_scale(name):
    return cv.magnitude(name[:, :, 0], name[:, :, 1])


def build_plot_shape(namearr_len):
    if namearr_len % 2 or namearr_len <= 2:
        namearr_len += 1
    var1 = namearr_len//2
    return var1, 2


def display_img_arr(namearr, imgarr, var1, var2):
    for i in range(0, len(namearr)):
        print(namearr[i])
        plt.subplot(var1, var2, i+1), plt.imshow(imgarr[i], cmap='gray')
        plt.title('{}'.format(namearr[i])), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    try:
        img = cv.imread(sys.argv[1], 0)
    except IndexError:
        img = cv.imread('/Users/seanmoran/Documents/Master/2020/August 2020/ezgif.com-webp-to-jpg(1).jpg', 0)
    name_array, img_array = list(), list()
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    name_array.append('img')
    name_array.append('dft_shift')
    img_array.append(img)
    temp0 = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    temp1 = np.fft.fftshift(temp0)
    # name_array.append('gaussian_mask')
    name_array.append('gaussian_transform')
    # img_array.append(gaussian_transform())
    tempZ, temp2 = gaussian_transform(img, temp1)
    # temp0 = visual_scale(temp0)
    temp1 = 20*np.log(visual_scale(temp1))
    img_array.append(temp1)
    # img_array.append(temp1)
    temp2 = visual_scale(temp2)
    img_array.append(temp2)

    param1, param2 = build_plot_shape(len(name_array))
    display_img_arr(name_array, img_array, param1, param2)
