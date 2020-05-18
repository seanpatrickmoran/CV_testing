#! /usr/bin/env python3

import numpy as np
import imutils
import cv2, sys


def shape_image(img_path):
    image = cv2.imread(img_path)
    resized = imutils.resize(image, width=750)
    lower = np.array([0, 95, 0])
    upper = np.array([32, 255, 80])
    shapeMask = cv2.inRange(resized, lower, upper)
    # cv2.imwrite("org_mask.png", shapeMask)
    return shapeMask, resized


def parsecontourAreas(in_mask):
    shapeMask,resized = in_mask
    remask = shapeMask.copy()
    cnts,hierarchy = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Mask1", shapeMask)
    areas = list()
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > 15.0:
             continue
        areas.append(area)
        cv2.drawContours(remask, cnts, i, (0, 255, 0), 3)
        cv2.drawContours(resized, cnts, i, (255, 0, 255), 3)
    # cv2.imwrite("Masked.png", remask)
    # cv2.imwrite("Unmasked.png", resized)
    return areas


def load_files(dirname=None):
    import os
    assert os.path.isdir(dirname)
    feed_dir=dirname
    batchlist=list()
    for file in os.listdir(feed_dir):
        if file.endswith(".jpg"):
            batchlist.append("{}/{}".format(feed_dir, file))
    return batchlist


def run_files(batchlist):
    print("Prot ID","gene ID","Filename","Avg Contour Area", "Total Sum", sep='\t')
    # from multiprocessing import Pool
    # from multiprocessing import cpu_count
    assert batchlist
    # path = batchlist.split('/')[0:-1]
    import math
    # N = cpu_count()
    # with Pool(N) as p:
    #     p.map(get.call_and_write, batchlist)
    for i in batchlist:
        name = i.split('/')[-1]
        slave = parsecontourAreas(shape_image(i))
        len_slave = len(slave)
        store_sum = sum(slave)
        if store_sum < 100:
            continue
        pix_mean = store_sum/len_slave
        tup_tabs = name.split('_')
        print(tup_tabs[1],tup_tabs[0],name,pix_mean,store_sum, sep='\t')


if __name__ == '__main__':
    return1 = load_files(sys.argv[1])
    run_files(return1)
