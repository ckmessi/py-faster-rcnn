#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'head')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    args = parser.parse_args()
    return args

class HeadDetector(object):
    def __init__(self):
        self.initial_network()

    def detect_img(self, img):
        scores, boxes = im_detect(self.net, img)
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        detect_list = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
            dets = dets[inds, :]
            detect_list.extend(dets)
        return detect_list

    def initial_network(self):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        args = parse_args() 
        prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'head', 'VGG16','faster_rcnn_end2end','test.prototxt')
        iter_num = 50000
        caffemodel = os.path.join(cfg.DATA_DIR, 'vgg16_faster_rcnn_iter_' + str(iter_num) + '.caffemodel')
        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
    
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(caffemodel)
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(self.net, im)

if __name__ == '__main__':
    
    head_detector = HeadDetector()   
    img_path = os.path.join('/home/tangwang/test/test1/', '0000001.jpg')
    img = cv2.imread(img_path)
    detect_list = head_detector.detect_img(img)
    for detect_item in detect_list:
        print(detect_item)

    
