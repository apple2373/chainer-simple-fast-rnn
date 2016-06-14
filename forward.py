#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import dlib
import argparse
import cv2 as cv
import numpy as np
import vgg
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def get_bboxes(orig_img, im_scale, min_size, dedup_boxes=1. / 16):
    rects = []
    dlib.find_candidate_object_locations(orig_img, rects, min_size=min_size)
    rects = [[0, d.left(), d.top(), d.right(), d.bottom()] for d in rects]
    rects = np.asarray(rects, dtype=np.float32)

    # bbox pre-processing
    rects *= im_scale
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(rects * dedup_boxes).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = rects[index, :]

    return rects


def nms(dets, thresh):
    """
    Copyed from python faster RCNN repocitory.
    Source: https://github.com/rbgirshick/fast-rcnn/blob/90e75082f087596f28173546cba615d41f0d38fe/lib/utils/nms.py#L10-L37
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def draw_result(out, im_scale, clss, bbox, rects, nms_thresh, conf):
    out = cv.resize(out, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)
    for cls_id in range(1, 21):
        _cls = clss[:, cls_id][:, np.newaxis]
        _bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]
        dets = np.hstack((_bbx, _cls))
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        orig_rects = rects[keep, 1:]

        inds = np.where(dets[:, -1] >= conf)[0]
        for i in inds:
            _bbox = dets[i, :4]
            x1, y1, x2, y2 = orig_rects[i]
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + 0.5 * width
            center_y = y1 + 0.5 * height

            dx, dy, dw, dh = map(int, _bbox)
            _center_x = dx * width + center_x
            _center_y = dy * height + center_y
            _width = np.exp(dw) * width
            _height = np.exp(dh) * height

            x1 = _center_x - 0.5 * _width
            y1 = _center_y - 0.5 * _height
            x2 = _center_x + 0.5 * _width
            y2 = _center_y + 0.5 * _height

            cv.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 0, 255), 2, cv.LINE_AA)
            ret, baseline = cv.getTextSize(CLASSES[cls_id],
                                           cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
            cv.rectangle(out, (int(x1), int(y2) - ret[1] - baseline),
                         (int(x1) + ret[0], int(y2)), (0, 0, 255), -1)
            cv.putText(out, CLASSES[cls_id], (int(x1), int(y2) - baseline),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1,
                       cv.LINE_AA)

            print CLASSES[cls_id], dets[i, 4]

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--img_fn', type=str, default='sample.jpg')
    parser.add_argument('--out_fn', type=str, default='result.jpg')
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--nms_thresh', type=float, default=0.2)
    parser.add_argument('--conf', type=float, default=0.75)
    args = parser.parse_args()

    print 
    vgg_model=vgg.VGG()
    serializers.load_npz('fast_rcnn_vgg_voc.model', vgg_model)

    #Gpu Setting
    if args.gpu_id >= 0:
        xp = cuda.cupy 
        cuda.get_device(args.gpu_id).use()
        vgg_model.to_gpu()
    else:
        xp=np

    orig_image = cv.imread(args.img_fn)
    img, im_scale = img_preprocessing(orig_image, PIXEL_MEANS)
    orig_rects = get_bboxes(orig_image, im_scale, min_size=args.min_size)

    img = xp.asarray(img)
    rects = xp.asarray(orig_rects)

    x = chainer.Variable(img[xp.newaxis, :, :, :])
    rois = chainer.Variable(rects)

    cls_score, bbox_pred = vgg_model(x,rois)

    clss = cls_score.data
    bbox = bbox_pred.data
    if args.gpu_id >= 0:
        clss = cuda.cupy.asnumpy(cls_score.data)
        bbox = cuda.cupy.asnumpy(bbox_pred.data)

    result = draw_result(orig_image, im_scale, clss, bbox, orig_rects,args.nms_thresh, args.conf)
    cv.imwrite(args.out_fn, result)
