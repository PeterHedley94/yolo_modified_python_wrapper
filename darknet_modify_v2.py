#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

from ctypes import *
import math
import numpy as np
import cv2
import random
import os,inspect,sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)



def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)

lib = CDLL("/home/peter/Documents/Imperial/Individual_Project/CODE/IMAGE_SEGMENTATION/YOLO/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def compare(image_1,image_2):
    image_shape = image_1.shape
    print("The image1 shape is : " + str(image_1.shape) + " : " + str(image_2.shape) )
    correct_count = 0
    incorrect_count = 0
    count = 0

    for channel in range(image_shape[0]):
        for row in range(image_shape[1]):
            for col in range(image_shape[2]):
                count += 1
                if(image_1[channel,row,col] == image_2[channel,row,col]):
                    if correct_count < 100:
                        print("Correct Match at location " + str([row,col,channel]) + " which is count : " + str(count))
                    correct_count += 1

                else:
                    if incorrect_count == 0:
                        print("Correct Match at location " + str([row,col,channel]) + " which is count : " + str(count))
                    incorrect_count += 1
    print("Correct a=%8d vs not correct b=%8d is d=%f" % (correct_count,incorrect_count,correct_count/(incorrect_count+correct_count)))

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    print(type(im.data))
    print(im.w,im.h,im.c)
    print("image is " + str(img.decode("utf-8", "strict") ))
    x = np.ascontiguousarray(cv2.imread(image.decode("utf-8", "strict")),dtype = np.float32)[...,::-1]
    h,w,c = x.shape
    x = np.swapaxes(x,0,1)
    x = x.reshape((-1),order= "F")/255

    im2 = IMAGE(w = c_int(w),h = c_int(h),c = c_int(c),data = x.ctypes.data_as(POINTER(c_float)))
    #image_c = np.ctypeslib.as_array((c_float * w*h*c).from_address(addressof(im.data.contents)))
    #image_py = np.ctypeslib.as_array((c_float * w*h*c).from_address(addressof(im2.data.contents)))

    compare(image_c,image_py)
    print("Comparison " + str([image_py.shape,im2.w,im2.h,im2.c]) + " vs " + str([image_c.shape,im.w,im.h,im.c]))

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im2)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    parentdir = os.path.dirname(parentdir)
    sys.path.insert(0,parentdir)
    print(os.getcwd())
    print(os.getcwd())
    pt1 = bytes("cfg/yolov3.cfg", 'ascii')
    pt2 = bytes("yolov3.weights", 'ascii')
    data = bytes("cfg/coco.data", 'ascii')
    img = bytes("data/dog.jpg", 'ascii')
    #img = bytes("data/small_image.jpg", 'ascii')
    net = load_net(pt1,pt2, 0)
    meta = load_meta(data)
    r = detect(net, meta, img)
    print(r)
