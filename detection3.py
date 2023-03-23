# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:49:14 2023

@author: Onur
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device=0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        fast=True):
    # Load model
    source = str(source)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

    model.model.half() if half else model.model.float()

    cudnn.benchmark = True

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

    if fast == False: #Slow inference,Low memory.
        cap = cv2.VideoCapture(0)
        while True:
            ret, im0 = cap.read()
            im = torch.from_numpy(im0).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            im = im.permute(0,3,1,2)
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):
                if len(det):# per image
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for b in range(len(det)):
                        x,y,w,h = det[:,:4][b]
                        x,y,w,h = int(x),int(y),int(w),int(h)
                        score = det[:,4][b]
                        cls = det[:,-1][b]
                        cls,score = int(cls),float(score)
                        cv2.rectangle(im0,(x,y),(w,h),(255,0,0),2)
                        cv2.putText(im0, str(cls),(w,y) , cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Window", im0)
            ch = cv2.waitKey(1)
            if ch == ord("q"):
                break

    if fast == True: #Fast inference,High memory.

        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half()   # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            for i, det in enumerate(pred):
                im0 = im0s[i].copy()
                if len(det):# per image
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for b in range(len(det)):
                        x,y,w,h = det[:,:4][b]
                        x,y,w,h = int(x),int(y),int(w),int(h)
                        score = det[:,4][b]
                        cls = det[:,-1][b]
                        cls,score = int(cls),float(score)
                        cv2.rectangle(im0,(x,y),(w,h),(255,0,0),2)
                        cv2.putText(im0, str(cls),(w,y) , cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0,0,255), 2, cv2.LINE_AA)

                        cv2.putText(im0, f"FPS:{1/(t3-t2):.0f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Window", im0)
            ch = cv2.waitKey(1)
            if ch == ord("q"):
                break



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)                             