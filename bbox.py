import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

StateLetters = ['', 'T', 'C', 'D']

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True,
                            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.class_names = self.yolo3.class_names

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.end_frame = min(int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT)), self.args.end_frame)

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 30, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        bbox = {}
        i = 0
        while self.vdo.grab() and i <= self.end_frame:
            start = time.time()
            bbox[i] = {}
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
            if bbox_xcycwh is not None:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs, scores = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    states = outputs[:, 4]
                    time_since_updates = outputs[:, 5]
                    for j in range(len(outputs)):
                        bbox[i][int(identities[j])] = [int(bbox_xyxy[j][0]), int(bbox_xyxy[j][1]), int(bbox_xyxy[j][2]),
                                                       int(bbox_xyxy[j][3]), StateLetters[states[j]],
                                                       int(time_since_updates[j]), scores[j]]

            if i % 10 == 0:
                print(f"processing frame {i}, t/frame={time.time()-start}")

            i += 1

        import pickle
        import json
        fileName = self.args.VIDEO_PATH.replace('_original', '').rsplit(".", 1)[0] + "_track"
        pickle.dump(bbox, open(fileName+'.pkl', "wb"))
        json.dump(bbox, open(fileName+'.json', "w"), sort_keys=True, indent=4, separators=(',', ': '))

def parse_args():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--end_frame", type=int, default=sys.maxsize)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
