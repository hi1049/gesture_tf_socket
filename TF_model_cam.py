# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import copy

import model_zoo

import argparse

import time
import datetime

from gtts import gTTS
import os
from darknet.python.darknet import *
import requests

with open('category.txt') as f:
    lines = map(lambda x: x.strip(), f.readlines())

ix2label = dict(zip(range(len(list(lines))), lines))

cwd = os.getcwd()
model_path = os.path.join('save_model', 'jester-finetune')

class CAM:
    def __init__(self):
        parser = argparse.ArgumentParser(description="test TF on a single video")
        parser.add_argument('--video_length', type=int, default=50)

        parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

        parser.add_argument('--cam', type=int, default=0) #0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
        parser.add_argument('--frame_width', type=int, default=640) #RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
        parser.add_argument('--frame_height', type=int, default=480) #DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240

        parser.add_argument('--person_detect', type=str, default='point') # None, point, crop
        parser.add_argument('--pad', type=int, default=100)

        parser.add_argument('--motion_detect', type=str, default='diff') # None, diff

        parser.add_argument('--test_mode', type=bool, default=True) # True, False
        parser.add_argument('--debug', type=str, default=False)

        self.args = parser.parse_args()
        self.center_detect = 0
        self.move_detect = False

        # cam setting
        print("connect to cam streaming server ...")
        self.cam_address = 0
        self.cap = cv2.VideoCapture(self.cam_address)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, False)
        print("connect success ...")


        # load yolo v3(tiny) and meta data
        print("load yolo model ...")
        self.yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
        self.meta = load_meta("./darknet/cfg/coco.data")
        print("load success ...")


    def center_crop(self, x):
        if x >= (self.args.frame_width) / 4 and x <= (3 * self.args.frame_width) / 4:
            return 1
        else:
            return 0

    def center_point(self, x, y, pad):
        if (x >= (self.args.frame_width/2-pad) and x <= (self.args.frame_width/2+pad)) and \
            (y >= (self.args.frame_height / 2 - pad*2) and y <= (self.args.frame_height / 2 + pad*2)):
            return 1
        else:
            return 0

    def fire(self, seq, start_ix):
        self.move_detect = True
        if start_ix<0:
            start_ix = len(seq)-1
        return start_ix

    def get_frames(self, num, fps, cam=0):
        seq = []
        cropped_frames = []
        x=-0.5
        y=-0.5
        w=-1
        h=-1

        r = []

        if self.args.motion_detect == 'diff':
            self.move_detect = False
            num_frame = 0

        if self.args.person_detect == 'crop' or self.args.person_detect == 'point':
            self.center_detect = 0

        start_ix = -1

        while True:
            if start_ix > self.args.video_length/2:
                # if late firing -> no op
                return []

            _, frame = self.cap.read()

            if not _:
                try:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.cam_address)
                except:
                    print("trying to reconnect cam server but failed.")
                    return False
                return False

            if not seq:
                r = np_detect(self.yolo, self.meta, frame)

            if r:
                for i in range(0, len(r)):
                    _, _, (_x, _y, _w, _h) = r[i]

                    if self.args.person_detect == 'point':
                        if self.center_point(_x,_y,self.args.pad):
                            if _w*_h>40000:
                                self.center_detect = 1
                                x, y, w, h = _x, _y, _w, _h

            if self.args.test_mode == True:
                #draw person center point
                full_frame = copy.deepcopy(frame)

                if r:
                    cv2.rectangle(full_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                if self.args.person_detect == 'point':
                    cv2.rectangle(full_frame, (int(self.args.frame_width / 2 - self.args.pad), int(self.args.frame_height / 2 - self.args.pad*2)), \
                                  (int(self.args.frame_width / 2 + self.args.pad), int(self.args.frame_height / 2 + self.args.pad*2)), (0, 0, 255), 2)

                ## send to local flask
                # ret, jpeg = cv2.imencode('.jpg', cv2.resize(full_frame,(320,240)))
                # stream = jpeg.tobytes()

                # # for ROI streaming
                # requests.post('http://127.0.0.1:5000/update_stream', data=stream)

                cv2.imshow('roi', full_frame)
                cv2.waitKey(1)

            ymin, ymax, xmin, xmax = int(y-h/2+2), int(y+h/2-2), int(x-w/2+2), int(x+w/2-2)

            if xmin <= 0: xmin = 2
            if ymin <= 0: ymin = 2
            if xmax >= self.args.frame_width: xmax = self.args.frame_width - 2
            if ymax >= self.args.frame_height: ymax = self.args.frame_height - 2

            if ymin > 0 and xmin > 0 and ymin < ymax and xmin < xmax:
                frame = frame[ymin:int(ymin+float(h)*3/5), xmin:xmax] # extend for x-axis

            try:
                resize_frame = cv2.resize(frame, (224, 224))
            except:
                continue

            if self.args.motion_detect == 'diff':
                cropped_frame = copy.deepcopy(frame)
                cropped_frame = cropped_frame[int(cropped_frame.shape[0]*3./5):int(cropped_frame.shape[0])]

                self.cropped_frame = cropped_frame

            if (self.center_detect == 1) or self.args.person_detect == 'None':
                if self.args.motion_detect == 'diff':
                    try:
                        if not self.move_detect:
                            seq.append(resize_frame)
                            cropped_frames.append(cropped_frame)
                            diff_len = 2
                            if len(seq) > diff_len:
                                prev_gray_frame = cv2.cvtColor(cropped_frames[0], cv2.COLOR_BGR2GRAY)
                                gray_frame = cv2.cvtColor(cropped_frames[num_frame], cv2.COLOR_BGR2GRAY)

                                frame_diff = cv2.absdiff(gray_frame, prev_gray_frame)
                                diff_mask = np.array(frame_diff>25, dtype=np.int32)
                                thresh = 4

                                if np.sum(diff_mask) >= thresh:
                                        start_ix = self.fire(seq, start_ix)

                            num_frame += 1
                            if len(seq) > diff_len+1:
                                seq = []
                                cropped_frames = []
                                num_frame = 0

                        else:
                            seq.append(resize_frame)

                    except:
                        print("In move detect try-except --> pass")
                        pass

                elif self.args.motion_detect == 'None':
                    seq.append(resize_frame)

                if len(seq) > num:
                    if cam < 10:
                        print("Too long to compute")
                        self.cap.release()
                        print("reconnect to cam streaming server...")
                        self.cap = cv2.VideoCapture(self.cam_address)
                    return seq

class TFModel:
    def __init__(self):
        print("TFmodel Init start")

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
        self.is_training = tf.placeholder(dtype=tf.bool)

        # build IC3D net
        print("build IC3D model...")
        self.net = model_zoo.I3DNet(inps=self.inputs, n_class=len(ix2label), batch_size=1,
                                    pretrained_model_path=None, final_end_point='SequatialLogits',
                                    dropout_keep_prob=1.0, is_training=self.is_training)

        # logits from IC3D net
        print("setting logits from IC3D net...")
        out, merge_op = self.net(self.inputs)
        self.softmax = tf.nn.softmax(out)
        self.merge_op = merge_op

        self.pred = tf.argmax(self.softmax, axis=-1)

        # gpu config
        print("connect to GPU...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # open session
        print("open session...")
        self.sess = tf.Session(config=config)
        #self.logger = tf.summary.FileWriter('./log', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        print("load model ckpt...")
        ckpt = tf.train.latest_checkpoint(model_path)

        if ckpt:
            # print 'restore from {}...'.format(ckpt)
            print('restore from {}...'.format(ckpt))
            saver.restore(self.sess, ckpt)
        print("model is successfuly builded and loaded...")

    def run_demo_wrapper(self, frames):
        summary, predictions, softmax = self.sess.run([self.merge_op, self.pred, self.softmax], feed_dict={self.inputs: frames, self.is_training: False})

        top_3 = map(lambda x: ix2label[int(x)], np.argsort(-softmax)[0][0][:3])
        mask = map(lambda x: int(ix2label[int(x)]!='Doing other things'), predictions[0])

        # casting
        mask = np.expand_dims(np.expand_dims(mask, axis=0), 2)
        predicted_label = map(lambda x: ix2label[int(x)], predictions[0])

        # apply mask to predictions
        softmax_masked = np.mean(mask*softmax, axis=1)

        from collections import Counter
        freq_predictions = Counter(predicted_label).most_common()[0][0]

        str_res = freq_predictions.strip()
        confidence = softmax_masked.max()

        return str_res, confidence, top_3