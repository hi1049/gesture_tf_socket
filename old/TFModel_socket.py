# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import copy

import model_zoo

import argparse

import pyrealsense2 as rs

import time
import datetime

from gtts import gTTS
import os
from darknet.python.darknet import *
import requests

with open('category.txt') as f:
    lines = map(lambda x: x.strip(), f.readlines())

ix2label = dict(zip(range(len(lines)), lines))

cwd = os.getcwd()
model_path = os.path.join('save_model', 'jester-finetune')


class RScam:
    def __init__(self):
        parser = argparse.ArgumentParser(description="test TF on a single video")
        parser.add_argument('--video_length', type=int, default=35)

        parser.add_argument('--fps', type=int, default=30)  # 6, 15(1920~424), 30(1280~320), 60

        parser.add_argument('--cam', type=int, default=13) #0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
        parser.add_argument('--frame_width', type=int, default=640) #RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
        parser.add_argument('--frame_height', type=int, default=480) #DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240

        parser.add_argument('--person_detect', type=str, default='point') # None, point, crop
        parser.add_argument('--pad', type=int, default=100)

        parser.add_argument('--motion_detect', type=str, default='diff') # None, diff

        parser.add_argument('--test_mode', type=bool, default=True) # True, False
        parser.add_argument('--debug', type=str, default=False)

        self.args = parser.parse_args()
        self.pipeline = None # for IR
        self.center_detect = 0
        self.move_detect = False

        if self.args.cam < 10:
            self.cap = cv2.VideoCapture(self.args.cam)
            self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

        elif self.args.cam >= 10:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # depth_aligned_to_color  rs.stream.depth
            self.config.enable_stream(rs.stream.color, self.args.frame_width, self.args.frame_height, rs.format.bgr8, self.args.fps)  # set color

            if self.args.cam == 11:
                self.config.enable_stream(rs.stream.depth, self.args.frame_width, self.args.frame_height, rs.format.z16, self.args.fps)  # set depth
                self.config.enable_stream(rs.stream.infrared, 1, self.args.frame_width, self.args.frame_height, rs.format.y8, self.args.fps)
            if self.args.cam == 12:
                self.config.enable_stream(rs.stream.depth, self.args.frame_width, self.args.frame_height, rs.format.z16, self.args.fps)  # set depth
                self.config.enable_stream(rs.stream.infrared, 2, self.args.frame_width, self.args.frame_height, rs.format.y8, self.args.fps)
            if self.args.cam == 13:
                self.config.enable_stream(rs.stream.depth, self.args.frame_width, self.args.frame_height, rs.format.z16, self.args.fps)  # set depth
                self.config.enable_stream(rs.stream.infrared, 1, self.args.frame_width, self.args.frame_height, rs.format.y8, self.args.fps) # right imager
                self.config.enable_stream(rs.stream.infrared, 2, self.args.frame_width, self.args.frame_height, rs.format.y8, self.args.fps) # left imager

            # Start streaming
            self.profile = self.pipeline.start(self.config)
            self.device = self.profile.get_device()

        if self.args.cam == 11 or self.args.cam == 12 or self.args.cam == 13:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

            for _ in range(30):
                self.pipeline.wait_for_frames()

        # load yolo v3(tiny) and meta data
        self.yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
        self.meta = load_meta("./darknet/cfg/coco.data")

        # # load yolo v3(tiny) and meta data
        # self.yolo = load_net("./darknet/cfg/yolov3-tiny-voc.cfg", "./darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
        # self.meta = load_meta("./darknet/cfg/voc.data")

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

    def scan_depth(self, depth_frame, x, y):
        depth_distance = None
        for j, k in ([0, 0], [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]):
            if depth_frame.get_distance(int(x + j), int(y + k)) != 0:
                depth_distance = depth_frame.get_distance(int(x), int(y))
                continue
        if depth_distance != None or depth_distance != 0 or depth_distance != []:
            return depth_distance
        else: return 5

    def fire(self, seq, start_ix):
        self.move_detect = True
        if start_ix<0:
            start_ix = len(seq)-1
        return start_ix

    def get_frames(self, num, fps, cam=0):
        seq = []
        cropped_frames = []
        x=-0.5;y=-0.5;w=-1;h=-1;

        r = []

        if self.args.motion_detect == 'diff':
            self.move_detect = False
            num_frame = 0

        if self.args.person_detect == 'crop' or self.args.person_detect == 'point':
            self.center_detect = 0

        start_ix = -1

        while True:
            try:
                if start_ix > self.args.video_length/2:
                    # if late firing -> no op
                    return []

                if cam < 10:
                    _, frame = self.cap.read()
                elif cam > 10:
                    # Wait for a coherent pair of frames: depth and color
                    all_frames = self.pipeline.wait_for_frames()
                    color_frame = np.asanyarray(
                        all_frames.get_color_frame().get_data()
                    )

                    if cam == 11 or cam == 12 or cam == 13:
                        depth_frame = all_frames.get_depth_frame()
                        if cam == 11:
                            ir_frame1 = all_frames.get_infrared_frame(1)
                            ir_frame2 = ir_frame1
                        elif cam == 12:
                            ir_frame2 = all_frames.get_infrared_frame(2)
                            ir_frame1 = ir_frame2
                        elif cam == 13:
                            ir_frame1 = all_frames.get_infrared_frame(1)
                            ir_frame2 = all_frames.get_infrared_frame(2)

                        if not depth_frame or not ir_frame1 or not ir_frame2:
                            continue

                        # convert infrared images to numpy arrays
                        if cam == 11:
                            ir_image1 = np.asanyarray(ir_frame1.get_data())
                            ir_image1 = cv2.cvtColor(ir_image1, cv2.COLOR_GRAY2BGR)
                            frame = ir_image1

                        elif cam == 12:
                            ir_image2 = np.asanyarray(ir_frame2.get_data())
                            ir_image2 = cv2.cvtColor(ir_image2, cv2.COLOR_GRAY2BGR)
                            frame = ir_image2
                            # frame = color_frame

                        elif cam == 13:
                            ir_image2 = np.asanyarray(ir_frame2.get_data())
                            ir_image2 = cv2.cvtColor(ir_image2, cv2.COLOR_GRAY2BGR)
                            frame = ir_image2

                if not seq:
                    r = np_detect(self.yolo, self.meta, frame)

                if r:
                    for i in range(0, len(r)):
                        _, _, (_x, _y, _w, _h) = r[i]
                        depth = depth_frame.get_distance(int(_x), int(_y))

                        if self.args.person_detect == 'crop':
                            if self.center_crop(_x):
                                if 0.3<depth<3 and _w*_h>40000:
                                    self.center_detect = 1
                                    x, y, w, h = _x, _y, _w, _h

                        elif self.args.person_detect == 'point':
                            if self.center_point(_x,_y,self.args.pad):
                                if 0.3<depth<3 and _w*_h>40000:
                                    self.center_detect = 1
                                    x, y, w, h = _x, _y, _w, _h

                if self.args.test_mode == True:
                    #draw person center point
                    full_frame = copy.deepcopy(frame)
                    if self.args.cam > 10 and r:
                        cv2.circle(full_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.putText(full_frame, ('depth : {0:.3f}'.format(depth_frame.get_distance(int(x), int(y)))), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                            (255, 255, 255, 2))

                    if r:
                        cv2.rectangle(full_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                    (255, 0, 0), 2)
                    if self.args.person_detect == 'point':
                        cv2.rectangle(full_frame, (int(self.args.frame_width / 2 - self.args.pad), int(self.args.frame_height / 2 - self.args.pad*2)), \
                                    (int(self.args.frame_width / 2 + self.args.pad), int(self.args.frame_height / 2 + self.args.pad*2)),
                                    (0, 0, 255), 2)

                    ret, jpeg = cv2.imencode('.jpg', cv2.resize(full_frame,(320,240)))
                    stream = jpeg.tobytes()

                    # for ROI streaming
                    requests.post('http://127.0.0.1:5000/update_stream', data=stream)

                    # cv2.imshow('roi', full_frame)
                    # cv2.waitKey(1)

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
                    # cropped_frame = cropped_frame[int(cropped_frame.shape[0]*3./5):int(cropped_frame.shape[0]),int(cropped_frame.shape[1]*1./5):int(cropped_frame.shape[1]*4./5)]
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

                                    #gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                                    frame_diff = cv2.absdiff(gray_frame, prev_gray_frame)
                                    diff_mask = np.array(frame_diff>25, dtype=np.int32)

                                    if self.args.cam >= 10:
                                        #thresh1 = 5-2*abs(math.log10(60*depth_frame.get_distance(int(x), int((y)))+30))
                                        #thresh2 = 1.0#*np.exp(-0.2*depth_frame.get_distance(int(x), int((y))))
                                        #if thresh1 < 0: thresh1 = 0
                                        crop_area = reduce(lambda x,y: x*y, gray_frame.shape)
                                        dist_w = 0.1*np.exp(-0.2*depth_frame.get_distance(int(x), int((y))))
                                        thresh = dist_w*crop_area

                                    else:
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
                            pass

                    elif self.args.motion_detect == 'None':
                        seq.append(resize_frame)

                    if len(seq) > num:
                        if cam < 10:
                            self.cap.release()
                        return seq
           except:
               print('error. try reconnection')
               # self.device.hardware_reset() # check usage
               
    def __del__(self):
        if self.pipeline:
            self.pipeline.stop()


class TFModel:
    def __init__(self):

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
        self.is_training = tf.placeholder(dtype=tf.bool)

        # build IC3D net
        self.net = model_zoo.I3DNet(inps=self.inputs, n_class=len(ix2label), batch_size=1,
                                    pretrained_model_path=None, final_end_point='SequatialLogits',
                                    dropout_keep_prob=1.0, is_training=self.is_training)

        # logits from IC3D net
        out, merge_op = self.net(self.inputs)
        self.softmax = tf.nn.softmax(out)
        self.merge_op = merge_op

        self.pred = tf.argmax(self.softmax, axis=-1)

        # gpu config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # open session
        self.sess = tf.Session(config=config)
        #self.logger = tf.summary.FileWriter('./log', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(model_path)

        if ckpt:
            print 'restore from {}...'.format(ckpt)
            saver.restore(self.sess, ckpt)

    def run_demo_wrapper(self, frames):
        summary, predictions, softmax = self.sess.run([self.merge_op, self.pred, self.softmax], feed_dict={self.inputs: frames,
                                                          self.is_training: False})
        #predictions, softmax = self.sess.run([self.pred, self.softmax], feed_dict={self.inputs: frames,
        #                                                                           self.is_training: False})
        
        top_3 = map(lambda x: ix2label[int(x)], np.argsort(-softmax)[0][0][:3])

        # for tensorboard
        #self.logger.add_summary(summary)

        mask = map(lambda x: int(ix2label[int(x)]!='Doing other things'), predictions[0])

        # casting
        mask = np.expand_dims(np.expand_dims(mask, axis=0), 2)

        predicted_label = map(lambda x: ix2label[int(x)], predictions[0])

        # if predicted_label.count('Doing other things') > int(RScam.args.video_length*0.8):
        #if predicted_label.count('Doing other things') > int(35 * 0.8):
        #    return 'Doing other things', 'Null'

        # apply mask to predictions
        softmax_masked = np.mean(mask*softmax, axis=1)

        from collections import Counter
        freq_predictions = Counter(predicted_label).most_common()[0][0]

        str_res = freq_predictions.strip()
        confidence = softmax_masked.max()

        return str_res, confidence, top_3

