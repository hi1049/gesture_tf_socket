
import tensorflow as tf
import cv2
import copy
import model_zoo
import argparse
import pyrealsense2 as rs
import os
import time
from darknet.python.darknet import *

import numpy as np

with open('category.txt') as f:
    lines = map(lambda x: x.strip(), f.readlines())

ix2label = dict(zip(range(len(lines)), lines))

cwd = os.getcwd()
model_path = os.path.join('save_model', 'jester-finetune')


def center_crop(x):
    if x >= (args.frame_width) / 4 and x <= (3 * args.frame_width) / 4:
        return 1
    else:
        return 0


def center_point_coord(frame_width, frame_height, pad):
    x1 = frame_width / 2 - pad
    y1 = frame_height / 2 - pad * 2
    x2 = frame_width / 2 + pad
    y2 = frame_height / 2 + pad * 2
    return int(x1), int(y1), int(x2), int(y2)


def center_point(x, y, pad):
    x1, y1, x2, y2 = center_point_coord(args.frame_width, args.frame_height, args.pad)
    if (x >= x1 and x <= x2) and (y >= y1 and y <= y2):
        return 1
    else:
        return 0


def scan_depth(depth_frame, x, y):
    depth_distance = None
    for j in (0, 1, -1):
        for k in (0, 1, -1):
            if depth_frame.get_distance(int(x + j), int(y + k)) != 0:
                depth_distance = depth_frame.get_distance(int(x), int(y))
                continue
    if depth_distance != None or depth_distance != 0 or depth_distance != []:
        return depth_distance
    else:
        return 5


def near_first(depth_frame, r):
    try:
        depth_center = []

        for i in range(0, len(r)):
            _, _, (x, y, w, h) = r[i]
            depth_center.append(scan_depth(depth_frame, x, y))

        if len(depth_center) >= 1:
            _, _, (x, y, w, h) = r[np.argmin(depth_center)]

            if args.detection_area == 'crop':
                if center_crop(x):
                    center_detect = 1

            elif args.detection_area == 'point':
                if center_point(x, y, args.pad):
                    center_detect = 1

            elif args.detection_area == 'None':
                return dict(x=x, y=y,
                            w=w, h=h)

            return dict(x=x, y=y,
                        w=w, h=h)
    except:
        return None


def center_first(depth_frame, r):
    try:
        depth_center = []
        new_r = []

        for i in range(0, len(r)):
            _, _, (x, y, w, h) = r[i]
            if args.detection_area == 'crop':
                if center_crop(x):
                    new_r.append(r[i])
                    depth_center.append(scan_depth(depth_frame, x, y))
                    center_detect = 1

            elif args.detection_area == 'point':
                if center_point(x, y, args.pad):
                    new_r.append(r[i])
                    depth_center.append(scan_depth(depth_frame, x, y))
                    center_detect = 1

            elif args.detection_area == 'None':
                new_r.append(r[i])
                depth_center.append(scan_depth(depth_frame, x, y))

        _, _, (x, y, w, h) = new_r[np.argmin(depth_center)]
        return dict(x=x, y=y,
                    w=w, h=h)

    except:
        return None


def clahe(frame, limit, mask):
    clahe_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(limit, mask)
    clahe_frame = clahe.apply(clahe_frame)
    frame = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)

    return frame


def sobel_filter(frame):
    sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_xy = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    return sobel_xy


def frame_difference(prev, cur):
    prev_gray_frame = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    prev_gray_frame = cv2.medianBlur(prev_gray_frame, 7)
    gray_frame = cv2.medianBlur(gray_frame, 7)

    #prev_sobel = sobel_filter(prev_gray_frame)
    #cur_sobel = sobel_filter(gray_frame)

    frame_diff = cv2.absdiff(gray_frame, prev_gray_frame)

    #frame_diff = cv2.absdiff(prev_sobel, cur_sobel)
    return frame_diff


def draw_histogram(img):

    h = np.zeros((256, 256), dtype=np.uint8)

    hist_item = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        cv2.line(h, (x, 0 + 10), (x, y + 10), (255, 255, 255))

    cv2.line(h, (0, 0 + 10), (0, 5), (255, 255, 255))
    cv2.line(h, (255, 0 + 10), (255, 5), (255, 255, 255))
    y = np.flipud(h)

    return y


def adjust_gamma(frame, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i/255.0)**inv_gamma)*255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(frame, table)


def thresh(depth_frame, frame_diff, brightness, x, y):
    distance = depth_frame.get_distance(int(x), int(y))
    crop_area = reduce(lambda x, y: x * y, frame_diff.shape)

    if args.diff_function == 'binary':
        if brightness >= 80:
            diff_mask = np.where(frame_diff > 20, 1, 0)
            dist_w = 6
        else:
            diff_mask = np.where(frame_diff > 20, 1, 0)
            dist_w = 16
        crop_area = reduce(lambda x, y: x * y, frame_diff.shape) # 179, 135
        thresh = crop_area / dist_w

    elif args.diff_function == 'exp':
        if brightness >= 80:
            diff_mask = np.array(frame_diff > 20, dtype=np.int32)
            dist_w = 0.05 * np.exp(-0.2 * distance)
            #dist_w = 0.025 * np.exp(-0.2 * depth_frame.get_distance(int(x), int((y))))
        else:
            diff_mask = np.array(frame_diff > 10, dtype=np.int32)
            dist_w = 0.005 * np.exp(-0.2 * distance)

    elif args.diff_function == 'log':
        diff_mask = np.array(frame_diff > 20, dtype=np.int32)
        dist_w = (-444*np.log(distance)+1945) / crop_area

    thresh = dist_w * crop_area

    return diff_mask, thresh


def crop_frame(resize_frame):
    cropped_frame = copy.deepcopy(resize_frame)  # frame

    cropped_frame = cropped_frame[int(cropped_frame.shape[0] * 2. / 5):int(cropped_frame.shape[0]),
                                  int(cropped_frame.shape[1] * 0.5 / 5):int(cropped_frame.shape[1] * 4.5 / 5)]  # y,x

    cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    brightness = np.array(cropped_frame_gray)

    brightness = np.average(brightness)

    return cropped_frame, brightness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--video_length', type=int, default=35)

    parser.add_argument('--fps', type=int, default=30)  # 6, 15(1920~424), 30(1280~320), 60

    parser.add_argument('--cam', type=int, default=13)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--frame_width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--frame_height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240

    parser.add_argument('--person_detect', type=str, default='point')  # None, point, crop
    parser.add_argument('--pad', type=int, default=100)

    parser.add_argument('--motion_detect', type=str, default='diff')  # None, diff

    parser.add_argument('--test_mode', type=bool, default=True)  # True, False

    args = parser.parse_args()
    pipeline = None # for IR
    center_detect = 0
    move_detect = False

    if args.args.cam < 10:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

    elif args.args.cam >= 10:
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # depth_aligned_to_color  rs.stream.depth
        config.enable_stream(rs.stream.color, args.frame_width, args.frame_height, rs.format.bgr8, args.fps)  # set color

        if args.args.cam == 11:
            config.enable_stream(rs.stream.depth, args.frame_width, args.frame_height, rs.format.z16, args.fps)  # set depth
            config.enable_stream(rs.stream.infrared, 1, args.frame_width, args.frame_height, rs.format.y8, args.fps)
        if args.args.cam == 12:
            config.enable_stream(rs.stream.depth, args.frame_width, args.frame_height, rs.format.z16, args.fps)  # set depth
            config.enable_stream(rs.stream.infrared, 2, args.frame_width, args.frame_height, rs.format.y8, args.fps)
        if args.args.cam == 13:
            config.enable_stream(rs.stream.depth, args.frame_width, args.frame_height, rs.format.z16, args.fps)  # set depth
            config.enable_stream(rs.stream.infrared, 1, args.frame_width, args.frame_height, rs.format.y8, args.fps) # right imager
            config.enable_stream(rs.stream.infrared, 2, args.frame_width, args.frame_height, rs.format.y8, args.fps) # left imager

        # Start streaming
        profile = pipeline.start(config)
        device = profile.get_device()

    if args.args.cam == 11 or args.args.cam == 12 or args.args.cam == 13:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        for _ in range(30):
            pipeline.wait_for_frames()

    # load yolo v3(tiny) and meta data
    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    # # load yolo v3(tiny) and meta data
    # yolo = load_net("./darknet/cfg/yolov3-tiny-voc.cfg", "./darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
    # meta = load_meta("./darknet/cfg/voc.data")

    seq = []

    while True:
        if args.cam < 10:
            _, frame = cap.read()
        elif args.cam > 10:
            # Wait for a coherent pair of frames: depth and color
            all_frames = pipeline.wait_for_frames()
            color_frame = np.asanyarray(
                all_frames.get_color_frame().get_data()
            )

            if args.cam == 11 or args.cam == 12 or args.cam == 13:
                depth_frame = all_frames.get_depth_frame()
                if args.cam == 11:
                    ir_frame1 = all_frames.get_infrared_frame(1)
                    ir_frame2 = ir_frame1
                elif args.cam == 12:
                    ir_frame2 = all_frames.get_infrared_frame(2)
                    ir_frame1 = ir_frame2
                elif args.cam == 13:
                    ir_frame1 = all_frames.get_infrared_frame(1)
                    ir_frame2 = all_frames.get_infrared_frame(2)

                if not depth_frame or not ir_frame1 or not ir_frame2:
                    continue

                # convert infrared images to numpy arrays
                if args.cam == 11:
                    ir_image1 = np.asanyarray(ir_frame1.get_data())
                    ir_image1 = cv2.cvtColor(ir_image1, cv2.COLOR_GRAY2BGR)
                    frame = ir_image1

                elif args.cam == 12:
                    ir_image2 = np.asanyarray(ir_frame2.get_data())
                    ir_image2 = cv2.cvtColor(ir_image2, cv2.COLOR_GRAY2BGR)
                    frame = ir_image2
                    # frame = color_frame

                elif args.cam == 13:
                    ir_image2 = np.asanyarray(ir_frame2.get_data())
                    ir_image2 = cv2.cvtColor(ir_image2, cv2.COLOR_GRAY2BGR)
                    frame = ir_image2

        if not seq:
            r = np_detect(yolo, meta, frame)

