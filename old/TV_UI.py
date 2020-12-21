import cv2
import os
from natsort import natsorted
import numpy as np

vid_list = [os.path.join('video_repos', video) for video in natsorted(os.listdir('video_repos')) ]

vid_num = 0

def read_param():

    f = open('tv_param.txt', 'r')
    param = f.readline()

    device_status, channel, vol = param.split(' ')

    channel = int(channel)
    vol = int(vol)

    return device_status, channel, vol


while 1:

    device_status, channel, vol = read_param()

    cap = cv2.VideoCapture(vid_list[channel])
    cap.set(cv2.CAP_PROP_FPS, 30)

    while 1:
        device_status, channel, vol = read_param()

        if device_status == 'ON':
            ret, vid_frame = cap.read()

            if ret == False:
                continue # for inf loop
            #else:
            vid_frame = cv2.resize(vid_frame, (640, 480))

        else:
            vid_frame = np.zeros((480, 640, 3))

        cv2.putText(vid_frame, ('CH:'+str(channel)), (10, 450), cv2.FONT_ITALIC, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vid_frame, ('vol:'+str(vol)), (225, 450), cv2.FONT_ITALIC, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('tv', vid_frame)
        cv2.waitKey(10)

        nxt_device_status, nxt_channel, nxt_vol = read_param()

        if nxt_device_status != device_status or nxt_channel != channel:
            break

cap.release()
cv2.destoryAllWindows()
