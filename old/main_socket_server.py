from TFModel_socket import TFModel
import numpy as np
import time

import requests

from socket import *

import cv2

def recv(csoc, count):
    buf = b''
    while count:
        newbuf = csoc.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main_loop():
    ssoc = socket(AF_INET, SOCK_STREAM)
    ssoc.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    ssoc.bind(('', 8080))
    ssoc.listen(1)
    csoc=None

    try:
        while True:
            if csoc is None:
                print("waiting for connection")
                csoc, addr_info = ssoc.accept()
                print("got connection from", addr_info)
            else:
                start_recv = time.time()
                # print('file receive start')

                length = recv(csoc, 16)
                print('get length', int(length))
                recvfile = recv(csoc, int(length))

                with open('frames_recv.avi', 'wb') as file:
                    file.write(recvfile)
                end_recv = time.time()
                print('file receive complete', end_recv - start_recv)

                start_read = time.time()
                cap = cv2.VideoCapture('frames_recv.avi')
                frames = []
                count = 0
                ret = True
                while ret:
                    ret, frame = cap.read()
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(1)
                    if ret:
                        frames.append(frame)
                        count = count + 1
                # print('frame read time:', time.time() - start_read)
                # print('total frame', len(frames))
                # print('frame read complete, total frame:', count)
                cap.release()
                pred_start_time = time.time()
                result, confidence, top_3 = pred_action(frames)
                pred_end_time = time.time()
                print("action pred time: ", pred_end_time - pred_start_time)
                #time.sleep(0.01)
                csoc.send(result)
                print('send result complete', result, confidence)
                #time.sleep(0.01)
                # csoc.close()
    except Exception as e:
        csoc.close()
        print("ERROR:", e)


def pred_action(frames):
    start_pred = time.time()
    result, confidence, top_3 = model.run_demo_wrapper(np.expand_dims(frames[:-2],0))
    end_pred = time.time()
    print('prediction time', end_pred - start_pred)

    if confidence > 0.1 and result != 'Doing other things':
        print(result, top_3)
    elif top_3[0] != 'Doing other things':
        result = top_3[0]
        print(result, confidence, top_3)
    else:
        result = 'Waiting...'
        print(result, confidence, top_3)
    requests.get(
        'http://127.0.0.1:5000/state/set/gesture',params={'gesture': result})

    return result, confidence, top_3


if __name__ == '__main__':
    model = TFModel()
    while True:
        main_loop()
