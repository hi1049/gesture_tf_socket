from TFModel_socket import RScam

import time

import cv2

from socket import *

def main_loop():
    while True:
        send_video()


def send_video():
    csoc = socket(AF_INET, SOCK_STREAM)
    csoc.connect(('127.0.0.1', 8080))

    while True:
        try:
            frames = model.get_frames(num=model.args.video_length,
                                      fps=model.args.fps,
                                      cam=model.args.cam)

            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('frames.avi', fourcc,
                                  30, (224, 224))
            count = 0
            print('frame length: ', len(frames))

            for i in range(0, len(frames)):
                out.write(frames[i])

                count = count + 1

            out.release()

            print('writing video...total frame:', count)

            start_save = time.time()
            with open('frames.avi', 'rb') as file:
                sendfile = file.read()
            csoc.send(str(len(sendfile)).ljust(16))
            csoc.send(sendfile)
            end_save = time.time()

            print('video send time:', end_save - start_save)

            # time.sleep(0.01)
            print('waiting result')

            result = csoc.recv(4096)

            print(result)

        except Exception as e :
            csoc.close()
            exit()



if __name__ == '__main__':
    model = RScam()

    main_loop()
