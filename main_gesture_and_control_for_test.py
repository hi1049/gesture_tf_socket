from TF_model_cam import TFModel, CAM

import requests
import json
import numpy as np
import cv2
import time
import socket
import os
#from socket import *
import threading
#import multiprocessing
from datetime import datetime
from pytz import timezone

today = lambda: datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d')
time_now = lambda: datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')
tester_name = ""

city_images = [cv2.imread("./images/city/" + x) for x in os.listdir("./images/city/")]
cat_images = [cv2.imread("./images/cat/" + x) for x in os.listdir("./images/cat/")]
dog_images = [cv2.imread("./images/dog/" + x) for x in os.listdir("./images/dog/")]
road_images = [cv2.imread("./images/road/" + x) for x in os.listdir("./images/road/")]
universe_images = [cv2.imread("./images/universe/" + x) for x in os.listdir("./images/universe/")]
category = [city_images, cat_images, dog_images, road_images, universe_images]
category_name = ["City", "Cat", "Dog", "Road", "Universe"]
category_num = 0
slide_num = [0, 0, 0, 0, 0]

ex_mode = False
result = "None"

def new_status_canvas():
    canvas = np.full((150, 640), 255, np.uint8)
    cv2.putText(canvas, "STATUS  :", (10,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    cv2.putText(canvas, "GESTURE :", (10,80), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    cv2.putText(canvas, "CONFIDENCE :", (10,120), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    return canvas

def new_slide_canvas(image):
    canvas = np.full((540, 640, 3), 0, np.uint8)
    canvas[50:image.shape[0]+50,:] = image
    cv2.putText(canvas, "Model Action : ", (10,35), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2, cv2.LINE_AA)
    cv2.putText(canvas, "Slide Name :", (10,515), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2, cv2.LINE_AA)
    return canvas

def putText_on_status(Text1, Text2, Text3):
    canvas = new_status_canvas()
    cv2.putText(canvas, Text1, (200,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    cv2.putText(canvas, Text2, (200,80), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    cv2.putText(canvas, Text3, (240,120), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    cv2.imshow("STATUS WONDOW", canvas)
    return canvas

def putText_on_slide(image, Text1, Text2):
    canvas = new_slide_canvas(image)
    cv2.putText(canvas, Text1, (280,35), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2, cv2.LINE_AA)
    cv2.putText(canvas, Text2, (280,515), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2, cv2.LINE_AA)
    cv2.imshow("SLIDE WINDOW", canvas)
    return canvas

def add_log(msg, tester_name):
    cur_today = today()
    cur_time = time_now()
    with open('./log_test/' + cur_today + "_" + tester_name + ".txt", "a") as f:
        f.write(cur_today + ' | '+ cur_time + ' | (model_result:{}) | (confidence:{}) | (Top 3 :[{}])\n'.format( \
            msg[0], msg[1], msg[2]))

class Gesture:
    def main_loop(self):
        global slide_num
        global ex_mode
        global category_num
        ex_mode = True
        category_num = 0
        slide_num = [0, 0, 0, 0, 0]
        putText_on_status("press ENTER to Start", "Wait for input", category_name[category_num] + " : " + str(slide_num[category_num]))
        #putText_on_slide(category[category_num][slide_num[category_num]], "Ready", category_name[category_num] + " _ " + str(slide_num[category_num]))
        start = raw_input("----------- press ENTER key to start -----------")
        while True:
            self.run_demo()

    def run_demo(self):
        global slide_num
        global ex_mode
        global result
        global category_num
        global category_name
        confidence = 0.0
        if ex_mode :
            #putText_on_status("Recording...", result, str(slide_num[category_num]))
            print("mode True")
        else:
            putText_on_status("Wait for Thumb Up", "Computing...", str(slide_num[category_num]))
            print("mode False")

        frames = cam.get_frames(num=cam.args.video_length,
                                  fps=cam.args.fps,
                                  cam=0)
        print("get frame")

        if not frames:
            print("Lost Contact")
            putText_on_status("Lost Contact", "check camera server", str(slide_num[category_num]))
            pass

        if frames and cam.center_detect and cam.move_detect:
            result, confidence, top_3 = model.run_demo_wrapper(np.expand_dims(frames,0))
            print("result :{}, confidence:{}, top_3:({})".format(result, confidence, top_3))
            #putText_on_status("Recording...", "Computing...", str(slide_num))

        # Swiping Right / Swiping Left / Sliding Two Fingers Right / Sliding Two Fingers Left / Thumb Up / Thumb Down
        # status : status, gesture, sl num // slide: image, action, sl name
        if confidence > 0.3:
            #requests.get('http://192.168.0.21:3001/api/v1/actions/action/{}/{}_{}_{}_{}_{}'.format('home', result, confidence, 'phue_lamp', 'controlA', 'controlB'))
            if result == "Doing other things":
                #if top_3[0] == "Swiping Right" or top_3[0] == "Swiping Left" or top_3[0] == "Sliding Two Fingers Right" or top_3[0] == "Sliding Two Fingers Left" or top_3[0] == "Thumb Up":
                if top_3[0] != "Doing other things":
                    result = top_3[0]
            
            add_log([result, confidence, top_3], tester_name)
            putText_on_status("Evaluation Proceeding", result, str(confidence))


            

            # if result == "Thumb Up":
            #     if not ex_mode:
            #         ex_mode = True
            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Demo Start", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Demo Start", category_name[category_num] + " _ " + str(slide_num[category_num]))

            # elif result == "Thumb Down":
            #     if ex_mode:
            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Demo Finished", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Demo Finished", category_name[category_num] + " _ " + str(slide_num[category_num]))
            #         ex_mode = True

            # elif result == "Swiping Right":
            #     if ex_mode:
            #         slide_num[category_num] += 1
            #         if slide_num[category_num] == 7:
            #             slide_num[category_num] = 0

            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Recognizing Gesture", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Next Slide", category_name[category_num] + " _ " + str(slide_num[category_num]))

            # elif result == "Swiping Left":
            #     if ex_mode:
            #         slide_num[category_num] += -1
            #         if slide_num[category_num] == -1:
            #             slide_num[category_num] = 6

            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Recognizing Gesture", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Prev Slide", category_name[category_num] + " _ " + str(slide_num[category_num]))

            # elif result == "Sliding Two Fingers Right":
            #     if ex_mode:
            #         category_num += 1
            #         if category_num == 5:
            #             category_num = 0

            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Recognizing Gesture", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Next Category", category_name[category_num] + " _ " + str(slide_num[category_num]))

            # elif result == "Sliding Two Fingers Left":
            #     if ex_mode:
            #         category_num += -1
            #         if category_num == -1:
            #             category_num = 4

            #         add_log([result, confidence, top_3], tester_name)
            #         putText_on_status("Recognizing Gesture", result, str(slide_num[category_num]))
            #         #putText_on_slide(category[category_num][slide_num[category_num]], "Prev Category", category_name[category_num] + " _ " + str(slide_num[category_num]))

            #else:
            #    result = "Doing other things"


if __name__ == '__main__':
    print("## load TF model")
    model = TFModel()
    print("## load complete\n## set camera")
    cam = CAM()
    print("## setting complete\n")
    multi_device = Gesture()

    print("## start demo")
    while True:
        try:
            tester_name = raw_input("please enter tester's ID ( 3 digit ex) 002, 047 ... ) : ")
            multi_device.main_loop()
        except KeyboardInterrupt :
            cv2.destroyWindow('roi')
            print("\n\nsave log...")
            while True:
                choice = raw_input("\n##############\nq : quit\nr : restart\n##############\nq or r :")
                if choice == 'q':
                    cv2.destroyAllWindows()
                    break
                elif choice == 'r':
                    cam.cap.release()
                    break
                else:
                    continue
            if choice == 'q':
                break
            else:
                continue
    print("End")
