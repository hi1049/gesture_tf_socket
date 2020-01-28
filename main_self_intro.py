from TFModel_socket import TFModel, RScam

import requests
import json
import numpy as np
import time

def main_loop():

    device = 'Hue' # 'TV' or 'Aircon' or 'Hue'
    channel = 10

    vol = 10

    device_status = 'ON' # 'ON' or 'OFF'
    activation = False

    while True:
        activation, device, device_status, channel, vol = run_demo(activation, device, device_status, channel, vol)

def run_demo(activation, device, device_status, channel, vol):
    max_channel = 30
    max_vol = 50

    result = 'None'; confidence = 0.0
    frames = cam.get_frames(num=cam.args.video_length,
                              fps=cam.args.fps,
                              cam=cam.args.cam)

    if frames and cam.center_detect and cam.move_detect:
        result, confidence, top_3 = model.run_demo_wrapper(np.expand_dims(frames,0))

    if confidence > 0.3 and result!='Doing other things':

        if result == 'Thumb Up':
            activation = True

        elif result == 'Thumb Down':
            activation = False

        if activation == True:
            if result == 'Sliding Two Fingers Left':
                if not channel <= 0:
                    channel = channel - 1
                else: channel = 0

            elif result == 'Sliding Two Fingers Right':
                if not channel > max_channel:
                    channel = channel + 1
                else: channel = max_channel

            elif result == 'Swiping Left':
                if not channel <= 0:
                    channel = channel - 3
                else: channel = 0

            elif result == 'Swiping Right':
                if not channel > max_channel:
                    channel = channel + 3
                else: channel = max_channel

            elif result == 'Sliding Two Fingers Up':
                if not vol > max_vol:
                    vol = vol + 1
                else: vol = max_vol

            elif result == 'Sliding Two Fingers Down':
                if not vol <= 0:
                    vol = vol - 1
                else: vol = 0

            elif result == 'Swiping Up':
                if not vol > max_vol:
                    vol = vol + 3
                else: vol = max_vol

            elif result == 'Swiping Down':
                if not vol <= 0:
                    vol = vol - 3
                else: vol = 0

            elif result == 'Stop Sign':
                device_status = 'OFF'
            elif result == 'Rolling Hand Backward':
                device_status = 'ON'
        print(activation, device, device_status, channel, vol, result)

    else:
        result = 'Waiting...'

    
    if not eval(cam.args.debug) and result != 'Waiting...':
        #print('control')
        # to main controller...
        try:
            #print('test')
            #requests.get(
            #'https://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home', result)) # ceslea.ml > domain error
            requests.post(
            'http://127.0.0.1:50001/api/v1/actions/action/{}/{}_{}_{}_{}_{}'.format('home', device, device_status, channel, vol, result))
        except:
            pass

    # to local webdemo page...
    requests.get(
        'http://127.0.0.1:5000/state/set/gesture',params={'gesture': result})

    # time.sleep(0.5)
    return activation, device, device_status, channel, vol

if __name__ == '__main__':
    model = TFModel()
    cam = RScam()
    #print(eval(cam.args.debug))
    main_loop()
