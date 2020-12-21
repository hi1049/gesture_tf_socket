from TFModel_socket import TFModel, RScam

import requests
import json
import numpy as np
import time
from socket import *
from phue import Bridge
from flask import Flask
import threading

app = Flask(__name__)

class Multi_device:
    def __init__(self):
        self.device = 'Hue'  # 'TV' or 'Aircon' or 'Hue'
        self.Hue_ip = '192.168.0.6'
        self.Aircon_ip = '155.230.14.111'

        self.device_status = 'OFF'
        self.activation = False

        if self.device == 'Hue':
            self.xycolor_select = 0
            self.bright = 0

        elif self.device == 'TV':

            tv_param = open('tv_param.txt', 'r')
            param = tv_param.readline()

            self.device_status, self.channel, self.vol = param.split(' ')

        elif self.device == 'Aircon':
            self.windStrength = 0
            self.temperature = 0

    def response(self, action_result):

        if self.device == 'Hue':

            if action_result == 'Thumb Up':
                self.activation = True
                print('gesture activation', self.activation)

            elif action_result == 'Thumb Down':
                self.activation = False
                print('gesture deactivation', self.activation)

            if self.activation:
                b = Bridge(self.Hue_ip)
                b.connect()
                print('light check: ', b.get_light(1, 'on'))

                if b.get_light(1, 'on') == True:
                    self.device_status = 'ON'
                else:
                    self.device_status = 'OFF'

                self.bright = b.get_light(1, 'bri')

                # Red: 0.675, 0.322 Green: 0.4091, 0.518 Blue: 0.167, 0.04

                xycolor_list = [[0.675, 0.322], [0.4091, 0.518], [0.167, 0.04], [1, 1]]  # R, G, B, W
                xycolor_name = ['Red', 'Green', 'Blue', 'White']

                # xycolor = b.get_light(1, 'xy')  # 1,1 = white
                # b.set_light(1, 'xy', xycolor)

                if action_result == 'Rolling Hand Backward':
                    b.set_light(1, 'on', True)
                    print('light on')
                    self.device_status = 'ON'
                    # Rolling Hand Backward
                    # Doing other things
                elif action_result == 'Stop Sign':
                    b.set_light(1, 'on', False)
                    print('light off')
                    self.device_status = 'OFF'

                elif action_result == 'Sliding Two Fingers Left':
                    self.xycolor_select = self.xycolor_select - 1
                    if self.xycolor_select < 0: self.xycolor_select = len(xycolor_list) - 1
                    print(xycolor_name[self.xycolor_select])

                elif action_result == 'Sliding Two Fingers Right':
                    self.xycolor_select = self.xycolor_select + 1
                    if self.xycolor_select > len(xycolor_list) - 1: self.xycolor_select = 0
                    print(xycolor_name[self.xycolor_select])


                # elif action_result == 'Swiping Left':

                # elif action_result == 'Swiping Right':

                elif action_result == 'Sliding Two Fingers Up':
                    self.bright = self.bright + 10
                    if self.bright > 254: self.bright = 254

                elif action_result == 'Sliding Two Fingers Down':
                    self.bright = self.bright - 10
                    if self.bright < 1: self.bright = 1

                elif action_result == 'Swiping Up':
                    self.bright = self.bright + 100
                    if self.bright > 254: self.bright = 254

                elif action_result == 'Swiping Down':
                    self.bright = self.bright - 100
                    if self.bright < 1: self.bright = 1

                if action_result != None and action_result != 'Waiting...':
                    b.set_light(1, 'xy', xycolor_list[self.xycolor_select])
                    b.set_light(1, 'bri', self.bright)

        if self.device == 'Aircon':
            HOST = self.Aircon_ip  # Standard loopback interface address (localhost)
            PORT = 5252  # Port to listen on (non-privileged ports are > 1023)

            max_windStrength = 8
            min_windStrength = 1
            max_temperature = 36
            min_temperature = 20

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as aircon_soc:
                aircon_soc.connect((HOST, PORT))
                st = 'status'

                byt = st.encode()
                aircon_soc.sendall(byt)
                aircon_soc.settimeout(10.0)

                # receive aircon data
                try:
                    data = aircon_soc.recv(2048)
                    data = data.decode("utf-8")
                    test = data.split(',')
                    Operation = int(test[5])  # ??
                    OpMode = int(test[6])  # ??
                    WindStrength_Left = int(test[8])
                    WindStrength_Right = int(test[9])
                    TempCfg = int(test[33]) / 2  # target humidity
                    HumidityCfg = int(test[116])  # target humidity

                    InTempCur = int(test[34]) / 2  # inside temperature
                    SensorHumidity = int(test[115])  # inside humidity

                except socket.timeout as e:
                    print('Timeout')
                    time.sleep(60)

                aircon_soc.shutdown(SHUT_RDWR)
                aircon_soc.close()

            time.sleep(60)

            self.windStrength = WindStrength_Left
            self.temperature = TempCfg

            if action_result == 'Thumb Up':
                self.activation = True

            elif action_result == 'Thumb Down':
                self.activation = False

            if self.activation:
                if action_result == 'Rolling Hand Backward':
                    st = ('capsule,turnOn')
                elif action_result == 'Stop Sign':
                    st = ('capsule,turnOff')

                elif action_result == 'Sliding Two Fingers Left':
                    if not self.windStrength <= min_windStrength:
                        self.windStrength = self.windStrength - 1
                    else: self.windStrength = min_windStrength

                elif action_result == 'Sliding Two Fingers Right':
                    if not self.windStrength > max_windStrength:
                        self.windStrength = self.windStrength + 1
                    else: self.windStrength = max_windStrength

                elif action_result == 'Swiping Left':
                    if not self.windStrength <= min_windStrength:
                        self.windStrength = self.windStrength - 3
                    else: self.windStrength = min_windStrength

                elif action_result == 'Swiping Right':
                    if not self.windStrength > max_windStrength:
                        self.windStrength = self.windStrength + 3
                    else: self.windStrength = max_windStrength

                elif action_result == 'Sliding Two Fingers Up':
                    if not self.temperature > max_temperature:
                        self.temperature = self.temperature + 1
                    else: self.temperature = max_temperature

                elif action_result == 'Sliding Two Fingers Down':
                    if not self.temperature <= min_temperature:
                        self.temperature = self.temperature - 1
                    else: self.temperature = min_temperature

                elif action_result == 'Swiping Up':
                    if not self.temperature > max_temperature:
                        self.temperature = self.temperature + 3
                    else: self.temperature = max_temperature

                elif action_result == 'Swiping Down':
                    if not self.temperature <= min_temperature:
                        self.temperature = self.temperature - 3
                    else: self.temperature = min_temperature


                    # turnOn, turnOff, windStrength:#, windMode:#, temperature:#, leftRightOn, leftRightOff
                if self.device_status == 'ON':
                    st = ('capsule,windStrength:%d,temperature:%d')%(self.windStrength, self.temperature)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as aircon_soc:
                    aircon_soc.connect((HOST, PORT))
                    byt = st.encode()
                    aircon_soc.sendall(byt)
                    aircon_soc.settimeout(10.0)

                    try:
                        data = aircon_soc.recv(2048)
                        data = data.decode("utf-8")

                    except socket.timeout as e:
                        print('Timeout')
                        time.sleep(60)

                    aircon_soc.shutdown(socket.SHUT_RDWR)
                    aircon_soc.close()

                time.sleep(60)

        elif self.device == 'TV':
            max_channel = 9
            max_vol = 50

            if action_result == 'Thumb Up':
                self.activation = True

            elif action_result == 'Thumb Down':
                self.activation = False

            if self.activation:
                if action_result == 'Sliding Two Fingers Left':
                    if self.channel > 0:
                        self.channel = self.channel - 1
                    if self.channel <= 0: 
                        self.channel = max_channel

                elif action_result == 'Sliding Two Fingers Right':
                    if self.channel < max_channel:
                        self.channel = self.channel + 1
                    if self.channel >= max_channel: 
                        self.channel = 0

                elif action_result == 'Swiping Left':
                    if self.channel > 0:
                        self.channel = self.channel - 3
                    if self.channel <= 0: 
                        self.channel = max_channel

                elif action_result == 'Swiping Right':
                    if self.channel < max_channel:
                        self.channel = self.channel + 3
                    if self.channel >= max_channel: 
                        self.channel = 0

                elif action_result == 'Sliding Two Fingers Up':
                    if self.vol < max_vol:
                        self.vol = self.vol + 1
                    if self.vol >= max_vol: 
                        self.vol = max_vol

                elif action_result == 'Sliding Two Fingers Down':
                    if self.vol > 0:
                        self.vol = self.vol - 1
                    if self.vol <= 0: 
                        self.vol = 0

                elif action_result == 'Swiping Up':
                    if self.vol < max_vol:
                        self.vol = self.vol + 3
                    if self.vol >= max_vol: 
                        self.vol = max_vol

                elif action_result == 'Swiping Down':
                    if self.vol > 0:
                        self.vol = self.vol - 3
                    if self.vol <= 0: 
                        self.vol = 0

                elif action_result == 'Stop Sign':
                    self.device_status = 'OFF'
                elif action_result == 'Rolling Hand Backward':
                    self.device_status = 'ON'

                tv_param = open('tv_param.txt', 'w')
                tv_data = '%s %d %d'%(self.device_status, int(self.channel), int(self.vol))
                tv_param.write(tv_data)
                tv_param.close()


        if self.device == 'Hue':
            paramA = self.xycolor_select
            paramB = self.bright

        elif self.device == 'TV':
            paramA = self.channel
            paramB = self.vol

        elif self.device == 'Aircon':
            paramA = self.windStrength
            paramB = self.temperature
        print(self.activation, self.device_status, paramA, paramB)
        return self.activation, self.device_status, paramA, paramB

    def main_loop(self):

        while True:
            self.run_demo()


    def run_demo(self):

        result = 'None'; confidence = 0.0
        frames = cam.get_frames(num=cam.args.video_length,
                                  fps=cam.args.fps,
                                  cam=cam.args.cam)

        if frames and cam.center_detect and cam.move_detect:
            result, confidence, top_3 = model.run_demo_wrapper(np.expand_dims(frames,0))

        if confidence > 0.3 and result!='Doing other things':
            self.activation, self.device_status, paramA, paramB = self.response(result)

            print('home', self.device, self.device_status, paramA, paramB, result)


        else:
            result = 'Waiting...'

        if not eval(cam.args.debug) and result != 'Waiting...':
        
            #print('control')
        #     # to main controller...
            try:
                #print('test')
                #requests.get(
                #'https://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home', result)) # ceslea.ml > domain error
                requests.get('http://192.168.0.4:3001/api/v1/actions/action/{}/{}_{}_{}_{}_{}'.format('home', self.device, self.device_status, paramA, paramB, result))
        
            except:
                pass
        
        # to local webdemo page...
        requests.get(
            'http://127.0.0.1:5000/state/set/gesture',params={'gesture': result})
        # time.sleep(0.5)

@app.route("/light/on")
def controller_response_on():
    print('request from light on')
    multi_device.response("Rolling Hand Backward")
    return 'ON'

@app.route("/light/off")
def controller_response_off():
    print('request from light off')
    multi_device.response("Stop Sign")
    return 'OFF'

def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.start()

    model = TFModel()
    cam = RScam()
    multi_device = Multi_device()

    print(eval(cam.args.debug))

    multi_device.main_loop()

