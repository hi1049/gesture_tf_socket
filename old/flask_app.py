from flask import Flask, render_template, request, Response, jsonify
from camera import Camera
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

stream = ''
_gesture = 'None'

def gen():
    while True:
        yield b'--frame\r\n' \
              b'Content-Type: image/jpeg\r\n\r\n' + stream + b'\r\n\r\n'

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/state/set/gesture')
def set_gesture():
    global _gesture
    _gesture = request.args.get('gesture') # update gesture
    return 'complete'

@app.route('/state/get/gesture')
def get_gesture():
    return jsonify(gesture=_gesture)

@app.route('/update_stream', methods=['GET', 'POST'])
def update():
    global stream
    stream = request.data  # update stream
    return 'Complete update'

@app.route('/live_cam')
def live_cam():
    global stream
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

