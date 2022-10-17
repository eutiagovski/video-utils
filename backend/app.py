from flask import Flask, render_template, Response, jsonify
from flask import request
from core import Capture
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

camera = Capture(video=False, url='http://192.168.0.67', browser=True)

@app.route('/')
def index():
    try:
        camera.force_exit()
    except:
        pass
    return render_template('index.html')

import json

@app.route('/get_state', methods = ['GET'])
def get_state():
    state = {
        'detect': camera.DETECT_OBJ,
    }
    return jsonify(state)

@app.route('/configure', methods = ['POST'])
def result():
    data = request.json
    # dict_map = {
    #     'add_photo': camera.make_image(),
    #     'set_detect': camera.set_detect(),
    #     'set_timelapse': camera.set_timelapse(),
    #     'set_detect_emotion': camera.set_detect_emotion(),
    #     'set_detect_moviment': camera.set_detect_motion(),
    #     'set_detect_mask': camera.set_detect_mask()
    # }
    if data == 'add_photo':
        camera.make_image()
    if data == 'set_detect':
        camera.set_detect()
    if data == 'set_detect_emotion':
        camera.set_detect_emotion()
    if data == 'set_detect_moviment':
        camera.set_detect_motion()
    if data == 'set_detect_mask':
        camera.set_detect_mask()
    if data == 'set_timelapse':
        camera.set_timelapse()
    if data == 'start_stream':
        camera.run_web_stream()
    if data == 'stop_stream':
        camera.force_exit()
    
    return json.dumps({'success': True}), 200, {'ContentType':'application/json'}


@app.route('/stream_video')
def video():
    return Response(camera.run_web_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

