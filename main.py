#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, request, jsonify
from camera import VideoCamera

app = Flask(__name__)

top_right_corner = (0, 0 )
bottom_left_corner = (0, 0 )
height = 0;
width = 0;

global_camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        #frame = camera.get_frame()
        frame = camera.stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global global_camera
    return Response(gen(global_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
                    
@app.route('/set_config', methods=['POST']) 
def set_config():
    data = request.json
    print(data)
    global top_right_corner 
    global bottom_left_corner 
    global height
    global width

    top_right_corner = (int(data['config']['tcx']) , int(data['config']['tcy']))
    bottom_left_corner = (int(data['config']['bcx']), int(data['config']['bcy']))
    height = int(data['config']['height'])
    width = int(data['config']['width'])
    
    return jsonify(data)

@app.route('/get_prediction', methods=['GET']) 
def get_prediction():
    origin_x = bottom_left_corner[0]
    origin_y = top_right_corner[1]
    origin = ( origin_x , origin_y )
    height_real = height;
    width_real = width;
    height_img = bottom_left_corner[1] - top_right_corner[1]
    width_img = top_right_corner[0] - bottom_left_corner[1]


    result = VideoCamera().get_prediction(origin, width_real, height_real, width_img, height_img)
    
    return jsonify(result)    
    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True)
    app.run(debug=True)