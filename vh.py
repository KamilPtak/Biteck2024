import cv2
from flask import Flask, render_template, Response
# from flask_apscheduler import APScheduler
from threading import Thread, Condition
import numpy as np
from flask_mqtt import Mqtt
import json
import dlib
from imutils import face_utils 
import joblib
import colorSender

res_global = (240, 320) #?
fov_global = 90
## FRAME GRABBING
class VideoCapture(Thread):
    res = res_global
    dummy_frame = np.zeros(res + (3,), dtype=np.uint8)

    def __init__(self):
        Thread.__init__(self)
        cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,   res_global[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  res_global[0])
        self.video = cap

        self.frame = self.dummy_frame
        # self.frame_enc = self.enc_frame(self.dummy_frame)
        self.frame_id = 0
        self.frame_lock = Condition()
        self.subs = [9999, 9999]

        print("camera init-ed") #dg

    def __del__(self):
        self.video.release()

    # @staticmethod
    def enc_frame(self, frame, dummy=None):
        enc_retries = 3
        """
        # try:
        #     ret, jpeg = cv2.imencode('.jpg', frame)
        #     return jpeg.tobytes()
        # except:
        #     try:
        #         ret, jpeg = cv2.imencode('.jpg', dummy)
        #         return jpeg.tobytes()
        #     except:
        #         ret, jpeg = cv2.imencode('.jpg', frame)
        #         return jpeg.tobytes()
        """
        dummy = dummy if dummy is not None else self.dummy_frame
        for _ in range(enc_retries):
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
            except:
                continue
        else:
            ret, jpeg = cv2.imencode('.jpg', dummy)
            return jpeg.tobytes()

    def _update_frame(self):
        # extracting frames
        with self.frame_lock: 
            self.frame_lock.wait_for( lambda : (
                    self.subs[0] >= self.frame_id
                and self.subs[1] >= self.frame_id
            ), timeout=.05)
            ret, frame = self.video.read()
            if not np.size(frame): frame = self.dummy_frame
            self.frame = frame 
            # self.frame_enc = self.enc_frame(frame, self.dummy_frame)

            self.frame_id += 1
            self.frame_lock.notify()
            # print("")
            # print("fr", end=" ") #dg

    def run(self): # thread target
        while True:
            self._update_frame()

    def sync_frame(self, sub_id, skip_factor=0):
        last_frame_id = self.frame_id
        while True: 
            # print("SK", self.frame_id, last_frame_id, skip_factor) # dg
            with self.frame_lock: 
                self.frame_lock.wait_for(
                    lambda : self.frame_id > last_frame_id + skip_factor)
                # print("1", end="") #dg
                yield None
                last_frame_id = self.frame_id
                self.subs[sub_id] = last_frame_id + skip_factor

vcap_t = VideoCapture()

## ONBOARD ANALYSIS
dlib_det = dlib.get_frontal_face_detector()

def face_detector(f_gray):
    rects_dlib = dlib_det(f_gray, 1)
    # rects_dmsc_l = [dmsc_det_l[di].detectMultiScale(f_gray, **dmsc_det_params) for di in range(dmsc_det_n)]
    rects = rects_dlib
    print("f:rects:", rects.__len__(), end=";")

    if      rects.__len__() == 1:   rect = rects[0]
    elif    rects.__len__() == 0:   rect = None
    else:                           rect = sorted(rects, 
            key=lambda arr: (arr.bottom()-arr.top())*(arr.right()-arr.left())
        )[-1] # of biggest w*h
    return rect

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def face_landmarker(f_gray, rect=None):
    rect = rect if rect is not None else dlib.rectangle(0, 0, f_gray.shape[1], f_gray.shape[0]) 
        # left, top, right, bottom
    shape = predictor(f_gray, rect)
    marks = face_utils.shape_to_np(shape)    
    return marks

def load_im(f_gray):
    im = f_gray
    im_wh = np.asarray(im.shape)
    im_wh_d2 = np.asarray(im.shape)/2

    rect = face_detector(f_gray)
    if rect is None:
        return None
    rect_bb = np.asarray([
        [max(0,         rect.top()      ),  max(0,          rect.left() )], 
        [min(im_wh[0],  rect.bottom()   ),  min(im_wh[1],   rect.right())]
    ])
    rect_wh         = np.asarray([rect_bb[1,0]-rect_bb[0,0], rect_bb[1,1]-rect_bb[0,1]])
    rect_wh_d2      = rect_wh//2
    rect_cc         = rect_bb[0] + rect_wh_d2
    rect_bb_normd   = (rect_bb - im_wh_d2) / im_wh_d2
    rect_cc_normd   = (rect_cc - im_wh_d2) / im_wh_d2

    f_gray_crop     = f_gray[rect_bb[0,0]:rect_bb[1,0], rect_bb[0,1]:rect_bb[1,1]]

    marks           = face_landmarker(f_gray_crop)
    f_marked        = None # dg_face_draw_landmark(f_gray_crop, marks)
    marks_normd     = (marks - rect_wh_d2) / rect_wh_d2 

    return (
        (f_gray,        im_wh,          im_wh_d2        ),
        (f_gray_crop,   rect_wh,        rect_wh_d2      ),
        (marks,         rect_bb,        rect_cc,        ),
        (marks_normd,   rect_bb_normd,  rect_cc_normd,  ),  
        f_marked, rect,     
    )

clf_emo = joblib.load("./clf_emo")

clf_a   = joblib.load("./clf_a")
clf_b   = joblib.load("./clf_b")

class FacePursue(Thread):
    res = res_global
    fov = fov_global

    skip_frame_f = 2
    dummy_frame = np.zeros(res + (3,), dtype=np.uint8)

    bb_color = (255, 0, 0)
    bb_thick = 10

    def __init__(self):
        Thread.__init__(self)

        self.ff = False # ff = [is] frame found
        self.bb = None
        self.cc = None
        self.bb_normd = None
        self.cc_normd = None

        self.frame_enc = vcap_t.enc_frame(self.dummy_frame)

        self.horz = 0
        self.vert = 0

        self.emo = np.asarray([0,] *7)
        self.rgb = (0,) *3

    @staticmethod
    def draw_rectangle(img, x, y, w, h, c, t):
        img[y:(y+h), (x+(w//2)-t):(x+(w//2)+t)] = c
        img[(y+(h//2)-t):(y+(h//2)+t), x:(x+w)] = c

    def _send_steering(self):
        if self.ff:
            adj = -self.fov * self.cc_normd * .11 # pid
            new_horz = int(np.clip(self.horz+adj[1], -180, 180))
            new_vert = int(np.clip(self.vert+adj[0], -45, 45))

            # MQTT_MSG=json.dumps({"L": str(l), "R": str(r)})
            # self.horz = adj[1]
            # self.vert = adj[0]
            mqtt_client.publish("pico/mot/hor",     str(new_horz), qos=0, retain=False)
            mqtt_client.publish("pico/mot/vert",    str(new_vert), qos=0, retain=False)

    def run(self):
        for _ in vcap_t.sync_frame(sub_id=0, skip_factor=self.skip_frame_f):

            try:
                frame = vcap_t.frame
                f_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            except Exception as ee:
                print("fp:ramka" + str(ee))
                self.ff = False
                continue
            
            
            im_params = load_im(f_gray)
            if im_params is None:
                self.ff = False
                print("fp:face", end=";")
                continue

            (
                (f_gray,        im_wh,          im_wh_d2        ),
                (f_gray_crop,   rect_wh,        rect_wh_d2      ),
                (marks,         rect_bb,        rect_cc,        ),
                (marks_normd,   rect_bb_normd,  rect_cc_normd,  ),  
                f_marked, rect,     
            ) = im_params

            print("fp:ok", end=";")
            self.ff = True
            self.cc = rect_cc
            self.bb = rect_bb
            self.bb_normd = rect_bb_normd
            self.cc_normd = rect_cc_normd
            
            self._send_steering()
            self.frame_enc = vcap_t.enc_frame(f_gray, self.dummy_frame)

            ## fn 
            emo = clf_emo.predict_proba([marks_normd.flatten()])
            self.emo = emo
            print("\nmood values:", 
                  {k: v for v, k in zip(emo[0], ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', "neutral"])}, end=";")

            lab = (1, float(clf_a.predict(emo)[0]), float(clf_b.predict(emo)[0]))
            pixel = np.asarray([[(lab[0]*255, (lab[1]+1)*127, (lab[2]+1)*127)]], dtype=np.uint8)
            # print("pixel", pixel, pixel.dtype)
            rgb = cv2.cvtColor(pixel,  cv2.COLOR_Lab2BGR)
            # print("rgb", rgb, rgb.dtype)
            rgb = rgb[0,0]
            self.rgb = rgb
            print("mood color lab:", lab, "; rgb:", rgb)

            colorSender.setLightsColor(rgb)

            
fpur_t = FacePursue()

## MQTT
MA_CONFIG = {
    "MQTT_BROKER_URL": "siur123.pl",
    "MQTT_BROKER_PORT": 18833,
    "MQTT_USERNAME": "flask",
    "MQTT_PASSWORD": "",
    "MQTT_KEEPALIVE": 30,
    "MQTT_TLS_ENABLED": False,
}

mqtt_client = Mqtt()

@mqtt_client.on_connect()
def handle_connect(client, userdata, flags, rc):
   if rc == 0:
       mqtt_client.subscribe("#") # subscribe topic
   else:
        pass # print('Bad connection. Code:', rc)

@mqtt_client.on_message()
def handle_mqtt_message(client, userdata, message):
    topic=message.topic
    content=message.payload.decode() #'ASCII')

    if type(topic) is tuple:
        topic = ''.join(topic)

    if      topic == "pico/mot/hor":
        fpur_t.horz = int(content)
    elif    topic == "pico/mot/vert":
        fpur_t.vert = int(content)
        

    print("mqtt received:",  topic, content)

## STREAM
app = Flask(__name__)
app.config.from_mapping(
    **MA_CONFIG
)

sw = 20
sh = 200
def feed_stats_cont():
    for _ in vcap_t.sync_frame(sub_id=1, skip_factor=0):
        frame = np.zeros((sh, sw*10, 3), dtype=np.uint8)
        for io, eo in enumerate(fpur_t.emo.flatten()):
            frame[0:int(eo*sh), int(io*sw):int((io+.9)*sw)] = 255
        for io, eo in enumerate(fpur_t.rgb):
            frame[0:int(eo*sh), (sw*7)+int(io*sw):(sw*7)+int((io+.9)*sw)] = fpur_t.rgb
        # print("\n\n\n", fpur_t.rgb, "\n\n\n")

    # for _ in vcap_t.sync_frame(sub_id=1, skip_factor=ppur_t.skip_frame_f):
    #     frame = ppur_t.frame_enc #dg
        frame = vcap_t.enc_frame(frame[::-1], None)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/stats_feed')
def stats_feed():
    return Response(feed_stats_cont(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def feed_frame_cont():
    for _ in vcap_t.sync_frame(sub_id=1, skip_factor=0):
        frame = vcap_t.frame
        if fpur_t.ff:
            try:
                fpur_t.draw_rectangle(
                    img=frame,
                    x=fpur_t.bb[0, 1],
                    y=fpur_t.bb[0, 0],
                    w=(fpur_t.bb[1, 1]-fpur_t.bb[0, 1]),
                    h=(fpur_t.bb[1, 0]-fpur_t.bb[0, 0]),
                    c=fpur_t.bb_color,
                    t=fpur_t.bb_thick,
                )
                #     frame, *ppur_t.bb, 
                # ppur_t.bb_color, ppur_t.bb_thick)
            except Exception as ee: 
                print("fp:-ramka" + str(ee))
                pass

    # for _ in vcap_t.sync_frame(sub_id=1, skip_factor=ppur_t.skip_frame_f):
    #     frame = ppur_t.frame_enc #dg
        frame = vcap_t.enc_frame(frame, None)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(feed_frame_cont(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def feed_frame_single():
    frame = vcap_t.frame

    frame = vcap_t.enc_frame(frame, None)
    return frame

@app.route('/frame_capture')
def frame_capture():
    return Response(feed_frame_single(), mimetype='image/jpeg')

if __name__ == '__main__':
    # threads
    vcap_t.start()
    fpur_t.start()

    mqtt_client.init_app(app)
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=False, use_reloader=False)

# # scp ./vid_cs.py pi@172.16.25.128:/home/pi/wizja/
# # libcamerify python3 vid_cs.py
# https://stackoverflow.com/questions/61047207/opencv-videocapture-does-not-work-in-flask-project-but-works-in-basic-example
