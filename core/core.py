import os
import cv2
import requests
import datetime
import emoji
from deepface import DeepFace

from core.tf_hub import inference_from_tf_model
from .helpers import make_video, save_image
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import imutils
from .rcnn import check_masks
from cvlib.object_detection import draw_bbox
import cvlib as cv

class Capture:
    def __init__(self, url="", detect_obj=False, detect_emotion=False, video=True, browser=False):
        self.AWB = 0
        self.URL = url
        self.PRED = False
        self.DETECT_OBJ = detect_obj
        self.DETECT_EMOTION = detect_emotion

        # self.CAP = cv2.VideoCapture()

        self.OBJ_DETECTOR = cv2.CascadeClassifier(
            f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.EMOJI_MAP = {'angry': ':angry_face:', 'happy': ':beaming_face_with_smiling_eyes:', 'disgust': ':smirking_face:',
                          'sad': ':sad_but_relieved_face:', 'surprise': ':astonished_face:', 'neutral': ':face_without_mouth:', 'fear': ':anguished_face:'}

        self.SAVE_PATH = "/images/"
        self.TIMELAPSE_PATH = "/timelapses/"
        self.COUNT = 0
        self.TIMELAPSE = False
        self.TIMELAPSE_INTERVAL = 60
        self.VIDEO = video

        self.FIRST_FRAME = None
        self.MIN_AREA = 500
        self.AMBIENT_STATE = "AMBIENTE"
        self.FIXED_TEXT = 'CASA | '
        self.AMBIENT = 'EXTERNO 1 |'
        self.AMBIENT_STATE = ''
        self.DETECT_MOVIMENT = False

        self.DETECT_MASK = False

        self.BROWSER = browser

        self.FRAME = None
        self.DESCRIBE_OBJECTS = False

        try:
            if not os.path.exists(self.TIMELAPSE_PATH):
                os.mkdir(self.TIMELAPSE_PATH)

            if not os.path.exists(self.SAVE_PATH):
                os.mkdir(self.SAVE_PATH)
        except PermissionError:
            pass

    def set_resolution(self, index: int = 1, verbose: bool = False):
        try:
            if verbose:
                resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
                print("available resolutions\n{}".format(resolutions))

            if index in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
                requests.get(
                    self.URL + "/control?var=framesize&val={}".format(index))
            else:
                print("Wrong index")
        except:
            print("SET_RESOLUTION: something went wrong")

    def set_quality(self, value: int = 1, verbose: bool = False):
        try:
            if value >= 10 and value <= 63:
                requests.get(
                    self.URL + "/control?var=quality&val={}".format(value))
        except:
            print("SET_QUALITY: something went wrong")

    def set_awb(self):
        awb = self.AWB
        try:
            awb = not awb
            requests.get(
                self.URL + "/control?var=awb&val={}".format(1 if awb else 0))
        except:
            print("SET_QUALITY: something went wrong")

        self.AWB = awb

    def force_exit(self):
        self.CAP.release()
        cv2.destroyAllWindows()

    def set_detect_model(self):
        models = os.listdir(cv2.data.haarcascades)
        models_string = ""
        for i, m in enumerate(models):
            models_string += f"{i}: {m}\n"

        model_idx = int(input(f'Selecione um modelo:\n\n{models_string}'))

        self.OBJ_DETECTOR = cv2.CascadeClassifier(
            f'{cv2.data.haarcascades}{models[model_idx]}')

    def set_detect(self):
        if self.DETECT_OBJ == True:
            self.DETECT_OBJ = False
        else:
            self.DETECT_OBJ = True
        print(f'[INFO] Detecção de Objetos: {self.DETECT_OBJ}')
        

    def set_detect_emotion(self):
        if self.DETECT_EMOTION == False:
            self.DETECT_OBJ = True
            self.DETECT_EMOTION = True
        else:
            self.DETECT_OBJ = False
            self.DETECT_EMOTION = False
        print(f'[INFO] Detecção de emoção: {self.DETECT_OBJ}')
        

    def get_version(self):
        fps = self.CAP.get(cv2.CAP_PROP_FPS)
        self.FPS = fps
        print(f'[INFO] Iniciando video a {self.FPS}fps')

    def set_timelapse_interval(self, interval=None):
        self.TIMELAPSE_INTERVAL = interval
        print(
            f'[INFO] Time lapse {"ativado" if self.TIMELAPSE == True else "desativado."}')

    def set_timelapse(self):
        if self.TIMELAPSE == False:
            self.TIMELAPSE = True

        else:
            self.TIMELAPSE = False
            self.make_timelapse_video()

        print(
            f'[INFO] Time lapse {"ativado" if self.TIMELAPSE == True else "desativado."}')

    def make_image(self):
        save_image(self.FRAME)

    def make_timelapse_video(self):
        make_video()

    def set_detect_motion(self):
        """
        THIS IS A EXPERIMENTAL CODE FOR MOTION DETECT
        """
        if self.DETECT_MOVIMENT == False:
            self.DETECT_MOVIMENT = True
        else:
            self.DETECT_MOVIMENT = False
        print(f'[INFO] Detecção de movimento: {self.DETECT_MOVIMENT}')
        

    def set_detect_mask(self):
        """
        THIS IS A EXPERIMENTAL CODE FOR MOTION DETECT
        """
        if self.DETECT_MASK == False:
            self.DETECT_MASK = True
        else:
            self.DETECT_MASK = False
        print(f'[INFO] Detecção de objetos com Coco: {self.DETECT_MASK}')


    def detect_motion(self, frame):
        """
        THIS IS A EXPERIMENTAL CODE FOR MOTION DETECT
        """

        motion_frame = cv2.resize(frame, (500, 500))
        motion_gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
        motion_blured = cv2.GaussianBlur(motion_gray, (21, 21), 0)

        if self.FIRST_FRAME is None:
            self.FIRST_FRAME = motion_blured

        motion_delta = cv2.absdiff(self.FIRST_FRAME, motion_blured)

        thresh = cv2.threshold(
            motion_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < self.MIN_AREA:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            self.AMBIENT_STATE = "AMBIENTE OCUPADO"

    def set_describe_objects(self):
        if self.DESCRIBE_OBJECTS == True:
            self.DESCRIBE_OBJECTS = False
        else:
            self.DESCRIBE_OBJECTS = True
        print(f'[INFO] Detecção de objetos com YoLo: {self.DESCRIBE_OBJECTS}')

    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        objects = self.OBJ_DETECTOR.detectMultiScale(
            gray, minSize=(112, 112))

        if self.DESCRIBE_OBJECTS:
            bbox, label, conf = cv.detect_common_objects(frame)
            frame = draw_bbox(frame, bbox=bbox, labels=label, confidence=conf)

        for (x, y, w, h) in objects:
            cv2.rectangle(
                frame, (x-100, y-100), (x + w + 50, y + h + 50), (255, 255, 255), 4)

            # if x >= 51 and y >= 51 and y >= 51 and y >= 51:
            #     frame = frame[y-50:y+h+50, x-50:w+x+50]
            
            # save_image(frame[y-50:y+h+50, x-50:w+x+50])
            # save_image(frame)
            
            if self.DETECT_EMOTION:
                if (self.COUNT % (self.FPS / 2)) == 0:
                    analyze = DeepFace.analyze(
                        frame, actions=['emotion'], enforce_detection=False)
                    pred = analyze['dominant_emotion']
                    self.PRED = pred

                # Convert the image to RGB (OpenCV uses BGR)  
                cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                
                # Pass the image to PIL  
                pil_im = Image.fromarray(cv2_im_rgb)  
                
                draw = ImageDraw.Draw(pil_im)  
                font = ImageFont.truetype(
                    "/home/tiago/Projetos/python-utils/core/video/Capture/fonts/OpenSansEmoji.ttf", int((h + w) / 2))

                if self.PRED:
                    tick = str(emoji.emojize(
                        str(self.EMOJI_MAP[self.PRED])))
                    draw.text((x, y-20), tick,
                                (255, 255, 255), font=font)

                # Convert back to OpenCV image and save
                frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  
            
        return frame

    def run_stream(self, callback=None):
        if str(self.URL).isnumeric():
            self.CAP = cv2.VideoCapture(self.URL)
        else:
            self.CAP = cv2.VideoCapture(self.URL + ":81/stream")

        if not self.CAP.isOpened():
            print(f'[INFO] Nenhum dispositivo encontrado com o endereço fornecido.')
            return

        self.get_version()
        print(f'[INFO] Iniciando transmissão  com {self.URL} ...')

        try:
            while True:
                self.COUNT += 1

                ret, frame = self.CAP.read()

                if not ret:
                    print("[INFO] Erro na transmissão")
                    break

                if ret:
                    if self.DETECT_OBJ:
                        # frame = self.detect_objects(frame)
                        frame = inference_from_tf_model(frame)
                
                if self.DETECT_MASK:
                    frame = check_masks(frame)

                if self.TIMELAPSE:
                    if (self.COUNT % (self.FPS * self.TIMELAPSE_INTERVAL)) == 0:
                        self.make_image()

                if self.DETECT_MOVIMENT:
                    self.detect_motion(frame)
                

                # cv2.putText(frame, str(
                #     f"{self.FIXED_TEXT} {self.AMBIENT} {self.AMBIENT_STATE}"), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, str(f"{str(datetime.datetime.now().strftime('%d %B %Y %I:%M:%S%p'))}"), (
                #     10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                # display video
                if self.VIDEO:
                    cv2.imshow("Live Video", frame)              

                # aguarda algum comando
                key = cv2.waitKey(1)

                # ajusta a resolução da imagem
                if key == ord('r'):
                    idx = int(input("Select resolution index: "))
                    self.set_resolution(index=idx, verbose=True)

                # ajusta a qualidade da imagem
                elif key == ord('q'):
                    val = int(input("Set quality (10 - 63): "))
                    self.set_quality(value=val)

                # ajusta o balanço de branco
                elif key == ord('a'):
                    self.set_awb()

                # salva imagem
                elif key == ord('s'):
                    self.make_image()

                # turn detection on
                elif key == ord('d'):
                    self.set_detect()

                # turn face emotionon detect
                elif key == ord('e'):
                    self.set_detect_emotion()

                # turn face emotionon detect
                elif key == ord('t'):
                    self.set_timelapse()

                # select detect model
                elif key == ord('n'):
                    self.set_detect_model()

                # select detect model
                elif key == ord('m'):
                    self.set_detect_motion()
                
                # select detect model
                elif key == ord('k'):
                    self.set_detect_mask()

                if key == ord('x'):
                    cv2.destroyAllWindows()
                    self.CAP.release()
                    break

                self.FRAME = frame

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            self.CAP.release()

    def run_web_stream(self, callback=None):
        if str(self.URL).isnumeric():
            self.CAP = cv2.VideoCapture(self.URL)
        else:
            self.CAP = cv2.VideoCapture(self.URL + ":81/stream")

        if not self.CAP.isOpened():
            print(f'[INFO] Nenhum dispositivo encontrado com o endereço fornecido.')
            return

        self.get_version()
        self.set_resolution(index=8)
        print(f'[INFO] Iniciando transmissão  com {self.URL} ...')

        try:
            while True:
                self.COUNT += 1

                ret, frame = self.CAP.read()

                if not ret:
                    print("[INFO] Erro na transmissão")
                    break

                if ret:
                    if self.DETECT_OBJ:
                        frame = self.detect_objects(frame)
                
                if self.DETECT_MASK:
                    frame = check_masks(frame)

                if self.TIMELAPSE:
                    if (self.COUNT % (self.FPS * self.TIMELAPSE_INTERVAL)) == 0:
                        self.make_image()

                if self.DETECT_MOVIMENT:
                    self.detect_motion(frame)
                

                cv2.putText(frame, str(
                    f"{self.FIXED_TEXT} {self.AMBIENT} {self.AMBIENT_STATE}"), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, str(f"{str(datetime.datetime.now().strftime('%d %B %Y %I:%M:%S%p'))}"), (
                    10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                # display video
                if self.VIDEO:
                    cv2.imshow("Live Video", frame)              

                # aguarda algum comando
                key = cv2.waitKey(1)

                self.FRAME = frame
                    
                ret, buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            self.CAP.release()

    def stream(self, url=False, detect_moviment=False, detect_obj=False, detect_emotion=False, callback=None, video=True):
        if url:
            self.URL = url

        if not video:
            self.VIDEO = not self.VIDEO

        if detect_moviment:
            self.DETECT_MOVIMENT = detect_moviment

        if detect_obj:
            self.DETECT_OBJ = detect_obj

        if detect_emotion:
            self.DETECT_EMOTION = detect_emotion
            self.OBJ_DETECTOR = cv2.CascadeClassifier(
                f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')


        self.set_resolution(index=8)

        self.COUNT = 0

        self.run_stream()