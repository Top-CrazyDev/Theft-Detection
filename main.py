from flask import Flask, request, redirect, flash, render_template,send_file , Response , jsonify
import json
import mediapipe as mp
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from twilio.rest import Client
import math
import threading

limitTime = 0
account_sid = 'AC97edb68096f012e6c804fe5acc120962'
auth_token = '9f33956e4727f39b36beaa3c14d16449'
client = Client(account_sid, auth_token)

def WhatsappMessage( content ):
    message = client.messages.create(
        body = content,
        from_ = "whatsapp:+14155238886",
        to = "whatsapp:+12026979993"
    )

def PhoneMessage( content ):
    message = client.messages.create(
    body = content,
    from_ = '+18336981804',
    to = '+14088575574'
    )
    
def Camera():
    cap = cv2.VideoCapture("theft.mp4")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            ret2, buff = cv2.imencode('.jpg', frame)
            if not ret2:
                break
            frame = buff.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture("theft.mp4")
    model = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )
    frame_count = 0
    theft_flag = False
    theftman = False
    theft_frame_count = 0
    lie_detect_count = 0
    near_detect_count = 0
    near_detect_flag = False
    lie_detect_flag = False
    while(cap.isOpened()):
        frame_count += 1
        yellow_flag = False
        red_flag = False
        ret, frame = cap.read()
        if ret:
            if frame_count % 6 == 0:
                frame_count = 0
                # theft detection using Yolov8
                result = model(frame, agnostic_nms=True)[0]
                detections = sv.Detections.from_ultralytics(result)
                #calculate IoU
                for detection in detections:
                    IoU = 0
                    x1 = detection[0][0]
                    y1 = detection[0][1]
                    x2 = detection[0][2]
                    y2 = detection[0][3]
                    for detection2 in detections:
                        isequal = False
                        if detection[0].all() != detection2[0].all() or detection[2] != detection2[2] or detection[3] != detection2[3]:
                            x3 = detection2[0][0]
                            y3 = detection2[0][1]
                            x4 = detection2[0][2]
                            y4 = detection2[0][3]
                            x_intersection = max(x1, x3)
                            y_intersection = max(y1, y3)
                            x_intersection_end = min(x2, x4)
                            y_intersection_end = min(y2, y4)
                            width_intersection = x_intersection_end - x_intersection
                            height_intersection = y_intersection_end - y_intersection
                            area_A = (x2 - x1) * (y2 - y1)
                            area_B = (x4 - x3) * (y4 - y3)
                            if (area_A + area_B - width_intersection * height_intersection) != 0:
                                IoU = (width_intersection * height_intersection) / (area_A + area_B - width_intersection * height_intersection)
                            if IoU > 0.5:
                                if area_A < area_B:
                                    detections = tuple(item for item in detections if item[0].all() != detection2[0].all() or item[2] != detection2[2] or item[3] != detection2[3])
                                elif area_B > area_A:
                                    detections = tuple(item for item in detections if item[0].all() != detection[0].all() or item[2] != detection[2] or item[3] != detection[3])
                limit_person = 0;
                persons = []
                flag = False
                max_car = []
                # main car detection
                for detection in detections:
                    if(model.names[detection[3]] == "car"):
                        if flag == True:
                            if (detection[0][2] - detection[0][0]) > (max_car[0][2] - max_car[0][0]):
                                max_car = detection
                        if flag == False:
                            max_car = detection
                            flag = True
                    if(model.names[detection[3]] == "person"):
                        limit_person += 1
                        persons.append(detection)
                print(theftman , theft_flag)
                theft_flag = False
                if persons == [] and theftman == False:
                    theft_flag = False
                    theft_frame_count = 0
                    
                if (persons == [] and theftman == True) or (persons != [] and theftman == True):
                    theft_flag = True
                    theft_frame_count += 1
                theftman = False
                if(max_car == []):
                    continue
                topleft = tuple([int(max_car[0][0]) , int(max_car[0][1])])
                bottomright = tuple([int(max_car[0][2]) , int(max_car[0][3])])
                car_pos = ([topleft , bottomright])
                car_x = int((car_pos[0][0] + car_pos[1][0]) / 2)
                car_y = int((car_pos[0][1] + car_pos[1][1]) / 2)
                if persons != []:
                    for item in persons:
                        if item == []:
                            continue
                        if (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]) > (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]):
                            item = item
                        topleft = tuple([int(item[0][0]) , int(item[0][1])])
                        bottomright = tuple([int(item[0][2]) , int(item[0][3])])
                        person_x = int((item[0][0] + item[0][2]) / 2)
                        person_y = int((item[0][1] + item[0][3]) / 2)
                        if (abs(item[0][2] - item[0][0]) * abs(item[0][3] - item[0][1])) > (abs(car_pos[0][0] - car_pos[1][0]) * abs(car_pos[0][1] - car_pos[1][1]) / 20) and abs(person_x - car_x) < (abs((car_pos[1][0] - car_pos[0][0]) / 2) + abs((item[0][2] - item[0][0]) / 2)) and abs(person_y - car_y) < (abs((car_pos[1][1] - car_pos[0][1])/2) + abs((item[0][3] - item[0][1]) / 2)):
                            theftman = True
                
                if theft_frame_count >= (limitTime * 29 / 6) :
                    if near_detect_flag == False:
                        near_detect_flag = True
                        # Create two threads
                        thread1 = threading.Thread(target=WhatsappMessage , args = ("Hi, I am Kevin. Theft is detected. Catch them!!!" , ))
                        # thread2 = threading.Thread(target=PhoneMessage)
                        # Start the threads
                        thread1.start()
                        red_flag = True
                        # thread2.start()
                if near_detect_flag == True:
                    frame = cv2.circle(frame , (50,50) , 50 , (0 , 0 , 255) , thickness = cv2.FILLED)
                    near_detect_count += 1
                if near_detect_count >= int(300 * 29 / 5):
                    near_detect_flag = False
                #theft detection using Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                landmarks = {}
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    i = 0
                    h, w, c = frame.shape
                    lf_pos = []
                    rf_pos = []
                    n_pos = []
                    for lm in landmarks:
                        i += 1
                        x, y = int(lm.x * w), int(lm.y * h)
                        if i == 1:
                            n_pos = lm
                        if i == 33 :
                            lf_pos = lm
                        if i == 32 : 
                            rf_pos = lm 
                
                    mf_pos = tuple([int(((lf_pos.x * w) + (rf_pos.x * w)) / 2) , int(((lf_pos.y * h) + (rf_pos.y * h)) / 2)])
                    head_pos = tuple([int(n_pos.x * w) , int(n_pos.y * h)])
                    center_pos = (int((mf_pos[0]+head_pos[0]) / 2) , int((mf_pos[1]+head_pos[1]) / 2))
                    if abs(center_pos[0] - car_x) < (abs((car_pos[1][0] - car_pos[0][0])/2) + 100) and abs(center_pos[1] - car_y) < (abs((car_pos[1][1] - car_pos[0][1]) / 2) + 100):
                        x_ang = abs(head_pos[0] - mf_pos[0])
                        y_ang = abs(head_pos[1] - mf_pos[1])
        
                        if y_ang < 0 or (x_ang != 0 and (y_ang / x_ang)) < 3:
                            if lie_detect_flag == False:
                                lie_detect_flag = True
                                # Create two threads
                                thread3 = threading.Thread(target=WhatsappMessage , args = ("Hi, I am Kevin. Someone is looking for Converter. Your car is in danger!!!" , ))
                                # thread4 = threading.Thread(target=PhoneMessage)
                                # Start the threads
                                thread3.start()
                                yellow_flag = True
                                # thread4.start()
                            theft_frame_count += 1
                if lie_detect_flag == True:
                    if near_detect_flag == False:
                        frame = cv2.circle(frame , (50,50) , 50 , (0 , 255 , 255) , thickness = cv2.FILLED)   
                    lie_detect_count += 1
                if lie_detect_count >= int(300 * 29 / 5):
                    lie_detect_flag = False
                ret2, buff = cv2.imencode('.jpg', frame)
                if not ret2:
                    break
                frame = buff.tobytes()
                if yellow_flag == True:
                    message = "warn message sent"
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' +
                        b'Content-Type: text/plain\r\n\r\n' + message.encode() + b'\r\n')
                elif red_flag == True:
                    message = "risk message sent"
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' +
                        b'Content-Type: text/plain\r\n\r\n' + message.encode() + b'\r\n')
                else :
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(Camera() , mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/theft_detection')
def theft_detection():
    return Response(main() , mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/detectionStart', methods = ["POST"])
def detectionStart():
    data = request.get_json()
    global limitTime
    limit = data.get("limit")
    limitTime = int(limit)
    if limitTime == 0:
        return "It will detect always!"
    return "ok"
    

if __name__ == "__main__":
    app.run()