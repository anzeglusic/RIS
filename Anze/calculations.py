#!/usr/bin/python3

import numpy as np
import json
import cv2

with open("/home/code8master/Desktop/wsROS/src/RIS/Anze/ROS_videos_data.json") as f:
        data = json.load(f)

# a = np.zeros(2)
# a[0] = 1
# print((a==1)*1)
canStart = False
for key in data:
    if key == "fly_by.mp4":
        canStart = True
    if not canStart:
        continue
    detect = np.zeros((3,len(data[key]["faces_per_frame_dlib"])))
    detect[0] += np.array(data[key]["faces_per_frame_dlib"])
    detect[1] += np.array(data[key]["faces_per_frame_dnn_16_9"])
    detect[2] += np.array(data[key]["faces_per_frame_dnn_1_1"])


    # all = 100
    # only_dlib = 200
    # only_dnn_16_9 = 300
    # only_dnn_1_1 = 400
    # dlib_dnn_16_19 = 500
    # dlib_dnn_1_1 = 600
    # dnn_1_1_dnn_16_9 = 700
    # nobody = 800

    has_face = (detect>0)*1

    detection_label = np.zeros(len(detect[0]))

#     print(f"""
# {key}
#     dlib: {np.sum(has_face[0])}
#     dnn_16_9: {np.sum(has_face[1])}
#     dnn_1_1: {np.sum(has_face[2])}
#     """)
    capture = cv2.VideoCapture(f"/home/code8master/Desktop/wsROS/src/RIS/Anze/analizing/{key}")
    if (capture.isOpened() == False):
        print("Error opening video stream or file")
    
    print(f"{key}")
    for i in range(has_face.shape[1]):
        if [has_face[0][i],has_face[1][i],has_face[2][i]] == [1,1,1]: # all
            detection_label[i] = 100
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [1,0,0]: # only_dlib
            detection_label[i] = 200
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [0,1,0]: # only_dnn_16_9
            detection_label[i] = 300
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [0,0,1]: # only_dnn_1_1
            detection_label[i] = 400
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [1,1,0]: # dlib_dnn_16_19
            detection_label[i] = 500
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [1,0,1]: # dlib_dnn_1_1
            detection_label[i] = 600
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [0,1,1]: # dnn_1_1_dnn_16_9
            detection_label[i] = 700
        elif [has_face[0][i],has_face[1][i],has_face[2][i]] == [0,0,0]: # nobody
            detection_label[i] = 800

    for i in range(len(detection_label)):
        print(f"\n\tcurrent_label: {detection_label[i]}")
        print(f"\tframe: {i}")
        #capture.set(2,400)
        ret, frame = capture.read()
        if detection_label[i]==800:
            continue
        if ret == True:
            cv2.imshow(f"{key}", cv2.resize(frame,(1920,1080)))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                print("Frame read")
                continue
                # break
        else:
            print("Couldn read the frame!")
            continue
            #break
    cv2.destroyAllWindows()

    capture.release()
    
#     print(f"""
# {key}
#     all: {np.sum((detection_label==100)*1)}
#     only_dlib: {np.sum((detection_label==200)*1)}
#     only_dnn_16_9: {np.sum((detection_label==300)*1)}
#     only_dnn_1_1: {np.sum((detection_label==400)*1)}
#     dlib_dnn_16_19: {np.sum((detection_label==500)*1)}
#     dlib_dnn_1_1: {np.sum((detection_label==600)*1)}
#     dnn_1_1_dnn_16_9: {np.sum((detection_label==700)*1)}
#     nobody: {np.sum((detection_label==800)*1)}""")
    


    