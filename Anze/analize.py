#!/usr/bin/python3

import numpy as np
import cv2
import dlib
import json


# def check_for_face_dnn_16_9(frame, thiknes, prototxtPath, caffePath):
def check_for_face_dnn_16_9_counter(frame, thiknes, face_net):
    # face_net = cv2.dnn.readNetFromCaffe(
    #     prototxtPath,
    #     caffePath)

    # A help variable for holding the dimensions of the image
    dims = (0, 0, 0)

    # Get the next rgb and depth images that are posted from the camera
    rgb_image_message = frame

    # Convert the images into a OpenCV (numpy) format
    rgb_image = rgb_image_message

    # Set the dimensions of the image
    dims = rgb_image.shape
    h = dims[0]
    w = dims[1]
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(
            src=rgb_image,
            dsize=(300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0))

    face_net.setInput(blob)
    face_detections = face_net.forward()

    counter = 0
    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:
            counter += 1
            
    return counter

# def check_for_face_dnn_1_1(frame, thiknes, prototxtPath, caffePath, debug):
def check_for_face_dnn_1_1_counter(frame, thiknes, debug, face_net):
    # face_net = cv2.dnn.readNetFromCaffe(
    #     prototxtPath,
    #     caffePath)

    # A help variable for holding the dimensions of the image
    dims = (0, 0, 0)

    # Get the next rgb and depth images that are posted from the camera
    rgb_image_message = frame

    # Convert the images into a OpenCV (numpy) format
    rgb_image = rgb_image_message


    # Set the dimensions of the image
    dims = rgb_image.shape
    if debug:
        print(rgb_image.shape)

    if dims[0] < dims[1]:
        num_of_dead_pixels = dims[1]-dims[0]
        shift_pixels = num_of_dead_pixels//2
        rgb_image_left = rgb_image[:,0:dims[0],:]
        rgb_image_middle = rgb_image[:,shift_pixels:(shift_pixels+dims[0]),:]
        rgb_image_right = rgb_image[:,(dims[1]-dims[0]):,:]
    else:
        print("\n\nSpodnja stranica bi morala biti daljša !!!!\n\n")
    
    dims = rgb_image_left.shape
    h = dims[0]
    w = dims[1]
    
    blob_left = cv2.dnn.blobFromImage(
        image=cv2.resize(
            src=rgb_image_left,
            dsize=(300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0))

    blob_middle = cv2.dnn.blobFromImage(
        image=cv2.resize(
            src=rgb_image_middle,
            dsize=(300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0))

    blob_right = cv2.dnn.blobFromImage(
        image=cv2.resize(
            src=rgb_image_right,
            dsize=(300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0))
    
    face_net.setInput(blob_left)
    face_detections = face_net.forward()

    counter = 0

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:
            counter += 1
            
    
    face_net.setInput(blob_middle)
    face_detections = face_net.forward()

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:
            counter += 1
            
    
    face_net.setInput(blob_right)
    face_detections = face_net.forward()

    if debug:
        print(f"to shift right {shift_pixels}")

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:
            counter += 1
    
    return counter

def check_for_face_dlib_counter(frame, thiknes, face_detector):
    # face_detector = dlib.get_frontal_face_detector()

    # A help variable for holding the dimensions of the image
    dims = (0, 0, 0)

    # Get the next rgb and depth images that are posted from the camera
    rgb_image_message = frame

    # Convert the images into a OpenCV (numpy) format
    rgb_image = rgb_image_message

    # Set the dimensions of the image
    dims = rgb_image.shape

    face_rectangles = face_detector(rgb_image, 0)

    counter = 0

    for face_rectangle in face_rectangles:
        counter += 1
        

    return counter




# =================================================================================================
# =================================================================================================

video_names = ["15_deg","30_deg","45_deg","60_deg", "75_deg", "90_deg","fly_by", "motion_blurr"]

video_names_first_batch = range(1,14)

prototxtPath = "/home/code8master/Desktop/wsROS/src/RIS/Anze/deploy.prototxt.txt"
caffePath = "/home/code8master/Desktop/wsROS/src/RIS/Anze/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(prototxtPath,caffePath)
face_dlib = dlib.get_frontal_face_detector()

#for video_name in video_names:
#for video_name in ["15_deg"]:
for video_name in video_names_first_batch:
    print(f"---------------- {video_name}.mp4 ----------------")
    path_clean = f"/home/code8master/Desktop/wsROS/src/RIS/Anze/videos/{video_name}.mp4"

    # path_dlib = f"/home/code8master/Desktop/wsROS/src/personal/video-testing/videos_analized_dlib/{video_name}.mp4"
    # path_dnn_1_1 = f"/home/code8master/Desktop/wsROS/src/personal/video-testing/videos_analized_dnn_1_1_x3/{video_name}.mp4"
    # path_dnn_16_9 = f"/home/code8master/Desktop/wsROS/src/personal/video-testing/videos_analized_dnn_16_9_1x/{video_name}.mp4"

    #path_save = f"/home/code8master/Desktop/wsROS/src/RIS/Anze/analizing_ROS/{video_name}.mp4"

    cap_clean = cv2.VideoCapture(path_clean)
    # cap_dlib = cv2.VideoCapture(path_dlib)
    # cap_dnn_1_1 = cv2.VideoCapture(path_dnn_1_1)
    # cap_dnn_16_9 = cv2.VideoCapture(path_dnn_16_9)


    num_of_frames = cap_clean.get(cv2.CAP_PROP_FRAME_COUNT)
    faces_per_frame_dlib = np.zeros(int(num_of_frames))
    faces_per_frame_dnn_1_1 = np.zeros(int(num_of_frames))
    faces_per_frame_dnn_16_9 = np.zeros(int(num_of_frames))

    pos_frame = int(cap_clean.get(cv2.CAP_PROP_POS_FRAMES))

    while cap_clean.isOpened():
        print(f"Capturing video [{int(pos_frame/num_of_frames*100)}%]: {int(pos_frame)}/{int(num_of_frames)}", end="\r")
        flag_clean, frame_clean = cap_clean.read()  # get the frame
        # flag_dlib, frame_dlib = cap_dlib.read()  # get the frame
        # flag_dnn_1_1, frame_dnn_1_1 = cap_dnn_1_1.read()  # get the frame
        # flag_dnn_16_9, frame_dnn_16_9 = cap_dnn_16_9.read()  # get the frame

        flag_dlib = True
        flag_dnn_1_1 = True
        flag_dnn_16_9 = True
        
        frame_dlib_counter = check_for_face_dlib_counter(frame=np.copy(frame_clean), thiknes=5, face_detector=face_dlib)
        frame_dnn_1_1_counter = check_for_face_dnn_1_1_counter(frame=np.copy(frame_clean), thiknes=5, debug=False, face_net = face_net)
        frame_dnn_16_9_counter = check_for_face_dnn_16_9_counter(frame=np.copy(frame_clean), thiknes=5, face_net=face_net)

        faces_per_frame_dlib[pos_frame] = frame_dlib_counter
        faces_per_frame_dnn_16_9[pos_frame] = frame_dnn_16_9_counter
        faces_per_frame_dnn_1_1[pos_frame] = frame_dnn_1_1_counter

        # print(f"\t{frame_clean.shape}")
        # print(f"\t{frame_dlib.shape}")
        # print(f"\t{frame_dnn_1_1.shape}")
        # print(f"\t{frame_dnn_16_9.shape}")
        
        if flag_clean and flag_dlib and flag_dnn_1_1 and flag_dnn_16_9:
            # The frame is ready and already captured

            pos_frame = int(cap_clean.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # The next frame is not ready, so we try to read it again
            cap_clean.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap_clean.get(cv2.CAP_PROP_POS_FRAMES) == cap_clean.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            print("Video captured.", " "*100)
            break
    faces_per_frame_dlib = list(faces_per_frame_dlib)
    faces_per_frame_dnn_16_9 = list(faces_per_frame_dnn_16_9)
    faces_per_frame_dnn_1_1 = list(faces_per_frame_dnn_1_1)

    video_json = {}
    video_json["faces_per_frame_dlib"] = faces_per_frame_dlib
    video_json["faces_per_frame_dnn_16_9"] = faces_per_frame_dnn_16_9
    video_json["faces_per_frame_dnn_1_1"] = faces_per_frame_dnn_1_1

    # print(video_json)
    # print(type(video_json))
     
    # print(json.dumps(str(video_json)))
    # print(type(json.dumps(str(video_json))))

    with open("/home/code8master/Desktop/wsROS/src/RIS/Anze/analizing_data.json") as f:
        data = json.load(f)
        
    data[f"{video_name}.mp4"] = video_json
    #!with open("/home/code8master/Desktop/wsROS/src/RIS/Anze/analizing_data.json","w") as f:
    #!    json.dump(data,f)

    # print(faces_per_frame_dlib)
    # print(faces_per_frame_dnn_16_9)
    # print(faces_per_frame_dnn_1_1)
    cap_clean.release()
    #cap_dlib.release()
    #cap_dnn_1_1.release()
    #cap_dnn_16_9.release()


    cv2.destroyAllWindows()


# (1080, 1920, 3)