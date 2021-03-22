#!/usr/bin/python3

import numpy as np
import cv2
import dlib

def playVideo(path, fast=False):
    capture = cv2.VideoCapture(path)
    sleepTimer = 1

    if fast:
        sleepTimer = 1
    else:
        print(capture.get(cv2.CAP_PROP_FPS))
        sleepTimer = int(1000.0/capture.get(cv2.CAP_PROP_FPS))
        if sleepTimer < 1:
            sleepTimer = 1

    if (capture.isOpened() == False):
        print("Error opening video stream or file")

    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret == True:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(sleepTimer) & 0xFF == ord("q"):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


def getFrames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
        print("Wait for the header")
    num_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while cap.isOpened():
        print(
            f"Capturing video [{int(pos_frame/num_of_frames*100)}%]: {int(pos_frame)}/{int(num_of_frames)}", end="\r")
        flag, frame = cap.read()  # get the frame
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)

            # store the current frame in as a numpy array
            frames.append(frame)

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            print("Video captured.", " "*100)
            break
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    cap.release()
    cv2.destroyAllWindows()
    return (frames, fps, fourcc)


def check_for_face_dnn_16_9(frame, thiknes, prototxtPath, caffePath):
    face_net = cv2.dnn.readNetFromCaffe(
        prototxtPath,
        caffePath)

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

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Extract region containing face
            border_color = np.array([0, 255, 0])
            rgb_image[(y1-thiknes):y1, (x1-thiknes):(x2+thiknes+1)] = border_color  # up
            rgb_image[(y2+1):(y2+thiknes+2), (x1-thiknes):(x2+thiknes+1)] = border_color  # down
            rgb_image[(y1-thiknes):(y2+thiknes+2),
                      (x1-thiknes):x1] = border_color  # left
            rgb_image[(y1-thiknes):(y2+thiknes+2), (x2+1)                      :(x2+thiknes+2)] = border_color  # right

    return rgb_image

def check_for_face_dnn_1_1(frame, thiknes, prototxtPath, caffePath, debug):
    face_net = cv2.dnn.readNetFromCaffe(
        prototxtPath,
        caffePath)

    # A help variable for holding the dimensions of the image
    dims = (0, 0, 0)

    # Get the next rgb and depth images that are posted from the camera
    rgb_image_message = frame

    # Convert the images into a OpenCV (numpy) format
    rgb_image = rgb_image_message

    rgb_image_original = np.copy(rgb_image)


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
    if debug:
        print(f"original {rgb_image_original.shape}")
        print(f"left     {rgb_image_left.shape}")
        print(f"middle   {rgb_image_middle.shape}")
        print(f"right    {rgb_image_right.shape}")
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

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Mark left region
            border_color = np.array([0, 0, 255])
            rgb_image_original[(y1-thiknes):y1, (x1-thiknes):(x2+thiknes+1)] = border_color  # up
            rgb_image_original[(y2+1):(y2+thiknes+2), (x1-thiknes):(x2+thiknes+1)] = border_color  # down
            rgb_image_original[(y1-thiknes):(y2+thiknes+2),(x1-thiknes):x1] = border_color  # left
            rgb_image_original[(y1-thiknes):(y2+thiknes+2), (x2+1):(x2+thiknes+2)] = border_color  # right
            
    
    face_net.setInput(blob_middle)
    face_detections = face_net.forward()

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Mark middle region
            border_color = np.array([0, 255, 0])
            rgb_image_original[(y1-thiknes):y1, (shift_pixels+x1-thiknes):(shift_pixels+x2+thiknes+1)] = border_color  # up
            rgb_image_original[(y2+1):(y2+thiknes+2), (shift_pixels+x1-thiknes):(shift_pixels+x2+thiknes+1)] = border_color  # down
            rgb_image_original[(y1-thiknes):(y2+thiknes+2),(shift_pixels+x1-thiknes):shift_pixels+x1] = border_color  # left
            rgb_image_original[(y1-thiknes):(y2+thiknes+2), (shift_pixels+x2+1):(shift_pixels+x2+thiknes+2)] = border_color  # right
            
    
    face_net.setInput(blob_right)
    face_detections = face_net.forward()

    shift_pixels = rgb_image_original.shape[1] - rgb_image_original.shape[0]
    if debug:
        print(f"to shift right {shift_pixels}")

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Mark right region
            border_color = np.array([255, 0, 0])
            rgb_image_original[(y1-thiknes):y1, (shift_pixels+x1-thiknes):(shift_pixels+x2+thiknes+1)] = border_color  # up
            rgb_image_original[(y2+1):(y2+thiknes+2), (shift_pixels+x1-thiknes):(shift_pixels+x2+thiknes+1)] = border_color  # down
            rgb_image_original[(y1-thiknes):(y2+thiknes+2),(shift_pixels+x1-thiknes):shift_pixels+x1] = border_color  # left
            rgb_image_original[(y1-thiknes):(y2+thiknes+2), (shift_pixels+x2+1):(shift_pixels+x2+thiknes+2)] = border_color  # right
    
    return rgb_image_original

def check_for_face_dlib(frame, thiknes):
    face_detector = dlib.get_frontal_face_detector()

    # A help variable for holding the dimensions of the image
    dims = (0, 0, 0)

    # Get the next rgb and depth images that are posted from the camera
    rgb_image_message = frame

    # Convert the images into a OpenCV (numpy) format
    rgb_image = rgb_image_message

    # Set the dimensions of the image
    dims = rgb_image.shape

    face_rectangles = face_detector(rgb_image, 0)

    for face_rectangle in face_rectangles:
        # The coordinates of the rectanle
        x1 = face_rectangle.left()
        x2 = face_rectangle.right()
        y1 = face_rectangle.top()
        y2 = face_rectangle.bottom()

        # Extract region containing face
        border_color = np.array([0, 255, 0])
        rgb_image[(y1-thiknes):y1, (x1-thiknes):(x2+thiknes+1)] = border_color  # up
        rgb_image[(y2+1):(y2+thiknes+2), (x1-thiknes):(x2+thiknes+1)] = border_color  # down
        rgb_image[(y1-thiknes):(y2+thiknes+2), (x1-thiknes):x1] = border_color  # left
        rgb_image[(y1-thiknes):(y2+thiknes+2), (x2+1):(x2+thiknes+2)] = border_color  # right

    return rgb_image


def playFrames(frames, fps):
    sleepTimer = int(1000.0/fps)
    if sleepTimer < 1:
        sleepTimer = 1

    for frame in frames:
        cv2.imshow("Video", frame)
        if cv2.waitKey(sleepTimer) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def analizeFrames_dnn_16_9(frames, prototxtPath, caffePath, thiknes=5):
    num_of_frames = len(frames)
    for i in range(num_of_frames):
        print(
            f"Analizing frames [{int(i/num_of_frames*100)}%]: {int(i)}/{int(num_of_frames)}", end="\r")
        frames[i] = check_for_face_dnn_16_9(
            frames[i], thiknes, prototxtPath, caffePath)
    print("Frames analized.", " "*100)
    return frames

def analizeFrames_dnn_1_1(frames, prototxtPath, caffePath, thiknes=5):
    num_of_frames = len(frames)
    for i in range(num_of_frames):
        print(
            f"Analizing frames [{int(i/num_of_frames*100)}%]: {int(i)}/{int(num_of_frames)}", end="\r")
        frames[i] = check_for_face_dnn_1_1(
            frames[i], thiknes, prototxtPath, caffePath,debug=(i==-1))
    print("Frames analized.", " "*100)
    return frames

def analizeFrames_dlib(frames, thiknes=5):
    num_of_frames = len(frames)
    for i in range(num_of_frames):
        print(
            f"Analizing frames [{int(i/num_of_frames*100)}%]: {int(i)}/{int(num_of_frames)}", end="\r")
        frames[i] = check_for_face_dlib(frames[i], thiknes)
    print("Frames analized.", " "*100)
    return frames




def saveVideo(path, frames, fps, fourcc):
    (height, width, channels) = frames[0].shape
    out = cv2.VideoWriter(path, int(fourcc), fps, (width, height))

    num_of_frames = len(frames)
    for i in range(num_of_frames):
        print(
            f"Saving video [{int(i/num_of_frames*100)}%]: {int(i)}/{int(num_of_frames)}", end="\r")
        out.write(frames[i])
    print("Video saved.", " "*100)
    out.release()


# =================================================================================================
# =================================================================================================
prototxtPath = "/home/code8master/Desktop/wsROS/src/video-testing/deploy.prototxt.txt"
caffePath = "/home/code8master/Desktop/wsROS/src/video-testing/res10_300x300_ssd_iter_140000.caffemodel"
for video_num in range(1, 14):
# for video_num in [11]:
    print(f"---------------- {video_num}.mp4 ----------------")
    path_in = f"/home/code8master/Desktop/wsROS/src/video-testing/videos/{video_num}.mp4"
    path_out= f"/home/code8master/Desktop/wsROS/src/video-testing/videos_analized_dnn_1_1_x3/{video_num}.mp4"
    (frames, fps, fourcc) = getFrames(path=path_in)
    frames_marked = analizeFrames_dnn_1_1(frames=frames, caffePath=caffePath, prototxtPath=prototxtPath)
    # playFrames(frames,fps)
    saveVideo(path = path_out, frames = frames_marked, fps = fps, fourcc = fourcc)

cv2.destroyAllWindows()


# (1080, 1920, 3)