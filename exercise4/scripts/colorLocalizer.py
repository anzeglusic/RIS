#!/usr/bin/python3

import sys
import rospy
import dlib
import os
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Twist
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class color_localizer:

    def __init__(self):
        rospy.init_node('color_localizer', anonymous=True)
        
        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the image and/or depth topic
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        # self.markers_pub = rospy.Publisher('face_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('face_im', Image, queue_size=1000)
        #self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        # self.twist_pub = rospy.Publisher('/our_pub1/chat1', Twist, queue_size=1000)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        
        #notri hranimo "priblizen center slike" pod katerim je mnzoica 100 ter normala stene na kateri je
        #key so stevilke znotraj imamo pa prvo povprecje vseh tock in nato se vse tocke (np.array) shranjene v seznamu
        # self.detected_pos_fin = {}
        # self.detected_norm_fin = {}
        # self.entries = 0
        # self.range = 0.2

    def find_color(self,image):
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_low = np.array([136, 87, 11], np.uint8)
        red_up = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvIm, red_low, red_up)

        red_low1 = np.array([0, 87, 11], np.uint8)
        red_up1 = np.array([18, 255, 255], np.uint8)
        red_mask1 = cv2.inRange(hsvIm, red_low1, red_up1)

        green_low = np.array([25,52,72], np.uint8)
        green_up = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvIm, green_low, green_up)

        blue_low = np.array([94,80,2], np.uint8)
        blue_up = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvIm, blue_low, blue_up)


        black_low = np.array([0,0,0], np.uint8)
        black_up = np.array([180, 255, 40], np.uint8)
        black_mask = cv2.inRange(hsvIm, black_low, black_up)

        kernel = np.ones((30,30), "uint8")
        kernel1 = np.ones((7,7), "uint8")

        # R
        red_mask = cv2.dilate(red_mask, kernel)
        red_mask = cv2.erode(red_mask, kernel)
        red_mask = cv2.erode(red_mask, kernel1)
        red_mask = cv2.dilate(red_mask, kernel1)
        # red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(image, image, mask = red_mask)

        # R
        red_mask1 = cv2.dilate(red_mask1, kernel)
        red_mask1 = cv2.erode(red_mask1, kernel)
        red_mask1 = cv2.erode(red_mask1, kernel1)
        red_mask1 = cv2.dilate(red_mask1, kernel1)
        # red_mask1 = cv2.dilate(red_mask1, kernel)
        res_red1 = cv2.bitwise_and(image, image, mask = red_mask1)

        # G
        green_mask = cv2.dilate(green_mask, kernel)
        green_mask = cv2.erode(green_mask, kernel)
        green_mask = cv2.erode(green_mask, kernel1)
        green_mask = cv2.dilate(green_mask, kernel1)
        # green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(image, image, mask = green_mask)

        # B
        blue_mask = cv2.dilate(blue_mask, kernel)
        blue_mask = cv2.erode(blue_mask, kernel)
        blue_mask = cv2.erode(blue_mask, kernel1)
        blue_mask = cv2.dilate(blue_mask, kernel1)
        # blue_mask = cv2.dilate(blue_mask, kernel)
        res_blue = cv2.bitwise_and(image, image, mask = blue_mask)

        # black
        black_mask = cv2.dilate(black_mask, kernel)
        black_mask = cv2.erode(black_mask, kernel)
        black_mask = cv2.erode(black_mask, kernel1)
        black_mask = cv2.dilate(black_mask, kernel1)
        # black_mask = cv2.dilate(black_mask, kernel)
        res_black = cv2.bitwise_and(image, image, mask = black_mask)

        final_mask = cv2.bitwise_or(red_mask,red_mask1)
        final_mask = cv2.bitwise_or(final_mask,green_mask)
        final_mask = cv2.bitwise_or(final_mask,blue_mask)
        final_mask = cv2.bitwise_or(final_mask,black_mask)
        # final_mask = cv2.bitwise_and(final_mask,bitMaskForAnd)

        # final_mask = black_mask

        contour, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cont in enumerate(contour):
            area = cv2.contourArea(cont)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(cont)
                image = cv2.rectangle(image, (x, y),(x+w,y+h), (0,255,255),2)
        
        """
        contour, hierarchy = cv2.findContours(red_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cont in enumerate(contour):
            area = cv2.contourArea(cont)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(cont)
                image = cv2.rectangle(image, (x, y),(x+w,y+h), (0,255,0),2)
        """
        return image
        # return cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)

    def find_objects(self):
        print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0


        # Convert the images into a OpenCV (numpy) format
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        markedImage = self.find_color(rgb_image)


        self.pic_pub.publish(CvBridge().cv2_to_imgmsg(markedImage, encoding="passthrough"))




















def main():

        color_finder = color_localizer()

        rate = rospy.Rate(1.25)
        while not rospy.is_shutdown():
            #print("hello!")
            color_finder.find_objects()
            #print("hello")
            print("\n-----------\n")
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()