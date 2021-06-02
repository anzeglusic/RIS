#!/usr/bin/python3

import os
import itertools
import sys
import getch

from numpy.lib.function_base import _CORE_DIMENSION_LIST
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/')
import rospy
import dlib
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from pprint import pprint
#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Twist
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import pickle
import subprocess
import pyzbar.pyzbar as pyzbar
import pytesseract
import module
from datetime import datetime

modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'
mouth = modelsDir+"mouth/Mouth.xml"

class ring_maker:

    def __init__(self):
        print()
        self.loopTimer = datetime.now()
        self.rgbTimer = 0
        self.depthTimer = 0

        self.foundAll = True

        self.basePosition = {"x":0, "y":0, "z":0}
        self.baseAngularSpeed = 0

        self.random_forest_HSV = pickle.load(open(f"{modelsDir}/random_forest_HSV.sav", 'rb'))
        self.random_forest_RGB = pickle.load(open(f"{modelsDir}/random_forest_RGB.sav", 'rb'))
        self.knn_HSV = pickle.load(open(f"{modelsDir}/knn_HSV.sav", 'rb'))
        self.knn_RGB = pickle.load(open(f"{modelsDir}/knn_RGB.sav", 'rb'))
        #self.mouth_finder = cv2.CascadeClassifier(mouth)
        #if self.mouth_finder.empty():
        #    raise IOError("no mouth detector")

        self.positions = {
            "ring": [],
            "cylinder": [],
            "face": [],
            "QR": [],
            "digits": []
        }

        rospy.init_node('color_localizer', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.m_arr = MarkerArray()
        self.nM = 0
        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('/ring_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('/face_im', Image, queue_size=1000)
        self.faceIm_pub = rospy.Publisher('/face_im2', Image, queue_size=1000)
        self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        self.twist_pub = rospy.Publisher('/our_pub1/chat2', Twist, queue_size=1000)
        #ta dva se uporabljata za objavo
        self.face_pub = rospy.Publisher('/face_tw', Twist, queue_size=1000)
        self.cylinder_pub = rospy.Publisher('/cylinder_pt', Point, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        #notri hranimo "priblizen center slike" pod katerim je mnzoica 100 ter normala stene na kateri je
        #key so stevilke znotraj imamo pa prvo povprecje vseh tock in nato se vse tocke (np.array) shranjene v seznamu
        # self.detected_pos_fin = {}
        # self.detected_norm_fin = {}
        # self.entries = 0
        # self.range = 0.2

        #! togle za sive markerje
        self.showEveryDetection = True

        #! digits detection start
        self.dictm = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.params =  cv2.aruco.DetectorParameters_create()
        self.params.adaptiveThreshConstant = 25 # TODO: fine-tune for best performance
        self.params.adaptiveThreshWinSizeStep = 2 # TODO: fine-tune for best performance

        print(f"   adaptiveThreshConstant: {self.params.adaptiveThreshConstant}")
        print(f" adaptiveThreshWinSizeMax: {self.params.adaptiveThreshWinSizeMax}")
        print(f" adaptiveThreshWinSizeMin: {self.params.adaptiveThreshWinSizeMin}")
        print(f"    minCornerDistanceRate: {self.params.minCornerDistanceRate}")
        print(f"adaptiveThreshWinSizeStep: {self.params.adaptiveThreshWinSizeStep}")
        print()
        #! digits detection end

        self.qrNormalLength = 0.25

        self.QR_to_fullData = {
                    "This QR code is on a cylinder.": None,
                    "This is a different QR code on a cylinder.": None,
                    "A third QR code on a cylinder.": None,
                    "The last QR code on a cylinder.": None,

                    "A QR code next to a face. Example 1.": None,
                    "Another QR code next to a face. Example 2.": None,
                    "One more QR code next to a face. Example 3.": None,
                    "The last QR code next to a face. Example 4.": None
        }

        #! face detection start
        self.faceNormalLength = 0.5
        self.marker_array = MarkerArray()
        self.marker_num = 1
        self.detected_pos_fin = {}
        self.detected_norm_fin = {}
        self.entries = 0
        self.range = 0.2
        self.face_net = cv2.dnn.readNetFromCaffe(os.path.dirname(os.path.abspath(__file__))+'/deploy.prototxt.txt', os.path.dirname(os.path.abspath(__file__))+'/res10_300x300_ssd_iter_140000.caffemodel')
        self.dims = (0, 0, 0)
        #! face detection end

        #! cylinder
        #element: (interval,vrstic_od_sredine)
        self.tru_intervals = []

    def chk_ring(self,de,h1,h2,w1,w2,cent):
        #depoth im je for some reason 480x640
        #sprememeba 1: dodal sem da shrani se koordinate v sliki notr MOZNO DA JIH BO TREBA OBRNT
        cmb_me = []
        if(h1[0]>=0 and h1[1]>=0 and h1[0]<640 and h1[1]<480 and de[h1[1],h1[0]]):
            cmb_me.append((de[h1[1],h1[0]],h1[1],h1[0])) #koordinate v sliki

        if(h2[0]>=0 and h2[1]>=0 and h2[0]<640 and h2[1]<480 and de[h2[1],h2[0]]):
            cmb_me.append((de[h2[1],h2[0]],h2[1],h2[0]))

        if(w1[0]>=0 and w1[1]>=0 and w1[0]<640 and w1[1]<480 and de[w1[1],w1[0]]):
            cmb_me.append((de[w1[1],w1[0]],w1[1],w1[0]))

        if(w2[0]>=0 and w2[1]>=0 and w2[0]<640 and w2[1]<480 and de[w2[1],w2[0]]):
            cmb_me.append((de[w2[1],w2[0]],w2[1],w2[0]))
        if len(cmb_me)<3:
            #print("garbo")
            return None
        points = []
        for i in itertools.combinations(cmb_me,3):
            #preverimo ce so vse 3 točke manj kto 20cm narazen(to je največji možn diameter kroga
            good = True
            # print(f"--> {i}")
            for j in itertools.combinations(i,2):
                """EDIT NI NUJNO DA JE TAKO DELOVANJE KOMBINACIJ """
                if np.abs(j[0][0] - j[1][0])>0.2:
                    good = False
                    break
            if good:
                #print("we good")
                """avgdist = sum(i)/3 ne bo delal na trenutni iteraciji"""
                avgdist = (i[0][0] + i[1][0] + i[2][0])/3


                # print(f"\t>  {de[cent[1],cent[0]]}")
                # print(f"\t\t {avgdist}")
                #hardcodana natancnost ker zadeva zna mal zajebavat mogoč obstaja bolj robusten način
                if(avgdist>3):
                    continue
                #print(de.shape)
                if de[cent[1],cent[0]] > avgdist+0.15 or (np.isnan(de[cent[1],cent[0]]) and (not np.isnan(avgdist))):
                    print("center je dlje od povprečja točk")
                    #MOŽNO DA JE TREBA ZAMENAT!!
                    # for z in i:
                    #     print(f"\t{z}")
                    """EDIT VRAČAMO VSE TRI TOČKE"""
                    return (cent[1],cent[0],avgdist,i)
        return None

    def find_elipses_first(self,image,depth_image,im_stamp,depth_stamp,grayBGR_toDrawOn):

            # change depth image
            minValue = np.nanmin(depth_image)
            maxValue = np.nanmax(depth_image)
            depth_im = depth_image.copy()
            temp_mask = ~np.isfinite(depth_image)
            nanToValue = maxValue*1.25
            depth_image[temp_mask] = nanToValue
            depth_image = depth_image - minValue
            depth_image = depth_image * 255/(nanToValue - minValue)
            depth_image = np.round(depth_image)
            depth_image = depth_image.astype(np.uint8)




            frame = image.copy()

            imagesToTest =  [
                             frame[:,:,0]
                            , frame[:,:,1]
                            , frame[:,:,2]
                            , module.bgr2gray(frame)
                            , cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,0]
                            , cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1]
                            , cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,2]
                            # , depth_image
                            ]

            for i in imagesToTest:
                temp = cv2.equalizeHist(i)
                # print(f"\tmax: {np.max(i)} --> {np.max(temp)}")
                # print(f"\tmin: {np.min(i)} --> {np.min(temp)}")
                # print(f"\t()()()()()()()()()()()()()()()()()()")

            # Set the dimensions of the image
            dims = frame.shape


            allFramesAnalized = []

            for i in range(len(imagesToTest)):
                currentImage = imagesToTest[i]

                threshDict = module.calc_thresholds(currentImage)

                allFramesAnalized.append(threshDict)

            # t = threshDict["normal"]
            # pprint(t)
            t = None
            counter = 0
            eachAvg = []
            for threshDict in allFramesAnalized:
                avg = None
                for j,key in enumerate(threshDict):
                    # current frame
                    if j==0:
                        avg = threshDict[key]/len(allFramesAnalized)
                    else:
                        avg += threshDict[key]/len(allFramesAnalized)
                    #all frames
                    if counter==0:
                        t = threshDict[key]
                        counter += 1
                    else:
                        part1 = counter/(counter+1)
                        part2 = 1/(counter+1)
                        t = cv2.addWeighted(t,part1, threshDict[key],part2, 0)
                        counter += 1
                eachAvg.append(avg)

            thresh = t

            # return thresh

            thresh = cv2.Canny(thresh,100,200)
            # return thresh

            # toReturn = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)


            depthThresh = module.calc_thresholds(depth_image)
            depth_t = None
            for i, key in enumerate(depthThresh):
                if i==0:
                    depth_t = depthThresh[key]
                else:
                    part1 = i/(i+1)
                    part2 = 1/(i+1)
                    depth_t = cv2.addWeighted(depth_t,part1, depthThresh[key],part2, 0)


            # return cv2.addWeighted(depth_t,0.5, t,0.5, 0)
            # return cv2.vconcat([t,depth_t])

            depth_t = cv2.Canny(depth_t, 100,200)


            toReturn = image.copy()
            toReturn[:,:,0] = thresh
            toReturn[:,:,1] = 0
            toReturn[:,:,2] = depth_t
            bestShift = 0
            bestScore = 0

            # print(round(np.abs(self.baseAngularSpeed),2))
            for shiftX in ([0] if round(np.abs(self.baseAngularSpeed),2)==0 else range(-100,100)):
                M = np.float32([[1,0,shiftX],[0,1,0]])
                toReturn[:,:,2] = cv2.warpAffine(depth_t,M,(toReturn.shape[1],toReturn.shape[0]),borderValue=np.nan).astype(np.uint8)

                b = np.sum((toReturn[:,:,0]*toReturn[:,:,2])>0)
                c = np.sum(toReturn[:,:,0]==255)
                d = np.sum(toReturn[:,:,2]==255)

                b = b/(c+d-b)
                # b = np.sum(((toReturn[:,:,0]==255)toReturn[:,:,2])>0)/(toReturn.shape[0]*toReturn.shape[1])
                if b > bestScore:
                    bestShift = shiftX
                    bestScore = b
                # print(b)
            print(f"\t\t\tbestShift: {bestShift}")
            print(f"\t\t\tbestScore: {bestScore}\n")
            shiftX = bestShift
            M = np.float32([[1,0,shiftX],[0,1,0]])
            toReturn[:,:,2] = cv2.warpAffine(depth_t,M,(toReturn.shape[1],toReturn.shape[0]),borderValue=np.nan).astype(np.uint8)
            depth_im_shifted = cv2.warpAffine(depth_im,M,(depth_im.shape[1],depth_im.shape[0]),borderValue=np.nan)

            toReturn = ((module.bgr2gray(toReturn)>0)*255).astype(np.uint8)


            kernel = np.ones((5,5), "uint8")

            # Extract contours
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(toReturn, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Example how to draw the contours
            # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

            # Fit elipses to all extracted contours
            elps = []
            for cnt in contours:
                #     print cnt
                #     print cnt.shape
                if cnt.shape[0] >= 20:
                    ellipse = cv2.fitEllipse(cnt)
                    elps.append(ellipse)

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! LOWER CODE IS NOT TO BE TOUCHED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Find two elipses with same centers
            """
            ((62.60783004760742, 82.68326568603516), --> (x,y)
            (32.40898513793945, 39.01688003540039), --> (širina,višina)
            134.35227966308594) --> naklon
            """
            candidates = []
            for n in range(len(elps)):
                for m in range(n + 1, len(elps)):
                    e1 = elps[n]
                    e2 = elps[m]
                    # pprint(e1)
                    # pprint(e2)
                    # print("----")
                    dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                    avg_cent = (int((e1[0][0] + e2[0][0])/2), int((e1[0][1] + e2[0][1])/2))
                    #             print dist
                    # candidates.append((e1,e2,avg_cent))
                    # continue
                    # cv2.ellipse(grayBGR_toDrawOn, e1, (255, 0, 0), 2)
                    # cv2.ellipse(grayBGR_toDrawOn, e2, (255, 0, 0), 2)

                    # centers must be less then 5px appart
                    if dist > 5:
                        continue

                    diff_w = e1[1][0] - e2[1][0]
                    diff_h = e1[1][1] - e2[1][1]
                    # width and height of e1 must be bouth smaller/bigger then from e2
                    if (diff_w == 0) or (diff_h == 0) or ((diff_h*diff_w)<0):
                        continue

                    # width and height must be smaller then 100
                    if np.abs(diff_w)>100 or np.abs(diff_h)>100:
                        continue


                    acceptNew = True
                    for i,c in enumerate(candidates):
                        (c_e1, c_e2, c_avg_cent) = c
                        # check if centers are almost the same
                        centers_dist = np.sqrt((c_avg_cent[0]-avg_cent[0])**2 + (c_avg_cent[1]-avg_cent[1])**2)
                        # print(f"\tcent_dist: {centers_dist}")

                        # check if they have same centers
                        if centers_dist < 5:
                            acceptNew = False

                            diff_w = e1[1][0] - e2[1][0]
                            diff_h = e1[1][1] - e2[1][1]
                            avgDiff = np.abs((diff_w + diff_h)/2)

                            c_diff_w = c_e1[1][0] - c_e2[1][0]
                            c_diff_h = c_e1[1][1] - c_e2[1][1]
                            c_avgDiff = np.abs((c_diff_w + c_diff_h)/2)

                            if avgDiff > c_avgDiff:
                                # print(F"replacement:")
                                # print(F"\toriginal: {e1[1][0]/e1[1][1] - e2[1][0]/e2[1][1]}")
                                # print(F"\tnew:      {c_e1[1][0]/c_e1[1][1] - c_e2[1][0]/c_e2[1][1]}")
                                candidates[i] = (e1, e2, avg_cent)


                    if acceptNew == True:
                        # print(F"appendment:")
                        # print(F"\tnew: {e1[1][0]/e1[1][1] - e2[1][0]/e2[1][1]}")
                        candidates.append((e1,e2,avg_cent))



            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UPPER CODE IS NOT TO BE TOUCHED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for c in candidates:


                e1 = c[0]
                e2 = c[1]

                h1,h2,w1,w2 = module.calc_pnts(e1)
                h11,h21,w11,w21 = module.calc_pnts(e2)

                e1_h1 = h1.copy().astype(int)
                e1_h2 = h2.copy().astype(int)
                e1_w1 = w1.copy().astype(int)
                e1_w2 = w2.copy().astype(int)


                e2_h1 = h11.copy().astype(int)
                e2_h2 = h21.copy().astype(int)
                e2_w1 = w11.copy().astype(int)
                e2_w2 = w21.copy().astype(int)

                h11= np.round((h11+h1)/2).astype(int)
                h21= np.round((h21+h2)/2).astype(int)
                w11= np.round((w11+w1)/2).astype(int)
                w21= np.round((w21+w2)/2).astype(int)




                cntr_ring = self.chk_ring(depth_im_shifted,h11,h21,w11,w21,c[2])

                # cntr_ring = self.chk_ring(depth_im,h11,h21,w11,w21,c[2])
                try:
                    #EDIT ali so koordinate pravilne
                    pnts = np.array( (image[cntr_ring[3][0][1],cntr_ring[3][0][2]], image[cntr_ring[3][1][1],cntr_ring[3][1][2]],image[cntr_ring[3][2][1],cntr_ring[3][2][2]]))

                    standin = np.zeros(image.shape)

                    """(32.40898513793945, 39.01688003540039), --> (širina,višina)"""
                    if(e1[1][0]<e2[1][0]):
                        cv2.ellipse(standin,e1,(0, 255, 0),-1)
                        cv2.ellipse(standin,e2,(0, 0, 0),-1)
                    else:
                        cv2.ellipse(standin,e2,(0, 255, 0),-1)
                        cv2.ellipse(standin,e1,(0, 0, 0),-1)

                    self.faceIm_pub.publish(CvBridge().cv2_to_imgmsg(standin, encoding="passthrough"))

                except Exception as e:
                    print(f"Ring error: {e}")
                    pass
            print()
            return  depth_im_shifted

    def find_objects(self):
        #print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0

        try:
            depth_image_message = rospy.wait_for_message("/camera/depth/image_raw", Image)
        except Exception as e:
            print(e)
            return 0

        try:
            odomMessage = rospy.wait_for_message("/odom", Odometry)
            position = odomMessage.pose.pose.position
            self.basePosition = {"x": position.x, "y": position.y, "z":position.z}
            self.baseAngularSpeed = odomMessage.twist.twist.angular.z
        except Exception as e:
            print(e)
            return 0

        #! timing operations
        print("--------------------------------------------------------")
        diff = (depth_image_message.header.stamp.to_sec()-rgb_image_message.header.stamp.to_sec())
        if (np.abs(diff) > 0.5):
            # stop everything if images are not up to date !!!
            pprint("skip")
            return

        # print how many secounds passed since previous loop
        tempLoop = datetime.now()
        tempRGB = rgb_image_message.header.stamp.to_sec()
        tempDepth = depth_image_message.header.stamp.to_sec()
        print(f"\t\t  loop: {np.round((tempLoop-self.loopTimer).total_seconds(),2)} s")
        print(f"\t\t   rgb: {np.round(tempRGB-self.rgbTimer,2)} s")
        print(f"\t\t depth: {np.round(tempDepth-self.depthTimer,2)} s\n")
        self.loopTimer = tempLoop
        self.rgbTimer = tempRGB
        self.depthTimer = tempDepth
        # print("--------------------------------------------------------")
        print(f"rgb_stamp_sec:    {rgb_image_message.header.stamp.to_sec()}")
        print(f"depth_stamp_sec:  {depth_image_message.header.stamp.to_sec()}")
        print(f"diff:             {diff}")
        print()



        # Convert the images into a OpenCV (numpy) format
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
        except CvBridgeError as e:
            print(e)

        grayImage = module.bgr2gray(rgb_image)
        grayImage = module.gray2bgr(grayImage)
        markedImage, depth_im_shifted = self.find_elipses_first(rgb_image, depth_image,rgb_image_message.header.stamp, depth_image_message.header.stamp, grayImage)
        #print(markedImage)
        #markedImage = self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)


def main():


        color_finder = ring_maker()


        rate = rospy.Rate(1.25)
        # rate = rospy.Rate(10)

        ch = input("Enter type(color|shape eq. rc - red cylinder)")
        skipCounter = 3
        loopTimer = rospy.Time.now().to_sec()
        # print(sleepTimer)
        while not rospy.is_shutdown():
            # print("hello!")
            if skipCounter <= 0:
                color_finder.find_objects()
            else:
                skipCounter -= 1


            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
