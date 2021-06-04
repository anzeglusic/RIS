#!/usr/bin/python3

import os
import itertools
import sys


import roslib
import time

import speech_recognition as sr

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
from std_msgs.msg import String
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
#from exercise4.srv import mess
from datetime import datetime



# /home/sebastjan/Documents/faks/3letnk/ris/ROS_task/src/exercise4/scripts

#modelsDir = "/home/code8master/Desktop/wsROS/src/RIS/exercise4/scripts"
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'
mouth = modelsDir+"mouth/Mouth.xml"
class color_localizer:

    def __init__(self):
        print()
        self.loopTimer = datetime.now()
        self.rgbTimer = 0
        self.depthTimer = 0

        #modern talking
        self.sr = sr.Recognizer()
        self.mic = sr.Microphone()



        self.foundAll = True

        self.basePosition = {"x":0, "y":0, "z":0}
        self.baseAngularSpeed = 0

        self.models = 0
        self.randomForestV2 = pickle.load(open(f"{modelsDir}/random_forestV2.sav", "rb"))
        self.random_forest_HSV = pickle.load(open(f"{modelsDir}/random_forest_HSV.sav", 'rb'))
        self.random_forest_RGB = pickle.load(open(f"{modelsDir}/random_forest_RGB.sav", 'rb'))
        self.knn_HSV = pickle.load(open(f"{modelsDir}/knn_HSV.sav", 'rb'))
        self.knn_RGB = pickle.load(open(f"{modelsDir}/knn_RGB.sav", 'rb'))
        print(mouth)
        self.mouth_finder = cv2.CascadeClassifier(mouth)
        if self.mouth_finder.empty():
           raise IOError("no mouth detector")

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

        #classifier topic
        self.class_pub = rospy.Publisher("/classifier", String,queue_size=1000)
        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('/ring_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('/face_im', Image, queue_size=1000)
        self.faceIm_pub = rospy.Publisher('/face_im2', Image, queue_size=1000)
        self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        self.twist_pub = rospy.Publisher('/our_pub1/chat2', Twist, queue_size=1000)

        #objava lokacij ob exploranju in izvajanju taskov
        self.face_pub = rospy.Publisher('/face_tw', Twist, queue_size=1000)
        self.cylinder_pub = rospy.Publisher('/cylinder_pt', Point, queue_size=1000)
        self.ring_pub = rospy.Publisher('/ring_pt', Point, queue_size=1000)

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

    # TODO: change for the brain of task 3

    def connect_QR_to_objects(self):

        # reset connections
        for QR_dict in self.positions["QR"]:
            QR_dict["isAssigned"] = False
        for cylinder_dict in self.positions["cylinder"]:
            cylinder_dict["QR_index"] = None
        for face_dict in self.positions["face"]:
            face_dict["QR_index"] = None

        for i,QR_dict in enumerate(self.positions["QR"]):
            closestObjectKey = None
            closestObjectIndx = None
            closestObjectDist = np.inf

            # chack all cylinders
            for j,cylinder_dict in enumerate(self.positions["cylinder"]):
                dist_vector = QR_dict["averagePostion"] - cylinder_dict["averagePostion"]
                dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                if dist < closestObjectDist:
                    closestObjectDist = dist
                    closestObjectKey = "cylinder"
                    closestObjectIndx = j

            # check all faces
            for j,face_dict in enumerate(self.positions["face"]):
                # include only those that have correct oriantation
                c_sim = np.dot(QR_dict["averageNormal"],face_dict["averageNormal"])/(np.linalg.norm(QR_dict["averageNormal"])*np.linalg.norm(face_dict["averageNormal"]))
                if c_sim<=0.5:
                    continue

                dist_vector = QR_dict["averagePostion"] - face_dict["averagePostion"]
                dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)

                if dist < closestObjectDist:
                    closestObjectDist = dist
                    closestObjectKey = "face"
                    closestObjectIndx = j


            # check if distance is ok
            if closestObjectDist > 0.5:
                continue

            # assign
            QR_dict["isAssigned"] = True
            # assert self.positions[closestObjectKey][closestObjectIndx]["QR_index"] is None, "Objekt ima že določeno svojo QR kodo"
            self.positions[closestObjectKey][closestObjectIndx]["QR_index"] = i

        # print("\n\n\n")
        # pprint(self.positions,compact=True)
        # print("\n\n\n")
        return

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

        # toReturn = cv2.cvtColor(t,cv2.COLOR_GRAY2BGR)
        # toReturn = cv2.cvtColor(cv2.addWeighted(depth_t,0.5, t, 0.5,0),cv2.COLOR_GRAY2BGR)
        # return toReturn
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
            # print(f"\t\t\tb: {b}")
            # print(f"\t\t\tc: {c}")
            # print(f"\t\t\td: {d}\n")
            # print(b/c)
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

        # print(f"all: {np.unique(toReturn)}")
        # return toReturn[:,:,2],depth_im_shifted
        toReturn = ((module.bgr2gray(toReturn)>0)*255).astype(np.uint8)
        # return toReturn.astype(np.uint8),depth_im_shifted
        # print(toReturn.shape)
        # print(f"YES: {np.sum(toReturn>0)}")
        # print(f"NO:  {np.sum(toReturn<=0)}")
        # print(f"all: {np.unique(toReturn)}")
        # print(f"all: {np.unique(module.gray2bgr(toReturn))}")


        # return toReturn,depth_im_shifted
        # grayBGR_toDrawOn = toReturn

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



                # #preverimo da sta res 2 razlicne elipse in en vec za malo različnih elips
                # if dist < 5 and np.abs(e1[1][0]-e2[1][0])>1 and np.abs(e1[1][1]-e2[1][1]) > 1:
                #     #print()
                #     conf = True
                #     #precerimo da ne dodajamo vec kot en isti par elips
                #     for d in candidates:
                #         dis = np.sqrt(((avg_cent[0] - d[2][0]) ** 2 + (avg_cent[1] - d[2][1]) ** 2))
                #         if dis < 5:
                #             conf = False
                #     if conf:
                #         candidates.append((e1,e2,avg_cent))
        ##print(len(candidates))
        #rabimo trimat tiste ki so si izredno blizu
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UPPER CODE IS NOT TO BE TOUCHED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for c in candidates:

            # print(c[0][0])
            # print(f"\t{c[0][1]}")
            # print(c[1][0])
            # print(f"\t{c[1][1]}")
            # print("----")

            #print("candidates found",)
            # the centers of the ellipses
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


            # drawing the ellipses on the image
            # cv2.ellipse(grayBGR_toDrawOn, e1, (0, 0, 255), 2)
            # cv2.ellipse(grayBGR_toDrawOn, e2, (0, 0, 255), 2)


            # cv2.circle(grayBGR_toDrawOn,tuple( c[2]),1,(0,255,0),2)

            # cv2.circle(grayBGR_toDrawOn,tuple( h11),1,(0,255,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( h21),1,(0,255,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( w11),1,(0,255,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( w21),1,(0,255,0),2)


            # cv2.circle(grayBGR_toDrawOn,tuple( e1_h1),1,(255,0,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e1_h2),1,(0,255,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e1_w1),1,(0,0,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e1_w2),1,(255,255,255),2)

            # cv2.circle(grayBGR_toDrawOn,tuple( e2_h1),1,(255,0,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e2_h2),1,(0,255,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e2_w1),1,(0,0,0),2)
            # cv2.circle(grayBGR_toDrawOn,tuple( e2_w2),1,(255,255,255),2)

            cntr_ring = self.chk_ring(depth_im_shifted,h11,h21,w11,w21,c[2])

            # cntr_ring = self.chk_ring(depth_im,h11,h21,w11,w21,c[2])
            try:

                #EDIT ali so koordinate pravilne
                pnts = np.array( (image[cntr_ring[3][0][1],cntr_ring[3][0][2]], image[cntr_ring[3][1][1],cntr_ring[3][1][2]],image[cntr_ring[3][2][1],cntr_ring[3][2][2]]))
                #---------------------NEW COLOR------------------------------------#

                standin = np.zeros(image.shape)

                # """(32.40898513793945, 39.01688003540039), --> (širina,višina)"""
                if(e1[1][0]<e2[1][0]):
                    cv2.ellipse(standin,e2,(0, 255, 0),-1)
                    cv2.ellipse(standin,e1,(0, 0, 0),-1)
                else:
                    cv2.ellipse(standin,e1,(0, 255, 0),-1)
                    cv2.ellipse(standin,e2,(0, 0, 0),-1)

                standin = standin.astype("uint8")
                mask = standin[:,:,1] == 255

                t = image[mask,:]
                pts = t.tolist()
                color = module.calc_rgbV2(pts,self.randomForestV2,"ring")
                print(f"\t\t\t\t\tRing color is {color}")
                #-------------------------------------------------------------------#

                #---------------------OLD COLOR------------------------#
                # print("dela 1")
                # color = module.calc_rgb(pnts,self.knn_RGB,self.random_forest_RGB,self.knn_HSV,self.random_forest_HSV)
                #------------------------------------------------------#

                # print("dela 2")
                ring_point = module.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im_shifted,"ring",depth_stamp,color,self.tf_buf)
                #ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im_shifted,"ring",depth_stamp,color)
                # print("dela 3")
                (self.nM, self.m_arr, self.positions) = module.addPosition(np.array((ring_point.position.x,ring_point.position.y,ring_point.position.z)),"ring",color,self.positions,self.nM, self.m_arr, self.markers_pub, showEveryDetection=self.showEveryDetection)
                # print("dela 4")
                # ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im,"ring")

                # drawing the ellipses on the image
                cv2.ellipse(grayBGR_toDrawOn, e1, (0, 0, 255), 2)
                cv2.ellipse(grayBGR_toDrawOn, e2, (0, 0, 255), 2)


                cv2.circle(grayBGR_toDrawOn,tuple( c[2]),1,(0,255,0),2)

                cv2.circle(grayBGR_toDrawOn,tuple( h11),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( h21),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( w11),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( w21),1,(0,255,0),2)

            except Exception as e:
                print(f"Ring error: {e}")
                pass
        print()
        return grayBGR_toDrawOn, depth_im_shifted

    #OHRANI
    def getLows(self, depth_line):
        points = [0]
        pointsClose = []
        if depth_line[0]>depth_line[1]:
            pointsClose.append(False)
        else:
            pointsClose.append(True)

        #False pomeni da sta obe točki okoli bližje
        for i in range(1,len(depth_line)-1):
            #preverjamo za preskok
            mejni_preskok = 0.03
            if (np.abs(depth_line[i-1]-depth_line[i])>mejni_preskok or np.abs(depth_line[i]-depth_line[i+1])>mejni_preskok):
                points.append(i)
                pointsClose.append(False)
                continue
            #ali je trneutni nan ali pa levi nan in je desni nan pol nadaljujemo
            if (np.isnan(depth_line[i]) or ( np.isnan(depth_line[i-1]) and np.isnan(depth_line[i+1])) ):
                continue
            if (np.isnan(depth_line[i-1]) or depth_line[i-1]>=depth_line[i]) and (np.isnan(depth_line[i+1]) or depth_line[i+1]>=depth_line[i]):
                points.append(i)
                pointsClose.append(True)
                continue
            if (np.isnan(depth_line[i-1]) or depth_line[i-1]<depth_line[i]) and (np.isnan(depth_line[i+1]) or depth_line[i+1]<depth_line[i]):
                points.append(i)
                pointsClose.append(False)
                continue

        points.append(len(depth_line)-1)
        if depth_line[len(depth_line)-2]>depth_line[len(depth_line)-1]:
            pointsClose.append(True)
        else:
            pointsClose.append(False)
        # pprint(list(zip(points,pointsClose)))
        return (points,pointsClose)

    def getRange(self, depth_line, center_point,levo,desno):
        #preverimo da ni robni
        if levo == -1:
            # print(f"Levo quit")
            return None
        if desno == -1:
            # print(f"Desno quit")
            return None

        counter = 0

        while True:
            if (center_point-counter) == levo or np.abs(depth_line[center_point-counter]-depth_line[center_point-counter+1])>0.1:
                counter -= 1
                break
            if (center_point+counter) == desno or np.abs(depth_line[center_point+counter]-depth_line[center_point+counter-1])>0.1:
                counter -= 1
                break
            counter += 1
        shift = np.floor((counter*2)/5).astype(int)
        #! shift = np.floor((counter*2)/7).astype(int)
        if shift < 1 or shift > 50:
        # if shift < 1:
            return None

        #vzamemo levo 3 in 6 in desno 3 in 6 oddaljena pixla ter računamo razmerje razmerja rabita biti priblizno enaka na obeh straneh
        #! LLL = depth_line[center_point - shift*3]
        LL = depth_line[center_point - shift*2]
        L = depth_line[center_point - shift]
        #! RRR = depth_line[center_point + shift*3]
        RR = depth_line[center_point + shift*2]
        R = depth_line[center_point + shift]
        C = depth_line[center_point]

        if np.isnan(LL) or np.isnan(L) or np.isnan(C) or np.isnan(R) or np.isnan(RR):
        #! if np.isnan(LL) or np.isnan(L) or np.isnan(C) or np.isnan(R) or np.isnan(RR) or np.isnan(RRR) or np.isnan(LLL):
            # print(f"NaN quit")
            return None
        # LL --> "left left"
        # C --> "center"
        # RR --> "right right"
        #! LLL_C = LLL-C
        LL_C = LL-C
        L_C = L-C
        C_R = R-C
        C_RR = RR-C
        #! C_RRR = RRR-C

        # LLL_LL = LLL - LL
        LL_L = LL - L
        R_RR = RR - R
        #! RR_RRR = RRR - RR


        if LL_C<=0 or L_C<=0 or C_R<=0 or C_RR<=0:
        #! if LL_C<=0 or L_C<=0 or C_R<=0 or C_RR<=0 or LLL_C<=0 or C_RRR<=0:
            # print(f"razlike niso uredu")
            return None

        detection_error = 0.1
        if (0.13-detection_error < L_C/LL_C < 0.25+detection_error) and (0.13-detection_error < C_R/C_RR < 0.25+detection_error) and (0.16-detection_error < L_C/LL_L < 0.33+detection_error) and (0.16-detection_error < C_R/R_RR < 0.33+detection_error):
            return (center_point-shift*2,center_point+shift*2)
        else:
            # print()
            # print(f"tested guess quit")
            # print(f"L_C/LL_C : {L_C/LL_C }")
            # print(f"C_R/C_RR: {C_R/C_RR}")
            # print(f"L_C/LL_L: {L_C/LL_L}")
            # print(f"C_R/R_RR: {C_R/R_RR}")
            return None

    #nastima tru_intervals na tiste ki so bli zaznani
    def get_ujemanja(self,acum_intervals,vrstica):
        #to so vrstice v katerih je interval to se rab prstet overall
        dolz = vrstica
        #potential vsebuje {interval: counter_ujemanj}
        potential = {}
        #samo seznam intervalov
        self.tru_intervals = []
        #gremo cez vse intervale
        for i in acum_intervals:
            #prazna vrstica
            if i == []:
                dolz -= 3
                #spraznimo potential
                potential = {}
                continue
            else:
                #ce je potetntial prazn ne rabmo primerjat
                if not potential:
                    for inter in i:
                        #dodamo potential za naslednjo iteracijo
                        potential[inter] = 0
                #gremo čez vse intervale
                else:
                    for inter in i:
                        #probs obstaja bolsi nacin
                        hold_meh = self.check_potets(inter,potential, dolz)
                        if hold_meh:
                            potential = hold_meh
                dolz -= 3

    def check_potets(self, interval, potets,dolz):
        #poglej ce je potets slucajn prazn
        if len(potets) == 0:
            return None
        if interval in potets.keys():
            potets[interval] += 1
            #tweak for max prileganje
            if potets[interval] == 5:
                self.tru_intervals.append((interval,dolz))
            return potets
        #gremo iskat vsebovanost
        for i in potets.keys():
            #ce je interval znotraj ali pa ce je drugi okol
            if (i[0]>=interval[0] and i[1]<=interval[1]) or (i[0]<= interval[0] and i[1]>= interval[1]):
                potets[i] += 1
                #tweak for max prileganje
                if potets[i] == 5:
                    self.tru_intervals.append((interval,dolz))
                return potets
        return None

    #not in use
    def check_lineup(self,acum):
        if len(acum)<=1:
            return None
        if acum[-2]:
            for i in acum[-1]:
                for j in acum[-2]:
                    if i == j:
                        return i
        return None

    #not in use
    def interval_in(self, acum, interval):
        if len(acum) == 0:
            return False
        if interval in acum:
            return True
        for i in acum:
            if (i[0]>=interval[0] and i[1]<=interval[1]) or (i[0]<= interval[0] and i[1]>= interval[1]):
                return True
        return False

    def check_if_ball(self, center_depth_up, center_depth, center_depth_down):
        # there is no value for up/denter/down
        if np.isnan(center_depth_up) or np.isnan(center_depth) or np.isnan(center_depth_down):
            print(f"BALL !!!!!!!!!!!!")
            return True

        detection_error = 0.01
        cd = center_depth
        cdd = center_depth_down
        cdu = center_depth_up
        if (np.abs(cd-cdd)>detection_error) or (np.abs(cd-cdu)>detection_error) or (np.abs(cdu-cdd)>detection_error):
            print(f"BALL !!!!!!!!!!!!")
            return True




        # print(f"center_depth_up:   {center_depth_up}")
        # print(f"center_depth:      {center_depth}")
        # print(f"center_depth_down: {center_depth_down}")
        # print()
        print(f"CYLINDER !!!!!!!!!!!!")
        return False

    def find_cylinderDEPTH(self,image, depth_image, grayBGR_toDrawOn,depth_stamp):
        # print(depth_stamp)
        centerRowIndex = depth_image.shape[0]//2
        max_row = (depth_image.shape[0]//7)*2
        acum_me= []
        curr_ints = []
        actual_add = (-1,-1)
        """rabim narediti count za vsak mozn cilinder v sliki posebi!!"""
        #count_ujemanj drži število zaznanih vrstic s cilindri(prekine in potrdi pri 3eh)
        #GRABO NOT BEING USED
        count = 0
        count_ujemanj = [0]
        #END OF GARBO
        for rows in range(centerRowIndex, max_row, -3):
        # for rows in range(centerRowIndex-20, centerRowIndex-21, -1):
            presekiIndex,blizDalec = self.getLows(depth_image[rows,:])
            cnt = 0
            acum_me.append([])
            # print(presekiIndex)
            # print("----")
            # print(blizDalec)
            for cnt in range(1,len(presekiIndex)-1):
                #se pojavi kot najblizja točka katere desni in levi pixl sta  za njo
                if blizDalec[cnt]:
                    #notri shranjen levo in desno interval rows vsebuje kera vrstica je trenutno
                    interval = self.getRange(depth_image[rows,:],presekiIndex[cnt],presekiIndex[cnt-1],presekiIndex[cnt+1])
                    # print(f"{presekiIndex[cnt]}: {interval}")
                    if not interval is None:
                        #grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(interval[0],centerRowIndex), radius=2,color=[255,0,0], thickness=-1)
                        #grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(interval[1],centerRowIndex), radius=2,color=[255,0,0], thickness=-1)

                        #izrise crto intervala

                        #grayBGR_toDrawOn[centerRowIndex,list(range(interval[0],interval[1]))] = [255,0,0]
                        #NASLI SMO CILINDER
                        # print(depth_image)
                        # for i in range(interval[0]+1,interval[1]):
                        #     #izrise crto intervala
                        #     grayBGR_toDrawOn[rows,i] = [255,255,0]

                        try:
                            #preverimo da je realen interval
                            if np.abs(interval[0]-interval[1]) <= 20 or depth_image[rows,presekiIndex[cnt]]>2.5:
                                continue


                            # interval je kul
                            #1. ga dodamo v akumulator
                            acum_me[count].append(interval)



                            for i in range(interval[0]+1,interval[1]):
                                #izrise crto intervala
                                grayBGR_toDrawOn[rows,i] = [255,255,0]
                        except Exception as e:
                            print(f"find_cylinderDEPTH error: {e}")
                            return grayBGR_toDrawOn
                #drawing on screen
                if blizDalec[cnt]:
                    grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(presekiIndex[cnt],rows), radius=2,color=[60,181,9], thickness=-1)
                    #grayBGR_toDrawOn[centerRowIndex-1,presekiIndex[cnt]-1] = np.array([[60,181,9]])
                else:
                    grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(presekiIndex[cnt],rows), radius=2,color=[0,7,255], thickness=-1)
                    #grayBGR_toDrawOn[centerRowIndex-1,presekiIndex[cnt]-1] = np.array([[0,7,255]])
                        #if not empty
            """
            if interval:
                count_ujemanj += 1
                if count_ujemanj == 3:

                    points = np.array([ image[rows,(inter[0]+inter[1])//2],
                                        image[rows,(inter[0]+inter[1])//2-1],
                                        image[rows,(inter[0]+inter[1])//2+1]])

                    print(f"Širina:{inter[1]} {inter[0]}\n\tna razdalji:{depth_image[centerRowIndex,presekiIndex[(inter[0]+inter[1])//2+1]]}")
                    colorToPush = self.calc_rgb(points)
                    pose = self.get_pose((inter[0]+inter[1])//2,rows,depth_image[rows,(inter[0]+inter[1])//2],depth_image,"cylinder",depth_stamp,colorToPush)
                    self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush)
            else:
                count_ujemanj = 0
                """
            count += 1
        #inter = self.check_lineup(acum_me)
        self.get_ujemanja(acum_me,centerRowIndex)
        print(self.tru_intervals)
        #ce ni prazn
        for inter in self.tru_intervals:

            # pogledamo da ni krogla
            vertical_shift = 10
            center_depth_up = depth_image[inter[1]-vertical_shift,(inter[0][0]+inter[0][1])//2]
            center_depth_down = depth_image[inter[1]+vertical_shift,(inter[0][0]+inter[0][1])//2]
            center_depth = depth_image[inter[1],(inter[0][0]+inter[0][1])//2]
            if self.check_if_ball(center_depth_up,center_depth,center_depth_down):
                continue

            for i in range(inter[0][0],inter[0][1]):
                grayBGR_toDrawOn[inter[1],i] = [0,0,255]
            points = np.array([ image[inter[1],(inter[0][0]+inter[0][1])//2],
                                image[inter[1],(inter[0][0]+inter[0][1])//2+1],
                                image[inter[1],(inter[0][0]+inter[0][1])//2-1]])

            print(f"Širina:{inter[0][0]} {inter[0][1]}\n\tna razdalji:{depth_image[inter[1],(inter[0][0]+inter[0][1])//2]}")
            # colorToPush = module.calc_rgb(points,self.knn_RGB,self.random_forest_RGB,self.knn_HSV,self.random_forest_HSV)
            training = image[inter[1]:inter[1]+13,inter[0][0]:inter[0][1],:].astype("uint8")

            t = training[np.zeros(training[:,:,0].shape)==0,:]
            send_me = t.tolist()
            colorToPush = module.calc_rgbV2(send_me,self.randomForestV2,"cylinder")
            print(f"\t\t\t\tClyinder color is {colorToPush}")
            pose = module.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush,self.tf_buf)
            #pose = self.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush)
            (self.nM, self.m_arr, self.positions) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush,self.positions,self.nM, self.m_arr, self.markers_pub, showEveryDetection=self.showEveryDetection)
            #self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush)

        return grayBGR_toDrawOn

    #NE OHRANI NAPREJ
    def find_objects(self,koncal_ile):
        #print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return True

        try:
            depth_image_message = rospy.wait_for_message("/camera/depth/image_raw", Image)
        except Exception as e:
            print(e)
            return True

        try:
            odomMessage = rospy.wait_for_message("/odom", Odometry)
            position = odomMessage.pose.pose.position
            self.basePosition = {"x": position.x, "y": position.y, "z":position.z}
            self.baseAngularSpeed = odomMessage.twist.twist.angular.z
        except Exception as e:
            print(e)
            return True

        #! timing operations
        print("--------------------------------------------------------")
        diff = (depth_image_message.header.stamp.to_sec()-rgb_image_message.header.stamp.to_sec())
        if (np.abs(diff) > 0.5):
            # stop everything if images are not up to date !!!
            pprint("skip")
            return True

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
        markedImage = self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)
        # TODO: make it so it marks face and returns the image to display
        self.find_faces(rgb_image,depth_im_shifted,depth_image_message.header.stamp)
        markedImage = self.find_QR(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
        markedImage = self.find_digits_new(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
        #!
        module.checkForApproach(self.positions["face"],"face",self.face_pub)
        module.checkForApproach(self.positions["cylinder"],"cylinder",self.cylinder_pub)
        #for objectType in ["ring","cylinder"]:
        #    module.checkPosition(self.positions[objectType],self.basePosition, objectType, self.points_pub)

        # for qr in self.positions["QR"]:
        #     print(qr["data"])

        # print(type(markedImage))
        # print(markedImage.shape)
        #print(markedImage)
        self.pic_pub.publish(CvBridge().cv2_to_imgmsg(markedImage, encoding="passthrough"))
        #self.markers_pub.publish(self.m_arr)

        #preverjamo ce smo vse ze zaznali
        self.foundAll = module.keepExploring(self.positions)
        cylinder_found = self.check_cylinders()

        if (not (cylinder_found is None)) and koncal_ile :
            self.grab_cylinders(cylinder_found)



        return self.foundAll

    def check_cylinders(self):

        for c in self.positions["cylinder"]:
            if c["QR_index"] is None:
                return c

        return None

    def grab_cylinders(self,li_cylinders):

        for destination_cylinder in li_cylinders:
            # tell him where to go
            self.cylinder_pub.publish(Point(destination_cylinder["averagePostion"][0],destination_cylinder["averagePostion"][1],destination_cylinder["averagePostion"][2]))
            while True:
                if self.listener():
                    return

#! ================================================== digits start ==================================================
    def find_digits(self, rgb_image, depth_image_shifted, stamp,grayBGR_toDrawOn):
        ret, thresh = cv2.threshold(module.bgr2gray(rgb_image), 40, 255, 0)
        corners, ids, rejected_corners = cv2.aruco.detectMarkers(module.gray2bgr(thresh),self.dictm,parameters=self.params)
        # print("\n\n\n")
        # print("\n\n\n")
        print()
        # pprint(corners)

        # thresh = cv2.adaptiveThreshold(module.bgr2gray(rgb_image),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
        # return module.gray2bgr(thresh)

        # grayBGR_toDrawOn = rgb_image
        allCorners = []

        # robot body is in the bottom of an image (480*0.875 = 420)
        top_limit = round(rgb_image.shape[0] * 0.875) # how much from top is posible for marker/corrner (edge) to be

        #! accepted corners --> GREEN
        for marker in corners:
            # print(marker[0])
            minimums = np.min(marker[0],axis=0)
            maximums = np.max(marker[0],axis=0)
            bgr_color = (0,255,0)
            start_point = (round(minimums[0]),round(minimums[1]))
            end_point = (round(maximums[0]),round(maximums[1]))
            thickness = 5
            if end_point[1] < top_limit:
                allCorners.append({ "start_point":np.array([start_point[0],start_point[1]]),
                                    "end_point":np.array([end_point[0],end_point[1]]),
                                    "center": np.array([(start_point[0]+end_point[0])//2, (start_point[1]+end_point[1])//2]),
                                    "color":"green",
                                    "original_corner": marker})
                # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, start_point, end_point, bgr_color, thickness)
        #! rejected corners --> RED
        for marker in rejected_corners:
            # print(marker[0])
            minimums = np.min(marker[0],axis=0)
            maximums = np.max(marker[0],axis=0)
            bgr_color = (0,0,255)
            start_point = (round(minimums[0]),round(minimums[1]))
            end_point = (round(maximums[0]),round(maximums[1]))
            thickness = 5
            if end_point[1] < top_limit:
                allCorners.append({ "start_point":np.array([start_point[0],start_point[1]]),
                                    "end_point":np.array([end_point[0],end_point[1]]),
                                    "center": np.array([(start_point[0]+end_point[0])//2, (start_point[1]+end_point[1])//2]),
                                    "color":"red",
                                    "original_corner": marker})
                # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, start_point, end_point, bgr_color, thickness)

        # area
        for marker in allCorners:
            marker_area_dims = marker["end_point"]-marker["start_point"]
            marker["area"] = marker_area_dims[0]*marker_area_dims[1]
            # print(f"{marker['color']}\t{marker['center']}")

        # pprint(allCorners)

        # sort by (x)
        allCorners = sorted(allCorners,key=lambda x: x["center"][0]) # min --> max (left --> right)

        # pprint(allCorners)

        # doubles must cross middle row of an image
        middle_row_indx = rgb_image.shape[0]//2

        doubles = [] # 2 markers merged
        for i in range(0,len(allCorners)-1):
            for j in range(i+1,len(allCorners)):
                m1 = allCorners[i]
                m2 = allCorners[j]
                # not verticaly aligned --> break, because they are sorted by ["center"][0]
                if np.abs(m1["center"][0]-m2["center"][0]) > 10:
                    break
                # else:
                #     print(f"diff: {np.abs(m1['center'][0]-m2['center'][0])}")
                # area not in 75% margin
                if (m1["area"]/m2["area"] < 0.75) or (m2["area"]/m1["area"] < 0.75):
                    continue
                # else:
                #     print(f"area_diff: {min(m1['area']/m2['area'],m2['area']/m1['area'])}")

                bgr_color = (255,0,0)
                start_point = ( min(m1["start_point"][0],m2["start_point"][0]),
                                min(m1["start_point"][1],m2["start_point"][1]))
                end_point = (   max(m1["end_point"][0],m2["end_point"][0]),
                                max(m1["end_point"][1],m2["end_point"][1]))

                # doubles area must cross middle row of an image
                if start_point[1]>middle_row_indx or end_point[1]<middle_row_indx:
                    continue

                thickness = 5
                # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, start_point, end_point, bgr_color, thickness)
                start_p = np.array([start_point[0],start_point[1]])
                end_p = np.array([end_point[0],end_point[1]])
                area_dims = end_p-start_p
                doubles.append({"start_point":start_p,
                                "end_point": end_p,
                                "area": area_dims[0]*area_dims[1],
                                "center": np.array([(start_p[0]+end_p[0])//2, (start_p[1]+end_p[1])//2]),
                                "top_original_corner": m1["original_corner"] if m1["center"][1] < m2["center"][1] else m2["original_corner"],
                                "bottom_original_corner": m1["original_corner"] if m1["center"][1] > m2["center"][1] else m2["original_corner"]})


        doubles = sorted(doubles,key=lambda x: x["center"][0])
        papers = []
        c = 0
        for i in range(1,len(doubles)):
            d1 = doubles[i-1]
            d2 = doubles[i]

            x1 = int(round(d1["start_point"][0]))
            y1 = int(round(min(d1["start_point"][1],d2["start_point"][1])))

            x2 = int(round(d2["end_point"][0]))
            y2 = int(round(max(d1["end_point"][1],d2["end_point"][1])))
            # print(type(x1))
            # ratio is bigger then it is commen for paper
            if (x2-x1)/(y2-y1) > 0.75:
                continue
            grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (x1,y1), (x2,y2), (0,0,255) if c%2==0 else (255,0,0), 5)
            c += 1
            papers.append({ "start_point": np.array([x1,y1]),
                            "end_point": np.array([x2,y2]),
                            "top_left_original_corner": d1["top_original_corner"],
                            "bottom_left_original_corner": d1["bottom_original_corner"],
                            "top_right_original_corner": d2["top_original_corner"],
                            "bottom_right_original_corner": d2["bottom_original_corner"]
            })
            # print(f"paper_{i-1}: {x2-x1}/{y2-y1} = {(x2-x1)/(y2-y1)}")

        # pprint(papers)


        # draw middle row
        # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (0,middle_row_indx-5), (rgb_image.shape[1]-1,middle_row_indx+5), (0,0,0), 4)


        # return grayBGR_toDrawOn
        # return rgb_image
        # return img_out

        # Increase proportionally if you want a larger image
        # image_size=(351,248,3)
        # marker_side=50

        # img_out = np.zeros(image_size, np.uint8)
        # out_pts = np.array([[marker_side/2,img_out.shape[0]-marker_side/2],
        #                 [img_out.shape[1]-marker_side/2,img_out.shape[0]-marker_side/2],
        #                 [marker_side/2,marker_side/2],
        #                 [img_out.shape[1]-marker_side/2,marker_side/2]])

        # src_points = np.zeros((4,2))
        # cens_mars = np.zeros((4,2))

        # pprint(corners)
        # print("---")
        # pprint(ids)
        image_size=(351,248,3)
        marker_side=50
        # img_out = np.zeros(image_size, np.uint8)
        for i in range(0,len(papers)):


            img_out = np.zeros(image_size, np.uint8)
            out_pts = np.array([[marker_side/2,img_out.shape[0]-marker_side/2],
                            [img_out.shape[1]-marker_side/2,img_out.shape[0]-marker_side/2],
                            [marker_side/2,marker_side/2],
                            [img_out.shape[1]-marker_side/2,marker_side/2]])
            src_points = np.zeros((4,2))
            cens_mars = np.zeros((4,2))
        # for i in range(0,1):
            # print("---")
            # pprint(temp_corners)
            ids = np.array([[4],[3],[2],[1]]).astype("int32")
            corners = [papers[i]["bottom_right_original_corner"].copy(),
                            papers[i]["bottom_left_original_corner"].copy(),
                            papers[i]["top_right_original_corner"].copy(),
                            papers[i]["top_left_original_corner"].copy()]
            # print("---")
            # pprint(temp_ids)

            if not ids is None:
                if len(ids)==4:
                    # print('4 Markers detected')

                    for idx in ids:
                        # Calculate the center point of all markers
                        cors = np.squeeze(corners[idx[0]-1])
                        cen_mar = np.mean(cors,axis=0)
                        cens_mars[idx[0]-1]=cen_mar
                        cen_point = np.mean(cens_mars,axis=0)

                    for coords in cens_mars:
                        #  Map the correct source points
                        if coords[0]<cen_point[0] and coords[1]<cen_point[1]:
                            src_points[2]=coords
                        elif coords[0]<cen_point[0] and coords[1]>cen_point[1]:
                            src_points[0]=coords
                        elif coords[0]>cen_point[0] and coords[1]<cen_point[1]:
                            src_points[3]=coords
                        else:
                            src_points[1]=coords

                    h, status = cv2.findHomography(src_points, out_pts)
                    img_out = cv2.warpPerspective(rgb_image, h, (img_out.shape[1],img_out.shape[0]))

                    ################################################
                    #### Extraction of digits starts here
                    ################################################

                    # Cut out everything but the numbers
                    img_out = img_out[125:221,50:195,:]

                    # Convert the image to grayscale
                    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

                    # Option 1 - use ordinairy threshold the image to get a black and white image
                    #ret,img_out = cv2.threshold(img_out,100,255,0)

                    # Option 1 - use adaptive thresholding
                    img_out = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)

                    # Use Otsu's thresholding
                    #ret,img_out = cv2.threshold(img_out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # Pass some options to tesseract
                    config = '--psm 13 outputbase nobatch digits'

                    # Visualize the image we are passing to Tesseract
                    # cv2.imshow('Warped image',img_out)
                    # cv2.waitKey(1)

                    # Extract text from image
                    text = pytesseract.image_to_string(img_out, config = config)

                    # Check and extract data from text
                    # print('Extracted>>',text)

                    # Remove any whitespaces from the left and right
                    text = text.strip()

                    # If the extracted text is of the right length
                    if len(text)==2:
                        x=int(text[0])
                        y=int(text[1])
                        print(f"The extracted datapoints are x={x}, y={y}")
                        # TODO: dodaj v akumulator !!!
                        # TODO: izračunaj pozicijo !!!
                        # TODO: izračunaj normalo !!!
            #         else:
            #             print(f"The extracted text has is of length {len(text)}. Aborting processing")

            #     else:
            #         print(f"The number of markers is not ok: {len(ids)}")
            # else:
            #     print("No markers found")


        return grayBGR_toDrawOn
        # return img_out

    def find_digits_new(self, rgb_image, depth_image_shifted, stamp,grayBGR_toDrawOn):

        corners, ids, rejected_corners = cv2.aruco.detectMarkers(rgb_image,self.dictm,parameters=self.params)

        if ids is None:
            return grayBGR_toDrawOn

        # drawing specific markers
        # for i,marker in enumerate(corners):
        #     # print(marker[0])
        #     minimums = np.min(marker[0],axis=0)
        #     maximums = np.max(marker[0],axis=0)
        #     bgr_color = (0,255,0)
        #     if ids[i][0] == 1:
        #         bgr_color = (255,0,0)
        #     elif ids[i][0] == 2:
        #         bgr_color = (0,255,0)
        #     elif ids[i][0] == 3:
        #         bgr_color = (0,0,255)
        #     elif ids[i][0] == 4:
        #         bgr_color = (255,255,0)

        #     start_point = (round(minimums[0]),round(minimums[1]))
        #     end_point = (round(maximums[0]),round(maximums[1]))
        #     thickness = 5
        #     grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, start_point, end_point, bgr_color, thickness)


        tempIds = ids.flatten()
        indxs_1 = np.where(tempIds==1)[0]
        indxs_2 = np.where(tempIds==2)[0]
        indxs_3 = np.where(tempIds==3)[0]
        indxs_4 = np.where(tempIds==4)[0]

        centers = np.array([ np.mean(x[0],axis=0) for x in corners])

        papers = []
        # find indx_1 (top-left)
        for indx_1 in indxs_1:
            # find indx_3 (bottom-left)
            for indx_3 in indxs_3:
                # check vertical alignment with indx_1
                if np.abs(centers[indx_1][0]-centers[indx_3][0]) > 10:
                    continue
                # find indx_2 (top-right)
                for indx_2 in indxs_2:
                    # check horizontal distnace from indx_1 with respect to vertical length between indx_1 and indx_3
                    if np.abs(centers[indx_1][0]-centers[indx_2][0])/np.abs(centers[indx_1][1]-centers[indx_3][1]) > 0.75:
                        continue
                    # find indx_4 (bottom-right)
                    for indx_4 in indxs_4:
                        # check vertical alignment with indx_2
                        if np.abs(centers[indx_2][0]-centers[indx_4][0]) > 10:
                            continue
                        current_corners = [corners[indx_4], corners[indx_3], corners[indx_2], corners[indx_1]]
                        current_ids = np.array([[4],[3],[2],[1]]).astype("int32")
                        center_cordinates = np.array([centers[indx_1], centers[indx_2], centers[indx_3], centers[indx_4]])
                        papers.append((current_corners, current_ids, center_cordinates))

        # Increase proportionally if you want a larger image
        image_size=(351,248,3)
        marker_side=50

        for (corners, ids, center_cordinates) in papers:

            img_out = np.zeros(image_size, np.uint8)
            out_pts = np.array([[marker_side/2,img_out.shape[0]-marker_side/2],
                            [img_out.shape[1]-marker_side/2,img_out.shape[0]-marker_side/2],
                            [marker_side/2,marker_side/2],
                            [img_out.shape[1]-marker_side/2,marker_side/2]])

            src_points = np.zeros((4,2))
            cens_mars = np.zeros((4,2))

            for idx in ids:
                # Calculate the center point of all markers
                cors = np.squeeze(corners[idx[0]-1])
                cen_mar = np.mean(cors,axis=0)
                cens_mars[idx[0]-1]=cen_mar
                cen_point = np.mean(cens_mars,axis=0)

            for coords in cens_mars:
                #  Map the correct source points
                if coords[0]<cen_point[0] and coords[1]<cen_point[1]:
                    src_points[2]=coords
                elif coords[0]<cen_point[0] and coords[1]>cen_point[1]:
                    src_points[0]=coords
                elif coords[0]>cen_point[0] and coords[1]<cen_point[1]:
                    src_points[3]=coords
                else:
                    src_points[1]=coords

            h, status = cv2.findHomography(src_points, out_pts)
            img_out = cv2.warpPerspective(rgb_image, h, (img_out.shape[1],img_out.shape[0]))

            ################################################
            #### Extraction of digits starts here
            ################################################

            # Cut out everything but the numbers
            img_out = img_out[125:221,50:195,:]

            # Convert the image to grayscale
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

            # Option 1 - use ordinairy threshold the image to get a black and white image
            #ret,img_out = cv2.threshold(img_out,100,255,0)

            # Option 1 - use adaptive thresholding
            img_out = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)

            # Use Otsu's thresholding
            #ret,img_out = cv2.threshold(img_out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Pass some options to tesseract
            config = '--psm 13 outputbase nobatch digits'

            # Visualize the image we are passing to Tesseract
            # cv2.imshow('Warped image',img_out)
            # cv2.waitKey(1)

            # Extract text from image
            text = pytesseract.image_to_string(img_out, config = config)

            # Check and extract data from text
            # print('Extracted>>',text)

            # Remove any whitespaces from the left and right
            text = text.strip()

            # If the extracted text is of the right length
            if len(text)==2:
                # izračunaj normalo --> zaključi, če normala ni možna
                x1 = round(max(center_cordinates[0][0],center_cordinates[2][0]))
                y1 = round(max(center_cordinates[0][1], center_cordinates[1][1]))
                x2 = round(min(center_cordinates[1][0],center_cordinates[3][0]))
                y2 = round(min(center_cordinates[2][1], center_cordinates[3][1]))
                #print(type(x1),type(y1),type(x2),type(y2))
                norm = module.get_normal(depth_image_shifted, (int(x1),int(y1),int(x2),int(y2)),stamp,None,None, self.tf_buf)
                # if we are too close to QR code
                if norm is None:
                    print(f"Too close to the digits!")
                    continue

                x=int(text[0])
                y=int(text[1])
                num = int(text)
                # print(f"The extracted datapoints are x={x}, y={y}")
                print(f"Extracted number: {num}")
                # draw
                color = (0,255,0)
                thicknes = 5
                pts = np.array([corners[3][0][0],corners[2][0][1],corners[0][0][2],corners[1][0][3]]).astype("int")
                grayBGR_toDrawOn = cv2.polylines(grayBGR_toDrawOn, [pts], True, color, thicknes)
                # izračunaj pozicijo
                [xin, yin] = np.round(np.mean(center_cordinates, axis=0)).astype("int")
                dist = depth_image_shifted[yin, xin]
                pose = module.get_pose(xin,yin,dist, depth_image_shifted,"digits",stamp,None, self.tf_buf)
                # pprint(pose)
                # dodaj v akumulator
                (self.nM, self.m_arr, self.positions) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"digits", None,self.positions,self.nM, self.m_arr, self.markers_pub,showEveryDetection=self.showEveryDetection,normal=norm, data=num)

        return grayBGR_toDrawOn

#! =================================================== digits end ===================================================
# ===================================================================================================================
#! ==================================================== QR start ====================================================
    def find_QR(self, rgb_image, depth_image_shifted, stamp,grayBGR_toDrawOn):
        decodedObjects = pyzbar.decode(rgb_image)
        # print("\n\n\n")
        # print(decodedObjects)
        # print(len(decodedObjects))
        # print("\n\n\n")
        print()
        for i,dObject in enumerate(decodedObjects):
            # get normal
            x1 = dObject.rect.left
            y1 = dObject.rect.top
            x2 = dObject.rect.left + dObject.rect.width
            y2 = dObject.rect.top + dObject.rect.height
            norm = module.get_normal(depth_image_shifted, (x1,y1,x2,y2),stamp,None,None, self.tf_buf)
            # if we are too close to QR code
            if norm is None:
                print(f"Too close to the QR code!")
                continue
            pose = module.get_pose((x1+x2)//2,(y1+y2)//2,depth_image_shifted[(y1+y2)//2,(x1+x2)//2],depth_image_shifted,"QR",stamp,None,self.tf_buf)
            #pose = self.get_pose((x1+x2)//2,(y1+y2)//2,depth_image_shifted[(y1+y2)//2,(x1+x2)//2],depth_image_shifted,"QR",stamp,None)
            (self.nM, self.m_arr, self.positions) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"QR", None,self.positions,self.nM, self.m_arr, self.markers_pub,showEveryDetection=self.showEveryDetection,normal=norm, data=dObject.data.decode(), modelName=self.models, publisher = self.class_pub)
            self.models +=1
            #self.addPosition(np.array([pose.position.x,pose.position.y, pose.position.z]),"QR",None,norm,dObject.data.decode())



            # print(f"QR-{i} in the image!")
            # print(f"\tdata:    {dObject.data.decode()}") # data in the QR code
            # print(f"\trect:    {dObject.rect}") # rectangle where the QR is in the image
            # print(f"\tpolygon: {dObject.polygon}") # polygon (probabily 4 points) where exatly is the QR code
            # print(f"\tnorm:    {norm}")
            # print(f"\tpose:    {np.array([pose.position.x,pose.position.y, pose.position.z])}")
            # print(f"\tdist:    {depth_image_shifted[(y1+y2)//2,(x1+x2)//2]}")

            bgr_color = (0,255,0)
            start_point = (x1,y1)
            end_point = (x2, y2)
            thickness = 5
            grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, start_point, end_point, bgr_color, thickness)


        return grayBGR_toDrawOn
#! ===================================================== QR end =====================================================
# ===================================================================================================================
#! ============================================== face detection start ==============================================
    def find_faces(self,rgb_image, depth_image, depth_image_stamp):

        # Set the dimensions of the image
        self.dims = rgb_image.shape
        h = self.dims[0]
        w = self.dims[1]


        # split image into 3 different images that are 1:1 format
        # it is assumed that image is in format that the images' width is larger than height
        if self.dims[0] < self.dims[1]:
            num_of_dead_pixels = self.dims[1]-self.dims[0] # 160
            shift_pixels = num_of_dead_pixels//2 # 80
            rgb_image_left = rgb_image[:,0:self.dims[0],:] # 480x480
            rgb_image_middle = rgb_image[:,shift_pixels:(shift_pixels+self.dims[0]),:] # 480x480
            rgb_image_right = rgb_image[:,(self.dims[1]-self.dims[0]):,:] # 480x480
        else:
            print("\n\nSpodnja stranica bi morala biti daljša !!!!\n\n")

        # h_custom and w_custom for custom sizes for multiplication
        dims_custom = (0,0,0)
        dims_custom = rgb_image_left.shape
        h_custom = dims_custom[0]
        w_custom = dims_custom[1]


        imgL = cv2.resize(src=rgb_image_left, dsize=(300, 300))
        imgM = cv2.resize(src=rgb_image_middle, dsize=(300, 300))
        imgR = cv2.resize(src=rgb_image_right, dsize=(300, 300))



        # Tranform image to gayscale
        #gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        #img = cv2.equalizeHist(gray)

        # Detect the faces in the image
        #face_rectangles = self.face_detector(rgb_image, 0)
        blob = cv2.dnn.blobFromImages(images=[imgL,imgM,imgR], scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        face_detections = self.face_net.forward()

        right_image_shift = rgb_image.shape[1] - rgb_image.shape[0]
        before_value = 1.0
        detection_happend = False
        string_prints = []
        normals_found = []


        good_face_detections = []
        for i in range(0, face_detections.shape[2]):
            imageIndx = int(face_detections[0,0,i,0]) # 0 == left, 1 == middle, 2 == right
            confidence = face_detections[0, 0, i, 2]
            if confidence>0.5:
                box = face_detections[0,0,i,3:7] * np.array([w_custom,h_custom,w_custom,h_custom])
                box = box.astype('int')
                x1_custom, y1_custom, x2_custom, y2_custom = box[0], box[1], box[2], box[3]

                # y stay the same because original images' width is larger than its width
                if imageIndx == 0: # left
                    x1 = x1_custom
                    y1 = y1_custom
                    x2 = x2_custom
                    y2 = y2_custom
                elif imageIndx == 1: # middle
                    x1 = shift_pixels + x1_custom
                    y1 = y1_custom
                    x2 = shift_pixels + x2_custom
                    y2 = y2_custom
                elif imageIndx == 2: # right
                    x1 = right_image_shift + x1_custom
                    y1 = y1_custom
                    x2 = right_image_shift + x2_custom
                    y2 = y2_custom

                # poskrbi za morebitne zaznave izven okvirja slike
                if x1 < 0: # x1 preveč levo
                    x1 = 0
                if x1 > self.dims[1]-1: # x1 preveč desno
                    x1 = self.dims[1]-1
                if x2 < 0: # x2 preveč levo
                    x2 = 0
                if x2 > self.dims[1]-1: # x2 preveč desno
                    x2 = self.dims[1]-1
                if y1 < 0: # y1 preveč zgoray
                    y1 = 0
                if y1 > self.dims[0]-1: # y1 preveč spodaj
                    y1 = self.dims[0]-1
                if y2 < 0: # y2 preveč zgoraj
                    y2 = 0
                if y2 > self.dims[0]-1: # y2 preveč spodaj
                    y2 = self.dims[0]-1

                this_dict = {
                    "indx" : i,
                    "imageIndx": imageIndx, # 0 == left, 1 == middle, 2 == right
                    "confidence" : confidence,
                    "x": (x1,x2),
                    "y": (y1,y2)
                    }
                good_face_detections.append(this_dict)

        good_face_detections_sorted = sorted(good_face_detections,key=lambda tempDictionary: tempDictionary["confidence"],reverse=True)

        good_face_detections_final = []
        for good_detection in good_face_detections_sorted:
            x1, x2 = good_detection["x"]
            y1, y2 = good_detection["y"]

            is_good_detection = True
            for checked_good_detection in good_face_detections_final:
                if module.does_it_cross(checked_good_detection["x"], checked_good_detection["y"], (x1,x2), (y1,y2)):
                    is_good_detection = False

            if is_good_detection:
                good_face_detections_final.append(good_detection)



        br = 0
        for face in good_face_detections_final:
            i = face["indx"]
            imageIndx = face["imageIndx"]
            confidence = face["confidence"]
            (x1, x2) = face["x"]
            (y1,y2) = face["y"]


            # Extract region containing face
            face_region = rgb_image[y1:y2, x1:x2]
            if min(face_region.shape)==0:
                continue
            has_mask = self.find_mouth(face_region)
            #self.faceIm_pub.publish(CvBridge().cv2_to_imgmsg(face_region_m, encoding="passthrough"))


            # Find the distance to the detected face
            #assert y1 != y2, "Y sta ista !!!!!!!!!!!"
            #assert x1 != x2, "X sta ista !!!!!!!!!!!"
            face_distance = float(np.nanmean(depth_image[y1:y2,x1:x2]))
            if face_distance > 1.7:
                continue
            im = depth_image[y1:y2,x1:x2]
            norm = module.get_normal(depth_image, (x1,y1,x2,y2) ,depth_image_stamp,face_distance, face_region, self.tf_buf)
            normals_found.append(norm)

            print('Norm of face', norm)

            print('Distance to face', face_distance)

            #trans = tf2_ros.Buffer().lookup_transform('mapa', 'base_link', rospy.Time(0))
            #print('my position ',trans)
            # Get the time that the depth image was recieved
            depth_time = depth_image_stamp

            # Find the location of the detected face
            pose = module.get_pose_face((x1,x2,y1,y2), face_distance, depth_time,self.dims, self.tf_buf)
            #pose = self.get_pose_face((x1,x2,y1,y2), face_distance, depth_time)
            if not (pose is None) and not (norm is None):
                newPosition = np.array([pose.position.x,pose.position.y, pose.position.z])
                (self.nM, self.m_arr, self.positions) = module.addPosition(newPosition,"face", None, self.positions,self.nM, self.m_arr, self.markers_pub, showEveryDetection=self.showEveryDetection, normal=norm, mask=has_mask)
                #self.addPosition(newPosition, "face", color_char=None, normal=norm)

    def find_mouth(self, face_im):
        face_im = cv2.cvtColor(face_im, cv2.COLOR_BGR2GRAY)
        mouth_rects = self.mouth_finder.detectMultiScale(face_im)
        if len(mouth_rects) > 0:
            return False
        else:
            return True
        # for (x,y,w,h) in mouth_rects:
        #     cv2.rectangle(face_im, (x,y), (x+w,y+h), (0,255,0), 3)
        # return face_im

    def norm_accumulator(self, norm, center_point,dist1):
        found = -1
        if norm.size == 0  or center_point.size == 0 or dist1>1.5:
            print("huh")
            return False
        print("central point",center_point)
        if self.detected_pos_fin:

            for i in self.detected_pos_fin:
                dist = np.linalg.norm(center_point-self.detected_pos_fin[i][0])
                print("distance:",dist," ",i)
                #ce si je sredinska tocka dovolj blizu preverimo se kosinusno podobnost normal
                if dist<0.5:
                    c_sim = np.dot(norm,self.detected_norm_fin[i][0])/(np.linalg.norm(norm)*np.linalg.norm(self.detected_norm_fin[i][0]))
                    print("similarity:",c_sim)
                    if c_sim>0.5:
                        found = i
            if found >=0:
                print("najdeno ujemanje:",found)
                n = len(self.detected_pos_fin[found])
                #dodamo točko na koncu seznama točk  ce le ta nima 5 elementov(+1 za povprecje)
                if n < 6:
                    self.detected_pos_fin[found].append(center_point)
                    self.detected_norm_fin[found].append(norm)
                    #zracunamo nova povprecja racunamo od 1 naprej saj prvi je povprecje
                    self.detected_pos_fin[found][0] = np.array([sum([x[0] for x in self.detected_pos_fin[found][1:]])/n,sum([x[1] for x in self.detected_pos_fin[found][1:]])/n,sum([x[2] for x in self.detected_pos_fin[found][1:]])/n])
                    self.detected_norm_fin[found][0] = np.array([sum([x[0] for x in self.detected_norm_fin[found][1:]])/n,sum([x[1] for x in self.detected_norm_fin[found][1:]])/n,sum([x[2] for x in self.detected_norm_fin[found][1:]])/n])
                #ce smo sedaj polni posljemo normalo kot point!!
                print(self.detected_pos_fin[found][1:])
                if n == 6:
                    #poslji povprecni norm
                    print(n)
                    #self.points_pub(Point(self.detected_norm_fin[found][0][0],self.detected_norm_fin[found][0][1],self.detected_norm_fin[found][0][2]))
                    act_norm = self.detected_pos_fin[found][0]-self.detected_norm_fin[found][0]
                    #self.twist_pub.publish(Vector3(act_norm[0],act_norm[1],act_norm[2]),Vector3(self.detected_pos_fin[found][0][0],self.detected_pos_fin[found][0][1],self.detected_pos_fin[found][0][2]))
                    #dodamo se enega v norm da blokiramo aktivacijo tega ifa
                    self.detected_norm_fin[found].append(np.array([-1,-1,-1]))
                    self.detected_pos_fin[found].append(np.array([-1,-1,-1]))

                    return False
            else:
                print("ni bilo ujemanj:1")

                #ustvarimo nov entry
                self.detected_pos_fin[self.entries] = [center_point,center_point]
                self.detected_norm_fin[self.entries] =[norm,norm]
                act_norm = self.detected_pos_fin[self.entries][0]-self.detected_norm_fin[self.entries][0]
                #self.points_pub(Point(self.detected_pos_fin[found][0][0],self.detected_pos_fin[found][0][1],self.detected_pos_fin[found][0][2]))
                #! self.twist_pub.publish(Vector3(act_norm[0],act_norm[1],act_norm[2]),Vector3(self.detected_pos_fin[self.entries][0][0],self.detected_pos_fin[self.entries][0][1],self.detected_pos_fin[self.entries][0][2]))
                self.entries += 1
                return True

        else:
            print("ni bilo ujemanj:")
            #ustvarimo nov entry
            self.detected_pos_fin[self.entries] = [center_point,center_point]
            self.detected_norm_fin[self.entries] =[norm,norm]
            act_norm = self.detected_pos_fin[self.entries][0]-self.detected_norm_fin[self.entries][0]
            #self.points_pub(Point(self.detected_pos_fin[found][0][0],self.detected_pos_fin[found][0][1],self.detected_pos_fin[found][0][2]))
            #! self.twist_pub.publish(Vector3(act_norm[1],act_norm[1],act_norm[2]),Vector3(self.detected_pos_fin[self.entries][0][0],self.detected_pos_fin[self.entries][0][1],self.detected_pos_fin[self.entries][0][2]))
            self.entries += 1
            return True

        #posljemo None nazaj da vemo da nismo dokoncno dolocili nobene tocke
        return False

#! =============================================== face detection end ===============================================
# ===================================================================================================================
#! =============================================== task_master start ================================================


    def recognize_speech(self, question):
        with self.mic as source:
            print('Adjusting mic for ambient noise...')
            self.sr.adjust_for_ambient_noise(source)
            self.say(question)
            print(f"\n{question}")
            audio = self.sr.listen(source)

        print('I am now processing the sounds you made.')
        recognized_text = ''
        try:
            recognized_text = self.sr.recognize_google(audio)
        except sr.RequestError as e:
            print('API is probably unavailable', e)
        except sr.UnknownValueError:
            print('Did not manage to recognize anything.')

        good = input(f"\nIs input [{recognized_text}] what you said?\n[y/n]: ")
        if good == "y":
            return recognized_text
        else:
            recognized_text = self.recognize_speech(question)
        # if recognized_text.endswith("stop"):
        #     recognized_text = self.recognize_speech(question)
        return recognized_text

    def processStage(self, indx):
        try:
            face = self.positions["face"][indx]
        except Exception as err:
            print(f"ProcessStage error: {err}")
            return

        stage = face["stage"]
        print(f"processStage({indx}):\t{stage}")

        if stage == "warning":
            # tell him where to go
            self.face_pub.publish(
                Vector3(face["averageNormal"][0]+face["averagePostion"][0],
                        face["averageNormal"][1]+face["averagePostion"][1],
                        face["averageNormal"][2]+face["averagePostion"][2]),
                Vector3(face["averagePostion"][0],face["averagePostion"][1],face["averagePostion"][2]))

            # TODO: check if he arrived
            while True:
                if self.listener():
                    break

            # check if it has a mask
            if module.has_mask(face["mask"]) == False:
                self.say("mask bad")
            # check for social distancing
            for i in range(len(self.positions["face"])):
                if i == indx:
                    continue
                else:
                    area_avg = face["averagePostion"]
                    dist_vector = area_avg - self.positions["face"][i]["averagePostion"]
                    dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                    if dist < 1:
                        self.say("distancing bad")
                        break
            # vocalize decision
            # self.say("no more warnings")

        if stage == "talk":
            # "right_vaccine"     --> "Greenzer" / "Rederna" / "StellaBluera" / "BlacknikV"
            # "age"                 --> 0 to 100
            age = self.positions["digits"][face["digits_index"]]["data"]
            #! from talking
            # "was_vaccinated"    --> 0 / 1
            was_vaccinated = self.recognize_speech("Have you been vaccinated?")
            while was_vaccinated!="no"and was_vaccinated!="yes":
                was_vaccinated = self.recognize_speech("Have you been vaccinated?")
            was_vaccinated = 0 if was_vaccinated=="no" else 1


            doctor = None
            tempDoctor = []
            # "doctor"            --> "red" / "green" / "blue" / "yellow"
            while doctor is None:
                doctor = self.recognize_speech("Who is your personal doctor?")
                tempDoctor = doctor.split(" ")
                if len(tempDoctor) == 2:
                    doctor = self.char_from_string(tempDoctor[1])
                else:
                    doctor = None

            # "hours_exercise"    --> 0 to 40
            while True:
                try:
                    hours_exercise = int(self.recognize_speech("How many hours per week do you exercise?"))
                    if 0 <= hours_exercise <= 40:
                        break
                except Exception as e:
                    pass
            #! from input
            # "was_vaccinated"    --> 0 / 1
            # was_vaccinated = input("was_vaccinated (yes/no): ")
            # while was_vaccinated!="no"and was_vaccinated!="yes":
            #     was_vaccinated = input("was_vaccinated (yes/no): ")
            # was_vaccinated = 0 if was_vaccinated=="no" else 1
            # doctor = None
            # tempDoctor = []

            # # "doctor"            --> "red" / "green" / "blue" / "yellow"
            # while doctor is None:
            #     doctor = input("doctor (red/green/blue/yellow): ")
            #     tempDoctor = doctor.split(" ")
            #     if len(tempDoctor) == 2:
            #         doctor = self.char_from_string(tempDoctor[1])
            #     else:
            #         doctor = None

            # # "hours_exercise"    --> 0 to 40
            # while True:
            #     try:
            #         hours_exercise = int(input("hours_exercise ( 0 <--> 40): "))
            #         if 0 <= hours_exercise <= 40:
            #             break
            #     except Exception as e:
            #         pass
            # right_vaccine = input("right_vaccine (Greenzer/Rederna/StellaBluera/BlacknikV): ")
            #!from QR
            # was_vaccinated = int(self.positions["QR"][face["QR_index"]]["data"].split(", ")[2])
            # doctor = self.char_from_string(self.positions["QR"][face["QR_index"]]["data"].split(", ")[3].lower())
            # hours_exercise = int(self.positions["QR"][face["QR_index"]]["data"].split(", ")[4])
            # right_vaccine = self.positions["QR"][face["QR_index"]]["data"].split(", ")[5]

            # print(was_vaccinated)
            # print(doctor)
            # print(hours_exercise)
            # print(right_vaccine)
            face["was_vaccinated"] = was_vaccinated
            # has already been vacinated
            face["doctor"] = doctor
            face["hours_exercise"] = hours_exercise

            if was_vaccinated == 1:
                print(f"nextStage({indx}):\t\t{face['stage']} --> done")
                face["stage"] = "done"
                return

            # face["right_vaccine"] = right_vaccine
            # vocalize decision
            self.say("data collected")

        if stage == "cylinder":
            destination_cylinder = None
            for cylinder in self.positions["cylinder"]:
                if module.chose_color(cylinder["color"]) == face["doctor"]:
                    destination_cylinder = cylinder

            temp_model = pickle.load(open(f"{modelsDir}{self.positions['QR'][destination_cylinder['QR_index']]['modelName']}.sav", "rb"))
            # tell him where to go
            self.cylinder_pub.publish(Point(destination_cylinder["averagePostion"][0],destination_cylinder["averagePostion"][1],destination_cylinder["averagePostion"][2]))

            # choose ring
            age = self.positions["digits"][face["digits_index"]]["data"]
            toPredict = np.array([[age, face["hours_exercise"]]])
            face["right_vaccine"] = temp_model.predict(toPredict)[0]

            # vocalize decition
            self.say(f"Going to {self.word_from_char(face['doctor'])} cylinder.")

            # TODO: check if he arrived
            while True:
                if self.listener():
                    break
            print(face["right_vaccine"])
            ringColor = self.word_from_vaccine(face["right_vaccine"].lower())
            self.say(f"Vaccine is at {ringColor} ring.")

        if stage == "ring":
            destination_ring = None
            for ring in self.positions["ring"]:
                if module.chose_color(ring["color"]) == self.char_from_string(self.word_from_vaccine(face["right_vaccine"].lower())):
                    destination_ring = ring

            # tell him where to go
            self.ring_pub.publish(Point(destination_ring["averagePostion"][0],destination_ring["averagePostion"][1],destination_ring["averagePostion"][2]))

            # vocalize decition
            self.say(f"Going to {self.word_from_vaccine(face['right_vaccine'].lower())} ring.")

            # TODO: check if he arrived
            while True:
                if self.listener():
                    break

        if stage == "vaccinate":
            # TODO: tell him where to go
            self.face_pub.publish(
                Vector3(face["averageNormal"][0]+face["averagePostion"][0],
                        face["averageNormal"][1]+face["averagePostion"][1],
                        face["averageNormal"][2]+face["averagePostion"][2]),
                Vector3(face["averagePostion"][0],face["averagePostion"][1],face["averagePostion"][2]))

            # vocalize decition
            self.say(f"Going back to the face.")

            # TODO: check if he arrived
            while True:
                if self.listener():
                    break

            # TODO: tell him to extend his hand

            # vocalize decition
            self.say(f"Here is you vaccination.")

        if stage == "done":
            # TODO: nothing
            pass
        self.nextStage(indx)

    def listener(self):
        try:
            link = rospy.wait_for_message("/sem_nekaj", String)
            print(link.stamp)
            print(rospy.Time.now())
            if link.data.startswith("p"):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False

    def listener_end(self):
        try:
            link = rospy.wait_for_message("/sem_nekaj", String)
            print(link)
            if link.data.startswith("k"):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False



    def nextStage(self, indx):
        face = self.positions["face"][indx]
        stageBefore = face["stage"]
        if face["stage"]=="warning":
            face["stage"] = "talk"

        elif face["stage"]=="talk":
            face["stage"] = "cylinder"

        elif face["stage"]=="cylinder":
            face["stage"] = "ring"

        elif face["stage"]=="ring":
            face["stage"] = "vaccinate"

        elif face["stage"]=="vaccinate":
            face["stage"] = "done"

        print(f"nextStage({indx}):\t\t{stageBefore} --> {face['stage']}")

    def say(self,statement):
        # TODO: actualy SAY IT !!!
        print(f"\n\t--> {statement}\n")

    def char_from_string(self, color_string):
        if color_string == "black":
            return "c"
        if color_string == "white":
            return "w"
        if color_string == "red":
            return "r"
        if color_string == "green":
            return "g"
        if color_string == "blue":
            return "b"
        if color_string == "yellow":
            return "y"
        return None

    def word_from_char(self, color_char):
        if color_char == "c":
            return "black"
        if color_char == "w":
            return "white"
        if color_char == "r":
            return "red"
        if color_char == "g":
            return "green"
        if color_char == "b":
            return "blue"
        if color_char == "y":
            return "yellow"

    def word_from_vaccine(self, vaccine):
        if vaccine == "Greenzer".lower():
            return "green"
        if vaccine == "Rederna".lower():
            return "red"
        if vaccine == "StellaBluera".lower():
            return "blue"
        if vaccine == "BlacknikV".lower():
            return "black"

    def brain(self):
        for indx in range(0,len(self.positions["face"])):
            print("----------------------------------------------------------------\n")
            for i in range(5):
                print()
                self.processStage(indx)
                if self.positions["face"][indx]["stage"]=="done":
                    break
                # text = input("Are you at next location yet?")

koncal_ilja = False
def callback(link):
    if if link.data.startswith("k"):
        koncal_ilja =  True

#! ================================================ task_master end ================================================
def main():


        color_finder = color_localizer()
        rospy.Subscriber("/sem_nekaj", String, callback)
        # color_finder.positions = tempKrNeki

        # rate = rospy.Rate(1.25)
        rate = rospy.Rate(10)

        #! TESTING
        explore = True

        skipCounter = 3
        loopTimer = rospy.Time.now().to_sec()
        # print(sleepTimer)
        while not rospy.is_shutdown():
            # print("hello!")
            if explore:
                if skipCounter <= 0:
                    #preverja ce je ze prisel signal koncal
                    explore = color_finder.find_objects(koncal_ilja)
                else:
                    skipCounter -= 1
            else:
                # text = color_finder.recognize_speech()
                # print('I recognized this sentence:', text)
                print("we done")
                #preverja ce je ze prisel signal koncal
                while True and not koncal_ilja:
                    if color_finder.listener_end():
                        break
                color_finder.brain()
                break
            rate.sleep()

        cv2.destroyAllWindows()








tempKrNeki = {'ring': [{'averagePostion':np.array([ 2.81610412, -0.94318618,  0.2972    ]), 'detectedPositions': [np.array([ 2.94788736, -0.95712757,  0.2972    ]),np.array([ 2.86090865, -1.05065474,  0.2972    ]),np.array([ 2.65596364, -0.88268708,  0.2972    ]),np.array([ 2.63763366, -0.8886651 ,  0.2972    ]),np.array([ 2.68567003, -0.88327188,  0.2972    ]),np.array([ 2.92282008, -0.96124302,  0.2972    ]),np.array([ 2.92007471, -0.96469404,  0.2972    ]),np.array([ 2.89787487, -0.95714603,  0.2972    ])], 'color': {'r': 0, 'g': 0, 'b': 8, 'y': 0, 'w': 0, 'c': 0}, 'approached': False, 'avgMarkerId': 87}, {'averagePostion':np.array([-1.21440732,  1.05342329,  0.2972    ]), 'detectedPositions': [np.array([-1.22989409,  1.05515189,  0.2972    ]),np.array([-1.21839109,  1.05447691,  0.2972    ]),np.array([-1.21624633,  1.05172252,  0.2972    ]),np.array([-1.19309776,  1.05234185,  0.2972    ])], 'color': {'r': 0, 'g': 4, 'b': 0, 'y': 0, 'w': 0, 'c': 0}, 'approached': False, 'avgMarkerId': 82}, {'averagePostion':np.array([1.3828572 , 1.92886465, 0.2972    ]), 'detectedPositions': [np.array([1.38985897, 1.93085347, 0.2972    ]),np.array([1.38180099, 1.919463  , 0.2972    ]),np.array([1.37538029, 1.90476483, 0.2972    ]),np.array([1.38438855, 1.96037731, 0.2972    ])], 'color': {'r': 0, 'g': 1, 'b': 0, 'y': 0, 'w': 0, 'c': 3}, 'approached': False, 'avgMarkerId': 187}, {'averagePostion':np.array([1.6669117 , 2.33890599, 0.2972    ]), 'detectedPositions': [np.array([1.6257893 , 2.38255613, 0.2972    ]),np.array([1.68742498, 2.31719537, 0.2972    ]),np.array([1.68752081, 2.31696647, 0.2972    ])], 'color': {'r': 3, 'g': 0, 'b': 0, 'y': 0, 'w': 0, 'c': 0}, 'approached': False, 'avgMarkerId': 255}], 'cylinder': [{'averagePostion':np.array([-1.76331041,  0.15568962,  0.2972    ]), 'detectedPositions': [np.array([-1.79645995,  0.09618325,  0.2972    ]),np.array([-1.76878749,  0.15839875,  0.2972    ]),np.array([-1.7704159 ,  0.17945724,  0.2972    ]),np.array([-1.76090852,  0.15240529,  0.2972    ]),np.array([-1.76069804,  0.15299632,  0.2972    ]),np.array([-1.7568068 ,  0.16253023,  0.2972    ]),np.array([-1.75680562,  0.16251377,  0.2972    ]),np.array([-1.75681537,  0.1625936 ,  0.2972    ]),np.array([-1.75667243,  0.16230303,  0.2972    ]),np.array([-1.75630019,  0.16198642,  0.2972    ]),np.array([-1.75574419,  0.16121792,  0.2972    ])], 'color': {'r': 0, 'g': 11, 'b': 0, 'y': 0, 'w': 0, 'c': 0}, 'approached': True, 'avgMarkerId': 2, 'QR_index': 0}, {'averagePostion':np.array([0.39332062, 1.32513901, 0.2972    ]), 'detectedPositions': [np.array([0.4176197 , 1.19432779, 0.2972    ]),np.array([0.40388624, 1.33765771, 0.2972    ]),np.array([0.40283732, 1.33790291, 0.2972    ]),np.array([0.39745508, 1.34132422, 0.2972    ]),np.array([0.39476845, 1.3422516 , 0.2972    ]),np.array([0.38869529, 1.34281598, 0.2972    ]),np.array([0.38757439, 1.34208728, 0.2972    ]),np.array([0.38274938, 1.34129872, 0.2972    ]),np.array([0.3642997 , 1.34658489, 0.2972    ])], 'color': {'r': 0, 'g': 0, 'b': 9, 'y': 0, 'w': 0, 'c': 0}, 'approached': True, 'avgMarkerId': 28, 'QR_index': 1}, {'averagePostion':np.array([ 3.93364722, -0.50891001,  0.2972    ]), 'detectedPositions': [np.array([ 3.93989992, -0.50080574,  0.2972    ]),np.array([ 3.93700886, -0.49459029,  0.2972    ]),np.array([ 3.93318318, -0.4979407 ,  0.2972    ]),np.array([ 3.93097849, -0.49781012,  0.2972    ]),np.array([ 3.93095283, -0.50847619,  0.2972    ]),np.array([ 3.93082039, -0.50863221,  0.2972    ]),np.array([ 3.93076723, -0.50853788,  0.2972    ]),np.array([ 3.93160251, -0.51048583,  0.2972    ]),np.array([ 3.93089371, -0.51004031,  0.2972    ]),np.array([ 3.93502493, -0.50829406,  0.2972    ]),np.array([ 3.93503451, -0.50951552,  0.2972    ]),np.array([ 3.93760013, -0.55179125,  0.2972    ])], 'color': {'r': 0, 'g': 0, 'b': 0, 'y': 12, 'w': 0, 'c': 0}, 'approached': True, 'avgMarkerId': 98, 'QR_index': 3}, {'averagePostion':np.array([ 0.92574202, -0.12808591,  0.2972    ]), 'detectedPositions': [np.array([ 0.88289588, -0.14672508,  0.2972    ]),np.array([ 0.93138161, -0.14040482,  0.2972    ]),np.array([ 0.92563918, -0.12957609,  0.2972    ]),np.array([ 0.92691301, -0.1277871 ,  0.2972    ]),np.array([ 0.93207716, -0.11524721,  0.2972    ]),np.array([ 0.93215079, -0.11525347,  0.2972    ]),np.array([ 0.94913653, -0.12160759,  0.2972    ])], 'color': {'r': 7, 'g': 0, 'b': 0, 'y': 0, 'w': 0, 'c': 0}, 'approached': True, 'avgMarkerId': 168, 'QR_index': 5}], 'face': [{'averagePostion':np.array([-0.10987265, -1.64377577,  0.2972    ]), 'averageNormal':np.array([-0.0195094 , -0.99952619, -0.02380723]), 'detectedPositions': [np.array([-0.11775418, -1.66505193,  0.2972    ]),np.array([-0.12022563, -1.66493652,  0.2972    ]),np.array([-0.11158538, -1.65688679,  0.2972    ]),np.array([-0.11266341, -1.65680778,  0.2972    ]),np.array([-0.10873119, -1.65193661,  0.2972    ]),np.array([-0.11038316, -1.65194426,  0.2972    ]),np.array([-0.11791542, -1.64677141,  0.2972    ]),np.array([-0.109031  , -1.62706538,  0.2972    ]),np.array([-0.1090294 , -1.62704638,  0.2972    ]),np.array([-0.08140773, -1.58931061,  0.2972    ])], 'detectedNormals': [np.array([-0.01733354, -0.99963262, -0.0208367 ]),np.array([-0.01919565, -0.99953873, -0.02353405]),np.array([-0.01297902, -0.99968169, -0.02163491]),np.array([-0.01402521, -0.99961293, -0.02402667]),np.array([-0.01393909, -0.99955349, -0.02642963]),np.array([-0.01664269, -0.99952817, -0.02581585]),np.array([-0.00898452, -0.99963834, -0.02534678]),np.array([ 0.15996332, -0.98681653, -0.02459399]),np.array([ 0.15999874, -0.98678887, -0.0254584 ]),np.array([-0.40976143, -0.91202066, -0.01771692])], 'approached': True, 'avgMarkerId': 53, 'QR_index': 2, 'digits_index': 0, 'mask': {'yes': 10, 'no': 0}, 'stage': 'warning', 'was_vaccinated': None, 'doctor': None, 'hours_exercise': None, 'right_vaccine': None}, {'averagePostion':np.array([0.55852742, 1.25890332, 0.2972    ]), 'averageNormal':np.array([-0.9833539 , -0.18030703, -0.02246052]), 'detectedPositions': [np.array([0.50497741, 1.22345543, 0.2972    ]),np.array([0.54067596, 1.2363011 , 0.2972    ]),np.array([0.54069273, 1.23629524, 0.2972    ]),np.array([0.54474395, 1.23482981, 0.2972    ]),np.array([0.51367617, 1.28729746, 0.2972    ]),np.array([0.61797551, 1.26798151, 0.2972    ]),np.array([0.59469486, 1.26809589, 0.2972    ]),np.array([0.59080309, 1.2723921 , 0.2972    ]),np.array([0.59061658, 1.27289203, 0.2972    ]),np.array([0.5894829 , 1.28261644, 0.2972    ]),np.array([0.59110015, 1.28353196, 0.2972    ]),np.array([0.54365138, 1.26261393, 0.2972    ]),np.array([0.54691253, 1.26245129, 0.2972    ]),np.array([0.50938066, 1.23389232, 0.2972    ])], 'detectedNormals': [np.array([-0.99192563,  0.1242659 , -0.02532862]),np.array([-0.97727719,  0.21043401, -0.02543268]),np.array([-0.97718786,  0.21045871, -0.02847836]),np.array([-0.97157064,  0.2353132 , -0.02604206]),np.array([-0.98476775,  0.17128369, -0.02990597]),np.array([-0.9422557 , -0.33468395, -0.01186838]),np.array([-0.93531253, -0.35348883, -0.01536611]),np.array([-0.92529251, -0.37887239, -0.01701425]),np.array([-0.92560136, -0.37811592, -0.0170433 ]),np.array([-0.92321824, -0.383953  , -0.01575338]),np.array([-0.91925531, -0.3933024 , -0.01681975]),np.array([-0.90031707, -0.43461316, -0.02325028]),np.array([-0.88543568, -0.46420116, -0.02282397]),np.array([-0.96640618, -0.25560173, -0.02696033])], 'approached': True, 'avgMarkerId': 129, 'QR_index': 4, 'digits_index': 1, 'mask': {'yes': 12, 'no': 2}, 'stage': 'warning', 'was_vaccinated': None, 'doctor': None, 'hours_exercise': None, 'right_vaccine': None}, {'averagePostion':np.array([2.01092963, 3.41776147, 0.2972    ]), 'averageNormal':np.array([-0.18313601,  0.98284524, -0.02182731]), 'detectedPositions': [np.array([1.89444299, 3.28694764, 0.2972    ]),np.array([2.0348282 , 3.43939542, 0.2972    ]),np.array([2.01737646, 3.46098661, 0.2972    ]),np.array([2.07221089, 3.41558804, 0.2972    ]),np.array([2.01212729, 3.39628661, 0.2972    ]),np.array([1.97859235, 3.4533621 , 0.2972    ]),np.array([2.01643697, 3.45514253, 0.2972    ]),np.array([2.00139996, 3.46289258, 0.2972    ]),np.array([2.00850108, 3.42011224, 0.2972    ]),np.array([2.01718701, 3.41557275, 0.2972    ]),np.array([2.01879837, 3.41472104, 0.2972    ]),np.array([2.02049298, 3.41428231, 0.2972    ]),np.array([2.02100065, 3.414506  , 0.2972    ]),np.array([2.0203617 , 3.41139443, 0.2972    ]),np.array([2.02037981, 3.41141539, 0.2972    ]),np.array([2.02073736, 3.41157785, 0.2972    ])], 'detectedNormals': [np.array([ 0.39050649,  0.92029093, -0.02385979]),np.array([-0.00244821,  0.99982143, -0.01873794]),np.array([ 0.00297869,  0.9998238 , -0.01853373]),np.array([ 0.02060236,  0.99962508, -0.01803462]),np.array([-0.34721947,  0.93765931, -0.01528571]),np.array([-0.12396776,  0.99196824, -0.02511997]),np.array([ 0.11852   ,  0.99266734, -0.02376067]),np.array([-0.08518164,  0.99603943, -0.02548598]),np.array([-0.33486498,  0.94201519, -0.02174465]),np.array([-0.33088447,  0.9434182 , -0.02185321]),np.array([-0.34482329,  0.93841888, -0.02160807]),np.array([-0.3478451 ,  0.93731464, -0.02109614]),np.array([-0.34787823,  0.93729687, -0.02133821]),np.array([-0.37421268,  0.92710287, -0.02109828]),np.array([-0.374164  ,  0.92711735, -0.02132422]),np.array([-0.37336269,  0.92744076, -0.02130582])], 'approached': True, 'avgMarkerId': 204, 'QR_index': 7, 'digits_index': 3, 'mask': {'yes': 4, 'no': 12}, 'stage': 'warning', 'was_vaccinated': None, 'doctor': None, 'hours_exercise': None, 'right_vaccine': None}, {'averagePostion':np.array([1.34648599, 3.4172927 , 0.2972    ]), 'averageNormal':np.array([-0.00414711,  0.9996854 , -0.02473658]), 'detectedPositions': [np.array([1.33429133, 3.42768026, 0.2972    ]),np.array([1.33194286, 3.42645473, 0.2972    ]),np.array([1.32741998, 3.42467002, 0.2972    ]),np.array([1.32914479, 3.42213254, 0.2972    ]),np.array([1.3407501 , 3.42029512, 0.2972    ]),np.array([1.35661575, 3.34223238, 0.2972    ]),np.array([1.35431968, 3.43484719, 0.2972    ]),np.array([1.37666322, 3.44133852, 0.2972    ]),np.array([1.36722621, 3.41598353, 0.2972    ])], 'detectedNormals': [np.array([ 0.07895869,  0.99654188, -0.02588052]),np.array([ 0.09157424,  0.99545184, -0.0262639 ]),np.array([ 0.10643157,  0.99397233, -0.02629321]),np.array([ 0.12998346,  0.9911677 , -0.02628467]),np.array([ 0.15268431,  0.98792634, -0.02624983]),np.array([-0.38587945,  0.92245552, -0.01314773]),np.array([ 0.07897771,  0.99655067, -0.02548096]),np.array([-0.0193345 ,  0.99946466, -0.02639253]),np.array([-0.27009332,  0.96256184, -0.02289761])], 'approached': True, 'avgMarkerId': 194, 'QR_index': 6, 'digits_index': 2, 'mask': {'yes': 3, 'no': 6}, 'stage': 'warning', 'was_vaccinated': None, 'doctor': None, 'hours_exercise': None, 'right_vaccine': None}], 'QR': [{'averagePostion':np.array([-1.74753383,  0.13118945,  0.2972    ]), 'averageNormal':np.array([-0.99572036, -0.09176526, -0.01095941]), 'detectedPositions': [np.array([-1.74768986,  0.13093603,  0.2972    ]),np.array([-1.74760392,  0.13099825,  0.2972    ]),np.array([-1.74730769,  0.13163406,  0.2972    ])], 'detectedNormals': [np.array([-0.9957258 , -0.09172747, -0.01077989]),np.array([-0.99564027, -0.09262158, -0.01103135]),np.array([-0.99579427, -0.09094666, -0.01106699])], 'data': 'https://box.vicos.si/rins/18.txt', 'avgMarkerId': 19, 'isAssigned': True, 'modelName': 0}, {'averagePostion':np.array([0.40624585, 1.31527475, 0.2972    ]), 'averageNormal':np.array([ 0.52933732,  0.84834414, -0.01068711]), 'detectedPositions': [np.array([0.41058279, 1.31608189, 0.2972    ]),np.array([0.40536772, 1.31702328, 0.2972    ]),np.array([0.40281817, 1.31658946, 0.2972    ]),np.array([0.40621475, 1.31140436, 0.2972    ])], 'detectedNormals': [np.array([ 0.60362334,  0.79720226, -0.01036452]),np.array([ 0.58325691,  0.8122181 , -0.01063679]),np.array([ 0.57381114,  0.81891761, -0.01071108]),np.array([ 0.34102938,  0.93999151, -0.0107205 ])], 'data': 'https://box.vicos.si/rins/12.txt', 'avgMarkerId': 43, 'isAssigned': True, 'modelName': 3}, {'averagePostion':np.array([-0.32594343, -1.61965095,  0.2972    ]), 'averageNormal':np.array([-0.21015762, -0.97757031, -0.01378633]), 'detectedPositions': [np.array([-0.32669599, -1.61278376,  0.2972    ]),np.array([-0.32519086, -1.62651815,  0.2972    ])], 'detectedNormals': [np.array([-0.28401853, -0.9587201 , -0.01375656]),np.array([-0.13507553, -0.99074009, -0.01373599])], 'data': '1, 82, 1, Red, 5, Rederna', 'avgMarkerId': 71, 'isAssigned': True, 'modelName': 7}, {'averagePostion':np.array([ 3.92476412, -0.47046035,  0.2972    ]), 'averageNormal':np.array([ 0.99923889,  0.03789093, -0.00926908]), 'detectedPositions': [np.array([ 3.92746362, -0.4779259 ,  0.2972    ]),np.array([ 3.92745693, -0.47789599,  0.2972    ]),np.array([ 3.92745065, -0.47789311,  0.2972    ]),np.array([ 3.91668527, -0.44812641,  0.2972    ])], 'detectedNormals': [np.array([ 0.99770853,  0.06674325, -0.01109159]),np.array([ 0.99770177,  0.06683636, -0.0111391 ]),np.array([ 0.9977018 ,  0.06684447, -0.01108802]),np.array([ 0.99878933, -0.049052  , -0.00371073])], 'data': 'https://box.vicos.si/rins/14.txt', 'avgMarkerId': 121, 'isAssigned': True, 'modelName': 9}, {'averagePostion':np.array([0.50728764, 1.46832644, 0.2972    ]), 'averageNormal':np.array([-0.99592154, -0.08930725, -0.01282561]), 'detectedPositions': [np.array([0.57667232, 1.44974484, 0.2972    ]),np.array([0.49084217, 1.47613624, 0.2972    ]),np.array([0.4891646, 1.4773171, 0.2972   ]),np.array([0.4897497 , 1.47742077, 0.2972    ]),np.array([0.49000942, 1.46101325, 0.2972    ])], 'detectedNormals': [np.array([-0.92968381, -0.36824925, -0.00897242]),np.array([-0.99885835, -0.04591345, -0.01318912]),np.array([-0.99826826, -0.05721956, -0.01365281]),np.array([-0.99863187, -0.05041007, -0.01390024]),np.array([-0.99666792,  0.08041205, -0.01367302])], 'data': '1, 75, 0, Blue, 32, StellaBluera', 'avgMarkerId': 151, 'isAssigned': True, 'modelName': 13}, {'averagePostion':np.array([ 0.9025056 , -0.10311455,  0.2972    ]), 'averageNormal':np.array([-0.04777953, -0.99865997, -0.01988437]), 'detectedPositions': [np.array([ 0.90370024, -0.10218351,  0.2972    ]),np.array([ 0.90379938, -0.10217816,  0.2972    ]),np.array([ 0.90001719, -0.10498199,  0.2972    ])], 'detectedNormals': [np.array([-0.2015942 , -0.97940044, -0.0115998 ]),np.array([-0.20165693, -0.97938724, -0.01162431]),np.array([ 0.26341196, -0.9640493 , -0.03497264])], 'data': 'https://box.vicos.si/rins/2.txt', 'avgMarkerId': 179, 'isAssigned': True, 'modelName': 18}, {'averagePostion':np.array([1.59715658, 3.39828323, 0.2972    ]), 'averageNormal':np.array([ 0.05586412,  0.99837225, -0.01149107]), 'detectedPositions': [np.array([1.59780757, 3.40452062, 0.2972    ]),np.array([1.59650559, 3.39204585, 0.2972    ])], 'detectedNormals': [np.array([ 0.31332936,  0.94954097, -0.01402346]),np.array([-0.20543783,  0.97863607, -0.00816947])], 'data': '0, 15, 0, Green, 22, Greenzer', 'avgMarkerId': 219, 'isAssigned': True, 'modelName': 21}, {'averagePostion':np.array([2.20407253, 3.45163222, 0.2972    ]), 'averageNormal':np.array([ 0.01684339,  0.99976425, -0.01370207]), 'detectedPositions': [np.array([2.17047621, 3.44369541, 0.2972    ]),np.array([2.2208583 , 3.45559684, 0.2972    ]),np.array([2.22088309, 3.4556044 , 0.2972    ])], 'detectedNormals': [np.array([ 0.17726722,  0.98405243, -0.01473612]),np.array([-0.06316249,  0.99791737, -0.01309293]),np.array([-0.06390226,  0.99787135, -0.01301056])], 'data': '0, 20, 1, Yellow, 20, BlacknikV', 'avgMarkerId': 245, 'isAssigned': True, 'modelName': 23}], 'digits': [{'averagePostion':np.array([ 0.115308  , -1.63104736,  0.2972    ]), 'averageNormal':np.array([ 0.03829475, -0.99899503, -0.0232902 ]), 'detectedPositions': [np.array([ 0.11071775, -1.62916305,  0.2972    ]),np.array([ 0.09532647, -1.61584434,  0.2972    ]),np.array([ 0.12230288, -1.63219802,  0.2972    ]),np.array([ 0.13288489, -1.64698404,  0.2972    ])], 'detectedNormals': [np.array([ 0.21566623, -0.97610426, -0.02661845]),np.array([ 0.2886732 , -0.95705655, -0.02665609]),np.array([-0.18629604, -0.98234199, -0.01726255]),np.array([-0.16850559, -0.98548939, -0.02040921])], 'data': 82, 'avgMarkerId': 65, 'isAssigned': True}, {'averagePostion':np.array([0.50310674, 1.0883997 , 0.2972    ]), 'averageNormal':np.array([-0.9918287 ,  0.12612405, -0.01919758]), 'detectedPositions': [np.array([0.48700087, 1.10885583, 0.2972    ]),np.array([0.5192126 , 1.06794357, 0.2972    ])], 'detectedNormals': [np.array([-0.99909868,  0.03215107, -0.02771527]),np.array([-0.97567559,  0.21896742, -0.01050795])], 'data': 75, 'avgMarkerId': 163, 'isAssigned': True}, {'averagePostion':np.array([1.12443148, 3.40945943, 0.2972    ]), 'averageNormal':np.array([-0.2512796 ,  0.9675397 , -0.02693483]), 'detectedPositions': [np.array([1.10898366, 3.42951349, 0.2972    ]),np.array([1.11808611, 3.40749867, 0.2972    ]),np.array([1.14622467, 3.39136613, 0.2972    ])], 'detectedNormals': [np.array([-0.08054085,  0.99641107, -0.02604131]),np.array([-0.2784663 ,  0.96005311, -0.02746911]),np.array([-0.38833371,  0.9211349 , -0.02659756])], 'data': 15, 'avgMarkerId': 208, 'isAssigned': True}, {'averagePostion':np.array([1.81034267, 3.4328791 , 0.2972    ]), 'averageNormal':np.array([-0.26241263,  0.96458337, -0.02680538]), 'detectedPositions': [np.array([1.81634443, 3.44949933, 0.2972    ]),np.array([1.80434092, 3.41625886, 0.2972    ])], 'detectedNormals': [np.array([-0.15723845,  0.98720276, -0.02658524]),np.array([-0.36454983,  0.93080058, -0.0267153 ])], 'data': 20, 'avgMarkerId': 231, 'isAssigned': True}]}

if __name__ == '__main__':
    main()
