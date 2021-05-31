#!/usr/bin/python3

import itertools
import sys
import rospy
import dlib
import os
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


# /home/sebastjan/Documents/faks/3letnk/ris/ROS_task/src/exercise4/scripts

#modelsDir = "/home/code8master/Desktop/wsROS/src/RIS/exercise4/scripts"
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'
class color_localizer:

    def __init__(self):
        print()
        self.found_rings = []
        self.found_cylinders = []

        self.basePosition = {"x":0, "y":0, "z":0}
        self.baseAngularSpeed = 0

        self.random_forest_HSV = pickle.load(open(f"{modelsDir}/random_forest_HSV.sav", 'rb'))
        self.random_forest_RGB = pickle.load(open(f"{modelsDir}/random_forest_RGB.sav", 'rb'))
        self.knn_HSV = pickle.load(open(f"{modelsDir}/knn_HSV.sav", 'rb'))
        self.knn_RGB = pickle.load(open(f"{modelsDir}/knn_RGB.sav", 'rb'))

        self.positions = {
            "ring": [],
            "cylinder": [],
            "face": [],
            "QR": []
        }

        rospy.init_node('color_localizer', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.m_arr = MarkerArray()
        self.nM = 0
        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('ring_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('face_im', Image, queue_size=1000)
        self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        # self.twist_pub = rospy.Publisher('/our_pub1/chat1', Twist, queue_size=1000)

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
        self.showEveryDetection = False

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
            assert self.positions[closestObjectKey][closestObjectIndx]["QR_index"] is None, "Objekt ima že določeno svojo QR kodo"
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
        print(f"rgb_stamp_sec:    {im_stamp.to_sec()}")
        print(f"depth_stamp_sec:  {depth_stamp.to_sec()}")
        diff = (depth_stamp.to_sec()-im_stamp.to_sec())
        print(f"diff:             {diff}")
        print()

        if (np.abs(diff) > 0.5):
            pprint("skip")
            return grayBGR_toDrawOn,depth_image

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
                # print("dela 1")
                color = module.calc_rgb(pnts)
                # print("dela 2")
                ring_point = module.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im_shifted,"ring",depth_stamp,color,self.tf_buf)
                #ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im_shifted,"ring",depth_stamp,color)
                # print("dela 3")
                (self.nM, slef.m_arr) = module.addPosition(np.array((ring_point.position.x,ring_point.position.y,ring_point.position.z)),"ring",color,self.positions["ring"],self.nM, self.m_arr, self.markers_pub)
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
                # print(f"Ring error: {e}")
                pass
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
            if (np.isnan(depth_line[i]) or ( np.isnan(depth_line[i-1]) and np.isnan(depth_line[i+1])) ):
                continue
            if (np.isnan(depth_line[i-1]) or depth_line[i-1]>depth_line[i]) and (np.isnan(depth_line[i+1]) or depth_line[i+1]>depth_line[i]):
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
            return None
        if desno == -1:
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
        if shift < 1 or shift > 50:
            return None
        #vzamemo levo 3 in 6 in desno 3 in 6 oddaljena pixla ter računamo razmerje razmerja rabita biti priblizno enaka na obeh straneh
        LL = depth_line[center_point - shift*2]
        L = depth_line[center_point - shift]
        RR = depth_line[center_point + shift*2]
        R = depth_line[center_point + shift]
        C = depth_line[center_point]
        LL_C = LL-C
        L_C = L-C
        C_R = R-C
        C_RR = RR-C

        if LL_C<=0 or L_C<=0 or C_R<=0 or C_RR<=0:
            return None

        if LL_C<L_C*3 or C_RR<C_R*3:
            return None
        return (center_point-shift*2,center_point+shift*2)

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
                dolz -= 1
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
                dolz -=1

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

    def find_cylinderDEPTH(self,image, depth_image, grayBGR_toDrawOn,depth_stamp):
        # print(depth_stamp)
        centerRowIndex = depth_image.shape[0]//2
        max_row = (depth_image.shape[0]//5)*2
        acum_me= []
        curr_ints = []
        actual_add = (-1,-1)
        """rabim narediti count za vsak mozn cilinder v sliki posebi!!"""
        #count_ujemanj drži število zaznanih vrstic s cilindri(prekine in potrdi pri 3eh)
        #GRABO NOT BEING USED
        count = 0
        count_ujemanj = [0]
        #END OF GARBO
        for rows in range(centerRowIndex, max_row, -1):
            presekiIndex,blizDalec = self.getLows(depth_image[rows,:])
            cnt = 0
            acum_me.append([])
            for cnt in range(1,len(presekiIndex)-1):
                #se pojavi kot najblizja točka katere desni in levi pixl sta  za njo
                if blizDalec[cnt]:
                    #notri shranjen levo in desno interval rows vsebuje kera vrstica je trenutno
                    interval = self.getRange(depth_image[rows,:],presekiIndex[cnt],presekiIndex[cnt-1],presekiIndex[cnt+1])
                    if interval != None:
                        #grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(interval[0],centerRowIndex), radius=2,color=[255,0,0], thickness=-1)
                        #grayBGR_toDrawOn = cv2.circle(grayBGR_toDrawOn,(interval[1],centerRowIndex), radius=2,color=[255,0,0], thickness=-1)

                        #izrise crto intervala

                        #grayBGR_toDrawOn[centerRowIndex,list(range(interval[0],interval[1]))] = [255,0,0]
                        #NASLI SMO CILINDER
                        # print(depth_image)

                        try:
                            #preverimo da je realen interval
                            if np.abs(interval[0]-interval[1]) <= 20 or depth_image[rows,presekiIndex[cnt]]>2.5:
                                continue


                            # interval je kul
                            #1. ga dodamo v akumulator
                            acum_me[count].append(interval)



                            for i in range(interval[0],interval[1]):
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
            for i in range(inter[0][0],inter[0][1]):
                grayBGR_toDrawOn[inter[1],i] = [0,0,255]
            points = np.array([ image[inter[1],(inter[0][0]+inter[0][1])//2],
                                image[inter[1],(inter[0][0]+inter[0][1])//2+1],
                                image[inter[1],(inter[0][0]+inter[0][1])//2-1]])

            print(f"Širina:{inter[0][0]} {inter[0][1]}\n\tna razdalji:{depth_image[inter[1],(inter[0][0]+inter[0][1])//2]}")
            colorToPush = self.calc_rgb(points)
            pose = module.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush,self.tf_buf)
            #pose = self.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush)
            (self.nM, slef.m_arr) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush,self.positions["cylinder"],self.nM, self.m_arr, self.markers_pub)
            #self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush)

        return grayBGR_toDrawOn

    #NE OHRANI NAPREJ



    def find_objects(self):
        #print('I got a new image!')
        print("--------------------------------------------------------")

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
        # TODO: make it so it marks face and returns the image to display
        markedImage = self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)
        self.find_faces(rgb_image,depth_im_shifted,depth_image_message.header.stamp)
        markedImage = self.find_QR(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
        markedImage = self.find_digits(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
        #!
        """
        for objectType in ["ring","cylinder"]:
            module.checkPosition(self.positions[objectType],self.basePosition, objectType, self.points_pub)
        """
        # print(type(markedImage))
        # print(markedImage.shape)
        #print(markedImage)
        self.pic_pub.publish(CvBridge().cv2_to_imgmsg(markedImage, encoding="passthrough"))
        #self.markers_pub.publish(self.m_arr)

    def find_elipse(self,contours):
        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        # candidates = []
        # for n in range(len(elps)):
        #     for m in range(n + 1, len(elps)):
        #         e1 = elps[n]
        #         e2 = elps[m]
        #         dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
        #         #             print dist
        #         if dist < 5:
        #             candidates.append((e1,e2))

        return elps
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

            x1 = round(d1["start_point"][0]).astype("int")
            y1 = round(min(d1["start_point"][1],d2["start_point"][1])).astype("int")

            x2 = round(d2["end_point"][0]).astype("int")
            y2 = round(max(d1["end_point"][1],d2["end_point"][1])).astype("int")
            print(type(x1))
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
            norm = module.get_normal(depth_image_shifted, (x1,y1,x2,y2),stamp,None,None, tf_buff)
            # if we are too close to QR code
            if norm is None:
                print(f"Too close to the QR code!")
                continue
            pose = module.get_pose((x1+x2)//2,(y1+y2)//2,depth_image_shifted[(y1+y2)//2,(x1+x2)//2],depth_image_shifted,"QR",stamp,None,self.tf_buf)
            #pose = self.get_pose((x1+x2)//2,(y1+y2)//2,depth_image_shifted[(y1+y2)//2,(x1+x2)//2],depth_image_shifted,"QR",stamp,None)

            (self.nM, slef.m_arr) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"QR", None,self.positions["QR"],self.nM, self.m_arr, self.markers_pub,norm, dObject.data.decode())
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

            """im_ms = Image()
            im_ms.header.stamp = rospy.Time(0)
            im_ms.header.frame_id = 'map'
            im_ms.height = face_region.shape[0]
            im_ms.width = face_region.shape[1]
            im_ms.data = face_region"""

            #self.pic_pub.publish(im_ms)
            #self.pic_pub.publish(CvBridge().cv2_to_imgmsg(face_region, encoding="passthrough"))

            # Visualize the extracted face
            #cv2.imshow("ImWindow", face_region)
            #cv2.waitKey(1)

            #self.pic_pub = rospy.Publisher('face_im', Image, queue_size=1000)

            # Find the distance to the detected face
            #assert y1 != y2, "Y sta ista !!!!!!!!!!!"
            #assert x1 != x2, "X sta ista !!!!!!!!!!!"
            face_distance = float(np.nanmean(depth_image[y1:y2,x1:x2]))
            im = depth_image[y1:y2,x1:x2]
            norm = module.get_normal(depth_image, (x1,y1,x2,y2) ,depth_image_stamp,face_distance, face_region, tf_buff)
            normals_found.append(norm)

            print('Norm of face', norm)

            print('Distance to face', face_distance)
            #trans = tf2_ros.Buffer().lookup_transform('mapa', 'base_link', rospy.Time(0))
            #print('my position ',trans)
            # Get the time that the depth image was recieved
            depth_time = depth_image_stamp

            # Find the location of the detected face
            pose = module.get_pose_face((x1,x2,y1,y2), face_distance, depth_time,self.tf_buf)
            #pose = self.get_pose_face((x1,x2,y1,y2), face_distance, depth_time)
            if pose != None:
                newPosition = np.array([pose.position.x,pose.position.y, pose.position.z])
                (self.nM, slef.m_arr) = module.addPosition(newPosition,"face", color_char=None, self.positions["face"],self.nM, self.m_arr, self.markers_pub, normal=norm)
                #self.addPosition(newPosition, "face", color_char=None, normal=norm)


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

def main():

        color_finder = color_localizer()

        rate = rospy.Rate(1.25)
        skipCounter = 3
        while not rospy.is_shutdown():
            # print("hello!")
            if skipCounter <= 0:
                color_finder.find_objects()
            else:
                skipCounter -= 1
            #print("hello")
            #print("\n-----------\n")
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
