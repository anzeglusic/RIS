#!/usr/bin/python3

import os
import itertools
import sys
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


class cylinders:
    def __init__(self):
        print()
        #self.found_rings = []
        self.found_cylinders = []

        self.basePosition = {"x":0, "y":0, "z":0}
        self.baseAngularSpeed = 0

        self.random_forest_HSV = pickle.load(open(f"{modelsDir}/random_forest_HSV.sav", 'rb'))
        self.random_forest_RGB = pickle.load(open(f"{modelsDir}/random_forest_RGB.sav", 'rb'))
        self.knn_HSV = pickle.load(open(f"{modelsDir}/knn_HSV.sav", 'rb'))
        self.knn_RGB = pickle.load(open(f"{modelsDir}/knn_RGB.sav", 'rb'))


        rospy.init_node('cylinders', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.m_arr = MarkerArray()
        self.nM = 0
        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('ring_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('cylinder_im', Image, queue_size=1000)

        #!!!!!!!!!!!!!!!spremeni da so vsi poslani z dvema tockama per iljas request
        self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        # self.twist_pub = rospy.Publisher('/our_pub1/chat1', Twist, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)


        #! togle za sive markerje
        self.showEveryDetection = False

        #! cylinder
        #element: (interval,vrstic_od_sredine)
        self.tru_intervals = []

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
            colorToPush = module.calc_rgb(points,self.knn_RGB,self.random_forest_RGB,self.knn_HSV,self.random_forest_HSV)
            pose = module.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush,self.tf_buf)
            #pose = self.get_pose((inter[0][0]+inter[0][1])//2,inter[1],depth_image[inter[1],(inter[0][0]+inter[0][1])//2],depth_image,"cylinder",depth_stamp,colorToPush)
            (self.nM, self.m_arr) = module.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush,self.positions["cylinder"],self.nM, self.m_arr, self.markers_pub)
            #self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush)

        return grayBGR_toDrawOn

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
        #markedImage, depth_im_shifted = self.find_elipses_first(rgb_image, depth_image,rgb_image_message.header.stamp, depth_image_message.header.stamp, grayImage)
        #print(markedImage)
        # TODO: make it so it marks face and returns the image to display
        markedImage = self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)
        #self.find_faces(rgb_image,depth_im_shifted,depth_image_message.header.stamp)
        #markedImage = self.find_QR(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
        #markedImage = self.find_digits(rgb_image,depth_im_shifted,depth_image_message.header.stamp, markedImage)
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


def main():
    cylinder_finder = cylinders()

    rate = rospy.Rate(1.25)
    skipCounter = 3
    while not rospy.is_shutdown():
        # print("hello!")
        if skipCounter <= 0:
            cylinder_finder.find_objects()
        else:
            skipCounter -= 1
        #print("hello")
        #print("\n-----------\n")
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
