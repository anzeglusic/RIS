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

modelsDir = "/home/sebastjan/Documents/faks/3letnk/ris/ROS_task/src/exercise4/scripts"

class color_localizer:

    def __init__(self):

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
            "cylinder": []
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
    
    def chose_color(self,colorDict):
        b = colorDict["b"]
        g = colorDict["g"]
        r = colorDict["r"]
        c = colorDict["c"]
        w = colorDict["w"]
        y = colorDict["y"]

        # black is selected if no other is selected
        c = 0
        # white is considered to be blue
        b += w
        w = 0

        tag =   ["b","g","r","y","w","c"]
        value = [ b , g , r , y , w , c ]

        if np.sum(value) < 3:
            return 'c'
        bestIndx = np.argmax(value)
        return tag[bestIndx]
    
    def checkPosition(self):
        '''
        positions = {
            "ring": [
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False }
                
            ],
            "cylinder": [
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False  },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False  },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False  }
                
            ]
        }
        '''
        for objectType in ["ring","cylinder"]:
            for area in self.positions[objectType]:
                if area["approached"] == True:
                    continue
                distArr = area["averagePostion"]-np.array([self.basePosition["x"],self.basePosition["y"],self.basePosition["z"]])
                dist = np.sqrt(distArr[0]**2 + distArr[1]**2 + distArr[2]**2)

                # you are too far from the ring
                if dist>2 and objectType=="ring":
                    continue
                # you are too far from the cylinder
                if dist>3 and objectType=="cylinder":
                    continue
                
                print(f"Approaching to object in position: {area['averagePostion']}")
                area["approached"] = True
                # ring --> z=1
                # cylinder --> z=0
                if objectType == "ring":
                    # ring
                    self.points_pub.publish(Point(area["averagePostion"][0],area["averagePostion"][1],1))
                elif objectType == "cylinder":
                    # cylinder
                    self.points_pub.publish(Point(area["averagePostion"][0],area["averagePostion"][1],0))    
                else:
                    print("SOMETHING WENT FUCKING WRONG !!!!!!!")

            
    def addPosition(self, newPosition, objectType, color_char):
        '''
        positions = {
            "ring": [
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None }
                
            ],
            "cylinder": [
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None  },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None  },
                { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None  }
                
            ]
        }
        '''
        unique_position = True
        for area in self.positions[objectType]:
            area_avg = area["averagePostion"]
            dist_vector = area_avg - newPosition
            dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
            if dist > 1:
                continue
            
            unique_position = False
            # collection
            area["detectedPositions"].append(newPosition.copy())
            # color
            area["color"][color_char] += 1
            # average
            area["averagePostion"] = np.sum(area["detectedPositions"],axis=0)/len(area["detectedPositions"])
            # average marker

            if len(area["detectedPositions"])>3:
                if area["avgMarkerId"] == None:
                    #! FIRST
                    # Create a Pose object with the same position
                    pose = Pose()
                    pose.position.x = area["averagePostion"][0]
                    pose.position.y = area["averagePostion"][1]
                    pose.position.z = area["averagePostion"][2]

                    # Create a marker used for visualization
                    self.nM += 1
                    marker = Marker()
                    marker.header.stamp = rospy.Time(0)
                    marker.header.frame_id = "map"
                    marker.pose = pose
                    marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
                    marker.action = Marker.ADD
                    marker.frame_locked = False
                    marker.lifetime = rospy.Duration.from_sec(0)
                    marker.id = self.nM
                    marker.scale = Vector3(0.1, 0.1, 0.1)
                    
                    best_color = self.chose_color(area["color"])
                    
                    area["avgMarkerId"] = self.nM
                        
                    marker.color = self.rgba_from_char(best_color)
                    self.m_arr.markers.append(marker)

                    self.markers_pub.publish(self.m_arr)
                else:
                    #! UPDATE
                    # Create a Pose object with the same position
                    pose = Pose()
                    pose.position.x = area["averagePostion"][0]
                    pose.position.y = area["averagePostion"][1]
                    pose.position.z = area["averagePostion"][2]

                    # Create a marker used for visualization
                    self.nM += 1
                    marker = Marker()
                    marker.header.stamp = rospy.Time(0)
                    marker.header.frame_id = "map"
                    marker.pose = pose
                    marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
                    marker.action = Marker.ADD
                    marker.frame_locked = False
                    marker.lifetime = rospy.Duration.from_sec(0)
                    marker.id = area["avgMarkerId"]
                    marker.scale = Vector3(0.1, 0.1, 0.1)
                    best_color = self.chose_color(area["color"])
                        
                    marker.color = self.rgba_from_char(best_color)
                    self.m_arr.markers.append(marker)

                    self.markers_pub.publish(self.m_arr)





        
        if unique_position == True:
            colorDict = {"r":0,"g":0,"b":0,"y":0,"w":0,"c":0}
            colorDict[color_char] += 1
            self.positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "color": colorDict,
                                                "approached": False,
                                                "avgMarkerId": None
                                                })
        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = newPosition[0]
        pose.position.y = newPosition[1]
        pose.position.z = newPosition[2]

        # Create a marker used for visualization
        self.nM += 1
        marker = Marker()
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "map"
        marker.pose = pose
        marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(0)
        marker.id = self.nM
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0.5,0.5,0.5,0.25)
        self.m_arr.markers.append(marker)

        self.markers_pub.publish(self.m_arr)
    
    def calc_rgb(self,point):
        # print("\n!!!!!!!!!!!!!!!!!!!")
        # print(point.shape)
        # pprint(point)
        # print("\n!!!!!!!!!!!!!!!!!!!")

        p0 = point[0,:]
        p1 = point[1,:]
        p2 = point[2,:]

        hsvColor0 = cv2.cvtColor(np.array([[p0]]),cv2.COLOR_BGR2HSV)[0][0]
        hsvColor0[0] = int((float(hsvColor0[0])/180)*255)

        hsvColor1 = cv2.cvtColor(np.array([[p1]]),cv2.COLOR_BGR2HSV)[0][0]
        hsvColor1[0] = int((float(hsvColor1[0])/180)*255)

        hsvColor2 = cv2.cvtColor(np.array([[p2]]),cv2.COLOR_BGR2HSV)[0][0]
        hsvColor2[0] = int((float(hsvColor2[0])/180)*255)
        
        arr = np.array([[p0[0],p0[1],p0[2],p1[0],p1[1],p1[2],p2[0],p2[1],p2[2]]])

        arr2 = np.array([[   hsvColor0[2],hsvColor0[1],hsvColor0[0],
                            hsvColor1[2],hsvColor1[1],hsvColor1[0],
                            hsvColor2[2],hsvColor2[1],hsvColor2[0]
                        ]])

        print(f"BGR: {arr}")
        print(f"VSH: {arr2}\n")
        
        y = self.knn_RGB.predict(arr)
        print(f"knn_BGR:")
        print(f"\tprediction: {y[0]}")
        y = self.random_forest_RGB.predict(arr)
        print(f"random_forest_RGB:")
        print(f"\tprediction: {y[0]}")

        y = self.knn_HSV.predict(arr2)
        print(f"knn_HSV:")
        print(f"\tprediction: {y[0]}")
        y = self.random_forest_HSV.predict(arr2)
        print(f"random_forest_HSV:")
        print(f"\tprediction: {y[0]}")

        return y[0]

        # print(f"p0: {p0}")
        # print(f"p1: {p1}")
        # print(f"p2: {p2}")

        distances = { "red":{}
                    , "blue":{}
                    , "green": {}
                    , "black": {}
                    , "white": {}
                    , "yellow": {}
                    }
        # red
        color = np.array([0,0,255])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["red"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["red"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["red"]["p2"] = dist

        #  blue
        color = np.array([255,0,0])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p2"] = dist
        
        # cyan --> blue
        color = np.array([255,255,0])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p0"] = min(dist,distances["blue"]["p0"])
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p1"] = min(dist,distances["blue"]["p1"])
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["blue"]["p2"] = min(dist,distances["blue"]["p2"])

        # green
        color = np.array([0,255,0])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["green"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["green"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["green"]["p2"] = dist

        # black
        color = np.array([0,0,0])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["black"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["black"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["black"]["p2"] = dist

        # white
        color = np.array([255,255,255])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["white"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["white"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["white"]["p2"] = dist

        # yellow
        color = np.array([0,255,255])
        calculatedPoint = p0-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["yellow"]["p0"] = dist
        
        calculatedPoint = p1-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["yellow"]["p1"] = dist
        
        calculatedPoint = p2-color
        dist = np.sqrt(calculatedPoint[0]**2 + calculatedPoint[1]**2 + calculatedPoint[2]**2)
        distances["yellow"]["p2"] = dist
        

        bestScore = np.array([1,1,1])*np.sqrt(255**2 + 255**2 + 255**2)
        color = ["nothing","nothing","nothing"]
        for colorKey in distances:
            score0 = distances[colorKey]["p0"]
            if score0 < bestScore[0]:
                bestScore[0] = score0
                color[0] = colorKey
            
            score1 = distances[colorKey]["p1"]
            if score1 < bestScore[1]:
                bestScore[1] = score1
                color[1] = colorKey
            
            score2 = distances[colorKey]["p2"]
            if score2 < bestScore[2]:
                bestScore[2] = score2
                color[2] = colorKey
        
        counter = np.array([0,0,0])
        counter[0] = np.sum([color[0]==color[0],color[1]==color[0],color[2]==color[0]])
        counter[1] = np.sum([color[0]==color[1],color[1]==color[1],color[2]==color[1]])
        counter[2] = np.sum([color[0]==color[2],color[1]==color[2],color[2]==color[2]])
        # print(counter)
        if np.sum(counter)==3:
            col = color[np.argmin(bestScore)]
        else:
            col = color[np.argmax(counter)]
        
        if col == "black":
            return "c"
        if col == "white":
            return "w"
        if col == "blue":
            return "b"
        if col == "yellow":
            return "y"
        if col == "red":
            return "r"
        if col == "green":
            return "g"
    
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

    def rgba_from_char(self, color_char):
        if color_char == "c":
            return ColorRGBA(0, 0, 0, 1)
        if color_char == "w":
            return ColorRGBA(1, 1, 1, 1)
        if color_char == "r":
            return ColorRGBA(1, 0, 0, 1)
        if color_char == "g":
            return ColorRGBA(0, 1, 0, 1)
        if color_char == "b":
            return ColorRGBA(0, 0, 1, 1)
        if color_char == "y":
            return ColorRGBA(1, 1, 0, 1)

    def get_pose(self,xin,yin,dist, depth_im,objectType,depth_stamp,color_char):
        # Calculate the position of the detected ellipse
        print(f"dist: {dist}")
        k_f = 525 # kinect focal length in pixels

        xin = -(xin - depth_im.shape[1]/2)

        angle_to_target = np.arctan2(xin,k_f)

        # Get the angles in the base_link relative coordinate system
        x,y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)


        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        # point_s.header.stamp = rospy.Time(0)
        point_s.header.stamp = depth_stamp
        # if objectType == "cylinder":
            # point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")



        allDist = []
        if objectType == "ring":
            for elipse in self.found_rings:
                pointsDist = np.sqrt((elipse["x"]-point_world.point.x)**2+(elipse["y"]-point_world.point.y)**2)
                # if pointsDist < 0.5 or dist>2:
                #     return
                # if pointsDist < 0.5:
                #     return
                allDist.append(pointsDist)
            # allDist = np.array(allDist)
            self.found_rings.append({"x":point_world.point.x, "y":point_world.point.y})
        elif objectType == "cylinder":
            for cylinder in self.found_cylinders:
                pointsDist = np.sqrt((cylinder["x"]-point_world.point.x)**2+(cylinder["y"]-point_world.point.y)**2)
                # if pointsDist < 0.5 or dist>2:
                #     return
                #! if dist>2:
                    #! return
                allDist.append(pointsDist)
            # allDist = np.array(allDist)
            self.found_cylinders.append({"x":point_world.point.x, "y":point_world.point.y})


        # pprint(sorted(allDist))


        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z

        # # Create a marker used for visualization
        # self.nM += 1
        # marker = Marker()
        # marker.header.stamp = point_world.header.stamp
        # marker.header.frame_id = point_world.header.frame_id
        # marker.pose = pose
        # marker.type = Marker.CUBE
        # marker.action = Marker.ADD
        # marker.frame_locked = False
        # marker.lifetime = rospy.Duration.from_sec(0)
        # marker.id = self.nM
        # marker.scale = Vector3(0.1, 0.1, 0.1)
        # # if objectType == "ring":
        # #     marker.color = ColorRGBA(1, 0, 0, 1)
        # # else:
        # #     marker.color = ColorRGBA(0, 1, 0, 1)
        # marker.color = self.rgba_from_char(color_char)
        # self.m_arr.markers.append(marker)

        # self.markers_pub.publish(self.m_arr)
        
        return pose

    #dela as intended!
    def calc_pnts(self, el):
        #width
        h1 = np.array([0,el[1][1]/2])
        h2 = np.array([0,-el[1][1]/2])
        w1 = np.array([el[1][0]/2,0])
        w2 = np.array([-el[1][0]/2,0])


        c = np.cos(np.radians(el[2]))
        s = np.sin(np.radians(el[2]))
        rot = np.array([[c,-s],[s,c]])


        h1 = el[0] + rot.dot(h1)
        h2 = el[0] + rot.dot(h2)
        w1 = el[0] + rot.dot(w1)
        w2 = el[0] + rot.dot(w2)
        return h1,h2,w1,w2
    
    def calc_thresholds(self, frame):
        # Do histogram equlization
        img = cv2.equalizeHist(frame)
            
        threshDict = {}
        # Binarize the image - original
        ret, thresh = cv2.threshold(img, 50, 255, 0)
        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        threshDict["normal"] = thresh

        # binary + otsu
        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshDict["binaryOtsu"] = thresh

        # # gauss + binary + otsu
        img2 = cv2.GaussianBlur(img.copy(),(5,5),2)
        ret, thresh = cv2.threshold(img2, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # threshDict["gaussBinaryOtsu"] = thresh
        
        # # adaptive mean threshold
        thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,3,2)
        # threshDict["adaptiveMean"] = thresh
        
        # # adaptive gaussian threshold
        thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,2)
        # threshDict["adaptiveGauss"] = thresh

        return threshDict

    def bgr2gray(self, bgr_image):
        return (np.sum(bgr_image.copy().astype(np.float),axis=2)/3).astype(np.uint8)

    def gray2bgr(self, bgr_image):
        newImage = np.zeros((bgr_image.shape[0],bgr_image.shape[1],3))
        
        newImage[:,:,0] = bgr_image.copy()
        newImage[:,:,1] = bgr_image.copy()
        newImage[:,:,2] = bgr_image.copy()

        return newImage.astype(np.uint8)

    def find_elipses_first(self,image,depth_image,im_stamp,depth_stamp,grayBGR_toDrawOn):
        print(f"rgb_stamp_sec:    {im_stamp.to_sec()}")
        print(f"depth_stamp_sec:  {depth_stamp.to_sec()}")
        diff = (depth_stamp.to_sec()-im_stamp.to_sec())
        print(f"diff:             {diff}")
        print()
        # print(self.detected_pos_fin)
        
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
                        , self.bgr2gray(frame)
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
            
            threshDict = self.calc_thresholds(currentImage)

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


        depthThresh = self.calc_thresholds(depth_image)
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
        print(f"\t\t\tbestScore: {bestScore}")
        shiftX = bestShift
        M = np.float32([[1,0,shiftX],[0,1,0]])
        toReturn[:,:,2] = cv2.warpAffine(depth_t,M,(toReturn.shape[1],toReturn.shape[0]),borderValue=np.nan).astype(np.uint8)
        depth_im_shifted = cv2.warpAffine(depth_im,M,(depth_im.shape[1],depth_im.shape[0]),borderValue=np.nan)

        # print(f"all: {np.unique(toReturn)}")
        # return toReturn[:,:,2],depth_im_shifted
        toReturn = ((self.bgr2gray(toReturn)>0)*255).astype(np.uint8)
        # return toReturn.astype(np.uint8),depth_im_shifted
        # print(toReturn.shape)
        # print(f"YES: {np.sum(toReturn>0)}")
        # print(f"NO:  {np.sum(toReturn<=0)}")
        # print(f"all: {np.unique(toReturn)}")
        # print(f"all: {np.unique(self.gray2bgr(toReturn))}")
        
        
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

            h1,h2,w1,w2 = self.calc_pnts(e1)
            h11,h21,w11,w21 = self.calc_pnts(e2)

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
            cv2.ellipse(grayBGR_toDrawOn, e1, (0, 0, 255), 2)
            cv2.ellipse(grayBGR_toDrawOn, e2, (0, 0, 255), 2)

            
            cv2.circle(grayBGR_toDrawOn,tuple( c[2]),1,(0,255,0),2)
            
            cv2.circle(grayBGR_toDrawOn,tuple( h11),1,(0,255,0),2)
            cv2.circle(grayBGR_toDrawOn,tuple( h21),1,(0,255,0),2)
            cv2.circle(grayBGR_toDrawOn,tuple( w11),1,(0,255,0),2)
            cv2.circle(grayBGR_toDrawOn,tuple( w21),1,(0,255,0),2)


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
                color = self.calc_rgb(pnts)
                # print("dela 2")
                ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im_shifted,"ring",depth_stamp,color)
                # print("dela 3")
                self.addPosition(np.array((ring_point.position.x,ring_point.position.y,ring_point.position.z)),"ring",color)
                # print("dela 4")
                # ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im,"ring")
                
            except Exception as e:
                print(f"Ring error: {e}")
        return grayBGR_toDrawOn, depth_im_shifted

    def calc_rgb_distance_mask(self, image, rgb_value, minDistance):
        rangeImage = image - np.array(rgb_value)
        temp = np.reshape(rangeImage, (image.shape[0]*image.shape[1],3))
        temp = np.linalg.norm(temp,axis=1)
        temp = np.reshape(temp, (image.shape[0],image.shape[1]))

        temp = (temp / np.sqrt(255**2 + 255**2 + 255**2))*255
        temp = temp.astype(np.uint8)

        distance = temp.copy()

        toMask = temp < minDistance
        temp = (toMask * 255).astype(np.uint8)

        return temp, distance

    def calc_mask(self, mask, kernel, kernel1, operations):
        oper1 = operations[0]
        oper2 = operations[1]
        oper3 = operations[2]
        oper4 = operations[3]
        oper5 = operations[4]

        if oper1:
            mask = cv2.dilate(mask, kernel)
        if oper2:
            mask = cv2.erode(mask, kernel)
        if oper3:
            mask = cv2.erode(mask, kernel1)
        if oper4:
            mask = cv2.dilate(mask, kernel1)
        if oper5:
            mask = cv2.dilate(mask, kernel)
        return mask

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
        if shift < 1:
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

    def find_cylinderDEPTH(self,image, depth_image, grayBGR_toDrawOn,depth_stamp):
        # print(depth_stamp)
        centerRowIndex = depth_image.shape[0]//2
        presekiIndex,blizDalec = self.getLows(depth_image[centerRowIndex,:])
        cnt = 0
        for cnt in range(1,len(presekiIndex)-1):
            #se pojavi kot najblizja točka katere desni in levi pixl sta  za njo
            if blizDalec[cnt]:

                interval = self.getRange(depth_image[centerRowIndex,:],presekiIndex[cnt],presekiIndex[cnt-1],presekiIndex[cnt+1])
                if interval != None:
                    #NASLI SMO CILINDER
                    # print(depth_image)
                    try:
                        points = np.array([ image[centerRowIndex,presekiIndex[cnt]],
                                            image[centerRowIndex,presekiIndex[cnt]-1],
                                            image[centerRowIndex,presekiIndex[cnt]+1]])
                        # print(f"Širina:{interval[0]-interval[1]}\n\tna razdalji:{depth_image[centerRowIndex,presekiIndex[cnt]]}")
                        if np.abs(interval[0]-interval[1]) <= 20 or depth_image[centerRowIndex,presekiIndex[cnt]]>2.5:
                            return
                        colorToPush = self.calc_rgb(points)
                        pose = self.get_pose(presekiIndex[cnt],centerRowIndex,depth_image[centerRowIndex,presekiIndex[cnt]],depth_image,"cylinder",depth_stamp,colorToPush)
                        self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",colorToPush)
                    except Exception as e:
                        print(f"find_cylinderDEPTH error: {e}")
                        return


    def find_cylinder(self,image, depth_image, grayBGR_toDrawOn,depth_stamp):
        # red
        red_mask, red_distance = self.calc_rgb_distance_mask(image, [0,0,255], 100)

        # green
        green_mask, green_distance = self.calc_rgb_distance_mask(image, [0,255,0], 100)

        # blue
        blue_mask, blue_distance = self.calc_rgb_distance_mask(image, [255,0,0], 100)

        # yellow
        yellow_mask, yellow_distance = self.calc_rgb_distance_mask(image, [0,255,255], 100)

        # cyan
        cyan_mask, cyan_distance = self.calc_rgb_distance_mask(image, [255,255,0], 100)

        # black
        black_mask, black_distance = self.calc_rgb_distance_mask(image, [0,0,0], 60)

        # white
        white_mask, white_distance = self.calc_rgb_distance_mask(image, [255,255,255], 5)

        kernel = np.ones((30,30), "uint8")
        kernel1 = np.ones((7,7), "uint8")

        oper1 = True
        oper2 = True
        oper3 = False
        oper4 = False
        oper5 = False

        operations = [oper1, oper2, oper3, oper4, oper5]

        # red
        red_mask = self.calc_mask(mask=red_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_red = cv2.bitwise_and(image, image, mask = red_mask)

        # green
        green_mask = self.calc_mask(mask=green_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_green = cv2.bitwise_and(image, image, mask = green_mask)

        # blue
        blue_mask = self.calc_mask(mask=blue_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_blue = cv2.bitwise_and(image, image, mask = blue_mask)

        # yellow
        yellow_mask = self.calc_mask(mask=yellow_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_yellow = cv2.bitwise_and(image, image, mask = yellow_mask)

        # cyan
        cyan_mask = self.calc_mask(mask=cyan_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_cyan = cv2.bitwise_and(image, image, mask = cyan_mask)

        # black
        black_mask = self.calc_mask(mask=black_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_black = cv2.bitwise_and(image, image, mask = black_mask)

        # white
        white_mask = self.calc_mask(mask=white_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_white = cv2.bitwise_and(image, image, mask = white_mask)

        gray = self.bgr2gray(image)
        imageGray = self.gray2bgr(gray)

        masksList = [red_mask, green_mask, blue_mask, yellow_mask, cyan_mask, black_mask, white_mask]
        colorsForBorder = [ (0,0,255),
                            (0,255,0),
                            (255,0,0), 
                            (0,255,255), 
                            (255,255,0), 
                            (0,0,0),
                            (255,255,255)
                            ]
        colorDistanceList = [red_distance, green_distance, blue_distance, yellow_distance, cyan_distance, black_distance, white_distance]
        i = -1
        colorsDict = {  0:"red",
                        1:"green",
                        2:"blue",
                        3:"yellow",
                        4:"cyan",
                        5:"black",
                        6:"white"}
        centerRowIndex = depth_image.shape[0]//2
        # print(depth_image[centerRowIndex])
        for mask, borderColor, colorDistance in zip(masksList,colorsForBorder,colorDistanceList):
            i += 1
            contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for pic, cont in enumerate(contour):
                area = cv2.contourArea(cont)
                if(area > 300):
                    x, y, w, h = cv2.boundingRect(cont)
                    # check if rectangle doesnt cross center line
                    if y>centerRowIndex or (y+h)<=centerRowIndex:
                        continue
                    depth_cut = depth_image[y:(y+h),x:(x+w)]
                    image_cut = image[y:(y+h),x:(x+w)]
                    dist_cut = colorDistance[y:(y+h),x:(x+w)]
                    mask_cut = mask[y:(y+h),x:(x+w)]
                    # print(colorsDict[i])
                    # print(mask_cut.shape)
                    mask_row_sum = np.sum(mask_cut>0,1)
                    # print("\t",mask_row_sum.shape)
                    # bestRowIndx = np.argmax(mask_row_sum)
                    bestRowIndx = centerRowIndex-y
                    mask_row = mask_cut[bestRowIndx,:]>0
                    depth_row = depth_cut[bestRowIndx,:]

                    tempColumnRow = depth_row*mask_row
                    tempColumnRow[(tempColumnRow==0)] = np.Inf
                    tempColumnRow[np.isnan(tempColumnRow)] = np.Inf
                    bestColumnIndx = np.argmin(tempColumnRow)

                    numOfCorrectColors = np.sum(mask_cut==255)
                    numOfAllColors = h*w
                    # print(f"{colorsDict[i]}\n\t{numOfCorrectColors}/{numOfAllColors} [{numOfCorrectColors/numOfAllColors}]")

                    if numOfCorrectColors/numOfAllColors < 0.5:
                        continue
                    
                    counter = 0
                    while True:
                        if (bestColumnIndx-counter) < 0:
                            counter -= 1
                            break
                        if (bestColumnIndx+counter) == len(mask_row):
                            counter -= 1
                            break 
                        if mask_row[bestColumnIndx-counter]==False or mask_row[bestColumnIndx+counter]==False or np.isinf(tempColumnRow[bestColumnIndx-counter])==True or np.isinf(tempColumnRow[bestColumnIndx+counter])==True:
                            counter -= 1
                            break
                        counter += 1

                    shift = np.floor((counter*2)/5).astype(int)
                    if shift < 1:
                        continue
                    
                    LL = depth_row[bestColumnIndx-shift*2]
                    L = depth_row[bestColumnIndx-shift]
                    C = depth_row[bestColumnIndx]
                    R = depth_row[bestColumnIndx+shift]
                    RR = depth_row[bestColumnIndx+shift*2]

                    LL_C = LL-C
                    L_C = L-C
                    C_R = R-C
                    C_RR = RR-C

                    if LL_C<=0 or L_C<=0 or C_R<=0 or C_RR<=0:
                        continue

                    if LL_C<L_C*3 or C_RR<C_R*3:
                        continue
                    
                    center_row_indx = np.floor(h/2).astype(int)
                    up_shif = np.floor(h/5).astype(int)
                    if up_shif<1:
                        continue
                    
                    rowIndx1 = bestRowIndx-up_shif
                    rowIndx2 = bestRowIndx
                    rowIndx3 = bestRowIndx+up_shif
                    if rowIndx1 < 0:
                        a = depth_cut[bestRowIndx+up_shif*2,bestColumnIndx]
                    else:
                        a = depth_cut[bestRowIndx-up_shif,bestColumnIndx]
                    
                    b = depth_cut[bestRowIndx,bestColumnIndx]
    
                    if rowIndx3 >= h :
                        c = depth_cut[bestRowIndx-up_shif*2,bestColumnIndx]
                    else:
                        c = depth_cut[bestRowIndx+up_shif,bestColumnIndx]

                    distThreshold = 0.005
                    if np.abs(a-b)>distThreshold or np.abs(a-c)>distThreshold or np.abs(b-c)>distThreshold:
                        continue
                    
                    print(f"\tcylinder detected --> {colorsDict[i]}")
                    print(f"\tselected point ditance: {depth_image[y+bestRowIndx,x+bestColumnIndx]}")
                    # pprint(depth_cut[bestRowIndx,:])
                    points = np.array([image_cut[bestRowIndx,bestColumnIndx],image_cut[bestRowIndx,bestColumnIndx],image_cut[bestRowIndx,bestColumnIndx],])
                    try:
                        # self.get_pose((x+w/2),(y+h/2),C,depth_image,"cylinder",depth_stamp)
                        pose = self.get_pose((x+bestColumnIndx),(y+bestRowIndx),C,depth_image,"cylinder",depth_stamp,self.calc_rgb(points))
                        self.addPosition(np.array([pose.position.x,pose.position.y,pose.position.z]),"cylinder",self.calc_rgb(points))
                    except Exception as e:
                        print(f"Cylinder error: {e}")
                    grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (x, y),(x+w,y+h), borderColor ,2)
                    grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (x, y+bestRowIndx),(x+w,y+bestRowIndx), borderColor ,2)
                    grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (x+bestColumnIndx-5, y+bestRowIndx-5),(x+bestColumnIndx+5,y+bestRowIndx+5), (255-borderColor[0],255-borderColor[1],255-borderColor[2]) ,2)
        
        """
        contour, hierarchy = cv2.findContours(red_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cont in enumerate(contour):
            area = cv2.contourArea(cont)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(cont)
                image = cv2.rectangle(image, (x, y),(x+w,y+h), (0,255,0),2)
        """
        # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn,(0,235),(640,245),(0,255,0),2)
        # print(grayBGR_toDrawOn.shape)
        # grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (0, centerRowIndex-10),(640,centerRowIndex+10), (0,0,0) ,2)
        return grayBGR_toDrawOn
        # return cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)

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

        grayImage = self.bgr2gray(rgb_image)
        grayImage = self.gray2bgr(grayImage)
        markedImage, depth_im_shifted = self.find_elipses_first(rgb_image, depth_image,rgb_image_message.header.stamp, depth_image_message.header.stamp, grayImage)
        self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)

        self.checkPosition()

        # print(type(markedImage))
        # print(markedImage.shape)
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

def main():

        color_finder = color_localizer()

        rate = rospy.Rate(1.25)
        while not rospy.is_shutdown():
            # print("hello!")
            color_finder.find_objects()
            #print("hello")
            #print("\n-----------\n")
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
