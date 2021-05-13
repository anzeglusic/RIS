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
            "face": []
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

        self.showEveryDetection = True


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
    
    def chose_color(self,colorDict):
        b = colorDict["b"]
        g = colorDict["g"]
        r = colorDict["r"]
        c = colorDict["c"]
        w = colorDict["w"]
        y = colorDict["y"]

        numOfDetections = np.sum([ b , g , r , y , w , c ])

        # black is selected if no other is selected
        c = 0
        # white is considered to be blue
        b += w
        w = 0

        tag =   ["b","g","r","y","w","c"]
        value = [ b , g , r , y , w , c ]

        if np.sum(value)==0 and numOfDetections>10:
            print("NOT ENOUGHF COLORS DETECTED")
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
                best_color = self.chose_color(area["color"])
                d = { "w":"white"
                    , "c" : "black"
                    , "y" : "yellow"
                    , "r" : "red"
                    , "g" : "green"
                    , "b" : "blue"
                }
                color_word = d[best_color]
                pprint(area["color"])
                #! to play a sound !!!!!!!!!!!!!!!!!!!!
                #! subprocess.run(["rosrun" , "sound_play", "say.py", f"{color_word} {objectType}"])

    def addPosition(self, newPosition, objectType, color_char, face_normal=None):
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
                
            ],
            "face": [
                { 
                    "averagePostion": np.array([x,y,z]),
                    "averageNormal": np.array([x,y,z]),
                    "detectedPositions": [ pos0, pos1, pos2, pos3],
                    "detectedNormals": [ pos0, pos1, pos2, pos3],
                    "approached": False,
                    "avgMarkerId": None
                },
                ...
            ]
        }
        '''
        # make normal of magnitude 1
        if objectType == "face":
            if face_normal is None:
                return
            face_normal = face_normal/np.sqrt(face_normal[0]**2 + face_normal[1]**2 + face_normal[2]**2)
        
        unique_position = True
        for area in self.positions[objectType]:
            area_avg = area["averagePostion"]
            dist_vector = area_avg - newPosition
            dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
            if objectType=="face":
                # TODO: check if the oriantation (normal) is correct --> it oculd be that the face is on the other side of the wall
                c_sim = np.abs(np.dot(face_normal,area_avg)/(np.linalg.norm(face_normal)*np.linalg.norm(area_avg)))
                print("similarity:",c_sim)
                if c_sim<=0.5:
                    continue
                      
                print("Dist --> ",dist)
            if dist > 0.5:
                continue
            
            unique_position = False
            
            # depending on the type of object
            if objectType=="cylinder" or objectType=="ring":
                # color
                area["color"][color_char] += 1
            elif objectType=="face":
                # collection of normals
                area["detectedNormals"].append(face_normal)
                # average normal
                print(area["detectedNormals"])
                area["averageNormal"] = np.sum(area["detectedNormals"],axis=0)/len(area["detectedNormals"])
                area["averageNormal"] = area["averageNormal"]/np.sqrt(area["averageNormal"][0]**2 + area["averageNormal"][1]**2 + area["averageNormal"][2]**2)
            
            # collection of positions
            area["detectedPositions"].append(newPosition.copy())
            
            # average
            area["averagePostion"] = np.sum(area["detectedPositions"],axis=0)/len(area["detectedPositions"])
            
            # average marker
            if len(area["detectedPositions"])>3 and (objectType=="ring" or objectType=="cylinder"):
                # Create a Pose object with the same position
                pose = Pose()
                pose.position.x = area["averagePostion"][0]
                pose.position.y = area["averagePostion"][1]
                pose.position.z = area["averagePostion"][2]+0.1

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
                marker.scale = Vector3(0.1, 0.1, 0.1)
                
                if area["avgMarkerId"] == None:
                    marker.id = self.nM
                    area["avgMarkerId"] = self.nM
                else:
                    marker.id = area["avgMarkerId"]
                
                best_color = self.chose_color(area["color"])
                marker.color = self.rgba_from_char(best_color)
                    
                self.m_arr.markers.append(marker)
                self.markers_pub.publish(self.m_arr)
            elif len(area["detectedPositions"])>3 and objectType=="face":
                self.nM += 1
                marker = Marker()
                marker.header.stamp = rospy.Time(0)
                marker.header.frame_id = "map"
                marker.action = Marker.ADD
                marker.frame_locked = False
                marker.lifetime = rospy.Duration.from_sec(0)

                if area["avgMarkerId"] == None:
                    marker.id = self.nM
                    area["avgMarkerId"] = self.nM
                else:
                    marker.id = area["avgMarkerId"]
                
                marker.ns = 'points_arrow'
                marker.type = Marker.ARROW
                marker.pose.orientation.y = 0
                marker.pose.orientation.w = 1
                marker.scale = Vector3(1,1,1)
                marker.color = ColorRGBA(1,0,0,0.5)
                
                orig_arr = area["averagePostion"].copy()
                orig_arr[2] += 0.1
                dest_arr = orig_arr - area["averageNormal"]*self.faceNormalLength
                head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
                tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])
                
                marker.points = [tail,head]
                
                self.m_arr.markers.append(marker)
                self.markers_pub.publish(self.m_arr)





        
        if unique_position == True:
            if objectType=="ring" or objectType=="cylinder":
                colorDict = {"r":0,"g":0,"b":0,"y":0,"w":0,"c":0}
                colorDict[color_char] += 1
                self.positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                    "detectedPositions":[newPosition.copy()],
                                                    "color": colorDict,
                                                    "approached": False,
                                                    "avgMarkerId": None
                                                    })
            elif objectType=="face":
                print("\n\nAdding new face\n\n")
                self.positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                    "averageNormal": face_normal.copy(),
                                                    "detectedPositions":[newPosition.copy()],
                                                    "detectedNormals":[face_normal.copy()],
                                                    "approached": False,
                                                    "avgMarkerId": None
                                                    })

        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = newPosition[0]
        pose.position.y = newPosition[1]
        pose.position.z = newPosition[2]
        if self.showEveryDetection==True:
            # Create a marker used for visualization
            self.nM += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(0)
            marker.id = self.nM
            
            if objectType=="ring" or objectType=="cylinder":
                marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
                marker.pose = pose
                marker.scale = Vector3(0.1, 0.1, 0.1)
                marker.color = ColorRGBA(0.5,0.5,0.5,0.25)
            elif objectType=="face":
                marker.ns = 'points_arrow'
                marker.type = Marker.ARROW
                marker.pose.orientation.y = 0
                marker.pose.orientation.w = 1
                marker.scale = Vector3(1,1,1)
                marker.color = ColorRGBA(0.5,0.5,0.5,0.1)
                
                orig_arr = newPosition
                dest_arr = orig_arr - face_normal*self.faceNormalLength
                head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
                tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])
                
                marker.points = [tail,head]
            
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
        c1 = y[0]
        print(f"knn_BGR:")
        print(f"\tprediction: {y[0]}")
        y = self.random_forest_RGB.predict(arr)
        c2 = y[0]
        print(f"random_forest_RGB:")
        print(f"\tprediction: {y[0]}")

        y = self.knn_HSV.predict(arr2)
        c3 = y[0]
        print(f"knn_HSV:")
        print(f"\tprediction: {y[0]}")
        y = self.random_forest_HSV.predict(arr2)
        c4 = y[0]
        print(f"random_forest_HSV:")
        print(f"\tprediction: {y[0]}")

        d = {
            "w": 0,
            "b": 0,
            "c": 0,
            "r": 0,
            "g": 0,
            "y": 0
        }
        d[c1] += 1
        d[c2] += 1
        d[c3] += 1
        d[c4] += 1

        ar =    ["b","g","r","y","w","c"]
        ar_num =[d["b"],d["g"],d["r"],d["y"],d["w"],d["c"]]
        if np.sum(np.array(ar_num)==1)==4:
            return "c"
        
        temp = ar[np.argmax(np.array(ar_num))]
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! --> {temp}")
        return temp
        

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


        #! USELESS CODE
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
        # TODO: make it so it marks face and returns the image to display
        self.find_cylinderDEPTH(rgb_image, depth_im_shifted, markedImage,depth_image_message.header.stamp)

        self.find_faces(rgb_image,depth_im_shifted,depth_image_message.header.stamp)

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


#! face detection start
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
                if self.does_it_cross(checked_good_detection["x"], checked_good_detection["y"], (x1,x2), (y1,y2)):
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
            norm = self.get_normal(depth_image, (x1,y1,x2,y2) ,depth_image_stamp,face_distance, face_region)
            normals_found.append(norm)

            print('Norm of face', norm)

            print('Distance to face', face_distance)
            #trans = tf2_ros.Buffer().lookup_transform('mapa', 'base_link', rospy.Time(0))
            #print('my position ',trans)
            # Get the time that the depth image was recieved
            depth_time = depth_image_stamp

            # Find the location of the detected face
            pose = self.get_pose_face((x1,x2,y1,y2), face_distance, depth_time)
            if pose != None:
                newPosition = np.array([pose.position.x,pose.position.y, pose.position.z])
                self.addPosition(newPosition, "face", color_char=None, face_normal=norm)          

    def does_it_cross(self, a_cords_x, a_cords_y, b_cords_x, b_cords_y):
        # it answers the qustion if rectangles cross in any way
        # returns True --> coardinates cross
        # returns False --> coardinates DONT cross
        a_x1, a_x2 = a_cords_x
        a_y1, a_y2 = a_cords_y
        
        b_x1, b_x2 = b_cords_x
        b_y1, b_y2 = b_cords_y

        # check if B has anything inside A
        if (a_x1 <= b_x1 <= a_x2 or a_x1 <= b_x2 <= a_x2) and (a_y1 <= b_y1 <= a_y2 or a_y1 <= b_y2 <= a_y2):
            return True
        # check if A has anything inside B
        if (b_x1 <= a_x1 <= b_x2 or b_x1 <= a_x2 <= b_x2) and (b_y1 <= a_y1 <= b_y2 or b_y1 <= a_y2 <= b_y2):
            return True
        
        return False

    def get_normal(self, depth_image, face_cord, stamp,dist, face_region):
        face_x1, face_y1, face_x2, face_y2 = face_cord
        image = depth_image[face_y1:face_y2, face_x1:face_x2]
        shift_left = face_x1
        shift_top = face_y1

        if min(depth_image.shape) == 0:
            return None


        biggerDim = np.argmax(image.shape) 
        # print(f"bigger dimention: {biggerDim}")
        bigLen = image.shape[biggerDim]
        smallLen = image.shape[~biggerDim]
        # print(f"bigLen = {bigLen}")
        # print(f"smallLen = {smallLen}")
        x1 = y1 = None # top-left
        x2 = y2 = None # bottom-right
        x3 = y3 = None # top-right
            
        for big_i in range(bigLen//2):
            whatPart = big_i/bigLen
            small_i = round(whatPart*smallLen)
            if biggerDim == 0:
                # bottom-left point
                if (x2==None and np.isnan(image[(bigLen-1)-big_i][small_i])==False):
                    x2 = small_i
                    y2 = (bigLen-1)-big_i
                # bottom-right point
                if (x1==None and np.isnan(image[(bigLen-1)-big_i][(smallLen-1)-small_i])==False):
                    x1 = (smallLen-1)-small_i
                    y1 = (bigLen-1)-big_i
                # top-right point
                if (x3==None and np.isnan(image[big_i][(smallLen-1)-small_i])==False):
                    x3 = (smallLen-1)-small_i
                    y3 = big_i
            else:
                # bottom-left point
                if (x2==None and np.isnan(image[(smallLen-1)-small_i][big_i])==False):
                    x2 = big_i
                    y2 = (smallLen-1)-small_i
                # bottom-right point
                if (x1==None and np.isnan(image[(smallLen-1)-small_i][(bigLen-1)-big_i])==False):
                    x1 = (bigLen-1)-big_i
                    y1 = (smallLen-1)-small_i
                # top-right point
                if (x3==None and np.isnan(image[small_i][(bigLen-1)-big_i])==False):
                    x3 = small_i
                    y3 = (bigLen-1)-big_i


        if x1 == None or x2 == None or x3 == None:
            return None
        # else:
        #     x1 = (face_x1+face_x2)//2+2
        #     x2 = (face_x1+face_x2)//2-2
        #     x3 = (face_x1+face_x2)//2
        #     y1 = (face_y1+face_y2)//2+2
        #     y2 = (face_y1+face_y2)//2-2
        #     y3 = (face_y1+face_y2)//2

        x1 += shift_left
        y1 += shift_top

        x2 += shift_left
        y2 += shift_top
        
        x3 += shift_left
        y3 += shift_top
        
        

        #print("\n< 1 > image [{:>3}] [{:>3}] = {:}".format(y1,x1,depth_image[y1][x1]))
        #print("< 2 > image [{:>3}] [{:>3}] = {:}".format(y2,x2,depth_image[y2][x2]))
        #print("< 3 > image [{:>3}] [{:>3}] = {:}".format(y3,x3,depth_image[y3][x3]))



        v1_cor = self.get_pose_face((x1,x1,y1,y1),depth_image[y1,x1],stamp)
        
        v2_cor = self.get_pose_face((x2,x2,y2,y2),depth_image[y2,x2],stamp)
        
        v3_cor = self.get_pose_face((x3,x3,y3,y3),depth_image[y3,x3],stamp)

        # try:
        #     # print("\n coords of points:")
        #     # print(f"\n{v1_cor.position}")
        #     # print(f"\n{v2_cor.position}")
        #     # print(f"\n{v3_cor.position}")
        # except:
        #     pass
        
        
        try: 
            #print(v1_cor,v2_cor,v3_cor)
            v3_cor.position.z += 0.1

            #rospy.sleep(2)
            v1 = np.array([v1_cor.position.x,v1_cor.position.y, v1_cor.position.z] )
            v2 = np.array([v2_cor.position.x,v2_cor.position.y, v2_cor.position.z] )
            v3 = np.array([v3_cor.position.x,v3_cor.position.y, v3_cor.position.z] )

            print(f"v1:  {v1}")
            print(f"v2:  {v2}")
            print(f"v3:  {v3}")

            v12 =v2-v1
            v13 =v3-v1
            
            v23 = v3-v2

            d12 = np.linalg.norm(v12)
            d13 = np.linalg.norm(v13)
            d23 = np.linalg.norm(v23)
            print(f"\n\tv12:  {v12}")
            print(f"\tv13:  {v13}")
            print(f"\tv13:  {v23}")
            print("\t----")
            print(f"\td12:  {d12}")
            print(f"\td13:  {d13}")
            print(f"\td23:  {d23}\n")

            if d12>0.2 or d13>0.2 or d23>0.2:
                print("Exiting like a pro!!!!!!!!!!!!")
                return None
            


            norm = np.cross(v12,v13)
            norm = norm/np.linalg.norm(norm)*0.5
            print(f"norm: {norm}")

            orig_arr = v1
            dest_arr = v1-norm
            print("before:",norm)

            #for pose in [v1_cor,v2_cor,v3_cor]:
            """
                if pose is not None:
                    # Create a marker used for visualization
                    self.marker_num += 1
                    marker = Marker()
                    marker.header.stamp = rospy.Time(0)
                    marker.header.frame_id = 'map'
                    marker.pose = pose
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.frame_locked = False
                    marker.lifetime = rospy.Duration.from_sec(10)
                    marker.id = self.marker_num
                    marker.scale = Vector3(0.1, 0.1, 0.1)
                    if pose == v1_cor:
                        marker.color = ColorRGBA(1,0,0,1)
                    elif pose == v2_cor:
                        marker.color = ColorRGBA(0,0,1,1)
                    else:
                        marker.color = ColorRGBA(0, 1, 0, 1)
                    #self.marker_array.markers.append(marker)

            #self.marker_array.markers.append(self.get_arrow_points(Vector3(1,1,1),Point(orig_arr[0],orig_arr[1],orig_arr[2]),Point(dest_arr[0],dest_arr[1],dest_arr[2]),3))
            self.markers_pub.publish(self.marker_array)
            """
            # doit = self.norm_accumulator(norm,v1,dist) #! ACCUMULATOR IS CALLED
            # # TODO: spremeni, da bo z akumulatorjem pozicij iz te skripte
            # if doit:
            #     self.marker_array.markers.append(self.get_arrow_points(Vector3(1,1,1),Point(orig_arr[0],orig_arr[1],orig_arr[2]),Point(dest_arr[0],dest_arr[1],dest_arr[2]),3))
            #     self.markers_pub.publish(self.marker_array)
            #     self.pic_pub.publish(CvBridge().cv2_to_imgmsg(face_region, encoding="passthrough"))
            print("after:",norm)
            
            #self.points_pub.publish(Point(orig_arr[0],orig_arr[1],orig_arr[2]))
            #self.points_pub.publish(Point(dest_arr[0],dest_arr[1],dest_arr[2]))
            #ustvarimo Image msg
            # Refrence :image = depth_image[face_y1:face_y2, face_x1:face_x2]
            """im_ms = Image()
            im_ms.header.stamp = rospy.Time(0)
            im_ms.header.frame_id = 'map'
            im_ms.height = image.shape[0]
            im_ms.width = image.shape[1]
            im_ms.data = image
            
            self.pic_pub.publish(im_ms)"""

        except Exception as e:
            print(e)
            norm = None
        return norm

    def get_pose_face(self,coords,dist,stamp):
        # Calculate the position of the detected face

        k_f = 554 # kinect focal length in pixels

        x1, x2, y1, y2 = coords

        face_x = self.dims[1] / 2 - (x1+x2)/2.
        face_y = self.dims[0] / 2 - (y1+y2)/2.

        angle_to_target = np.arctan2(face_x,k_f)

        # Get the angles in the base_link relative coordinate system
        x = dist*np.cos(angle_to_target)
        y = dist*np.sin(angle_to_target)

        ### Define a stamped message for transformation - directly in "base_link"
        #point_s = PointStamped()
        #point_s.point.x = x
        #point_s.point.y = y
        #point_s.point.z = 0.3
        #point_s.header.frame_id = "base_link"
        #point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp

        # Get the point in the "map" coordinate system
        try:
            point_world = self.tf_buf.transform(point_s, "map")

            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z
        except Exception as e:
            print(e)
            pose = None

        return pose

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
    
    def get_arrow_points(self, scale, head, tail, idn):
        self.marker_num += 1
        m = Marker()
        m.action = Marker.ADD
        m.header.frame_id= 'map'
        m.header.stamp = rospy.Time.now()
        m.ns = 'points_arrow'
        m.id = self.marker_num
        m.type = Marker.ARROW
        m.pose.orientation.y = 0
        m.pose.orientation.w = 1
        m.scale = scale
        m.color.r = 0.2
        m.color.g = 0.5
        m.color.b = 1.0
        m.color.a = 0.3

        m.points = [tail,head]
        return m
       

#! face detection end

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