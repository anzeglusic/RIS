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
import pandas as pd
from collections import Counter
import random
import functools
import operator

def keepExploring(positions):
    if len(positions["face"]) < 4:
        return True
    else:
        # cehck if valid detection
        for face_dict in positions["face"]:
            if len(face_dict["detectedPositions"]) < 3:
                return True

    if len(positions["cylinder"]) <4:
        return True
    # else:
    #     # cehck if valid detection
    #     for cylinder_dict in positions["cylinder"]:
    #         if len(cylinder_dict["detectedPositions"]) < 3:
    #             return True

    if len(positions["ring"]) <4:
        return True
    else:
        # cehck if valid detection
        for ring_dict in positions["ring"]:
            if len(ring_dict["detectedPositions"]) < 3:
                return True

    # if len(positions["QR"])< 8:
    #     return True
    for cylinder in positions["cylinder"]:
        if cylinder["QR_index"] is None:
            return True
    if len(positions["digits"]) <4:
        return True
    return False

def checkForApproach(positions,objectType,publisher):
    for obj in positions:
        if obj["approached"]:
            continue
        if len(obj["detectedPositions"]) < 3 and objectType=="face":
            continue
        if objectType == "face":
            #rabimo pristopiti poslati moramo twist message za normalo
            obj["approached"] = True
            say("Going to check a face.")
            publisher.publish(Vector3(obj["averageNormal"][0]+obj["averagePostion"][0],obj["averageNormal"][1]+obj["averagePostion"][1],obj["averageNormal"][2]+obj["averagePostion"][2]),Vector3(obj["averagePostion"][0],obj["averagePostion"][1],obj["averagePostion"][2]))
        elif objectType == "cylinder":
            obj["approached"] = True
            say("Going to check a cylinder.")
            publisher.publish(Point(obj["averagePostion"][0],obj["averagePostion"][1],obj["averagePostion"][2]))
    return True

def resetApproachesForTask(positions):
    for objT in positions:
        if objT == "QR" or objT == "digits":
            continue
        positions[objT]["approached"] = False
    return positions

def checkPosition(positions,base,objectType, publisher):
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

        ],
        "face": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "approached": False,
                "avgMarkerId": None,
                "QR_index": None,
                "digits_index": None
            },
    }
    '''
    for area in positions:
        if area["approached"] == True:
            continue
        distArr = area["averagePostion"]-np.array([base["x"],base["y"],base["z"]])
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
            publisher.publish(Point(area["averagePostion"][0],area["averagePostion"][1],1))
        elif objectType == "cylinder":
            # cylinder
            publisher.publish(Point(area["averagePostion"][0],area["averagePostion"][1],0))
        else:
            print("SOMETHING WENT FUCKING WRONG !!!!!!!")
        best_color = chose_color(area["color"])
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

def chose_color(colorDict):
    b = colorDict["b"]
    g = colorDict["g"]
    r = colorDict["r"]
    c = colorDict["c"]
    y = colorDict["y"]

    tag =   ["b","g","r","y","c"]
    value = [ b , g , r , y , c ]


    bestIndx = np.argmax(value)
    return tag[bestIndx]

def rgba_from_char(color_char):
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

def addPosition(newPosition, objectType, color_char, positions, nM, m_arr, markers_pub, showEveryDetection=True, normal=None, data=None, mask=None, modelName=None, publisher = None):
    '''
    positions = {
        "ring": [
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None }

        ],
        "cylinder": [
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None }

        ],
        "face": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "approached": False,
                "avgMarkerId": None,
                "QR_index": None,
                "digits_index": None,
                #!DODANO
                "has_mask": False,
                "stage": "warning",
                "was_vaccinated": None,
                "doctor": None,
                "hours_exercise": None,
                "right_vaccine": None
            },
            ...
        ],
        "QR": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "data": None,
                "avgMarkerId": None,
                "isAssigned": False,
                "modelName": None
            },
            ...
        ],
        "digits": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "data": None,
                "avgMarkerId": None,
                "isAssigned": False
            },
            ...
        ]
    }
    '''
    if not (objectType in positions):
        return (nM, m_arr, positions)

    qrNormalLength = 0.25
    digitsNormalLength = 0.25
    faceNormalLength = 0.5
    # make normal of magnitude 1
    if objectType == "face" or objectType=="QR" or objectType=="digits":
        if normal is None:
            return (nM, m_arr, positions)
        normal = normal/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

    unique_position = True
    for area in positions[objectType]:
        if objectType=="face" or objectType=="QR" or objectType=="digits":
            c_sim = np.dot(normal,area["averageNormal"])/(np.linalg.norm(normal)*np.linalg.norm(area["averageNormal"]))
            # print("similarity:",c_sim)
            if c_sim<=0.5:
                continue
        area_avg = area["averagePostion"]
        dist_vector = area_avg - newPosition
        dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)

            # print("Dist --> ",dist)
        if objectType == "ring":
            if dist > 0.35:
                continue
        else:
            if dist > 0.5:
                continue

        unique_position = False

        if objectType == "face":
            if mask == True:
                area["mask"]["yes"] += 1
            else:
                area["mask"]["no"] += 1

        if objectType=="QR" or objectType=="digits":
            # print(data)
            if area["data"] is None:
                area["data"] = data
            # else:
            #     assert area["data"]==data, f"\n\n\tnovi {objectType} podatki so drugačni kot so predvideni za to pozicijo"
        if objectType=="QR":
            if area["modelName"] is None:
                area["modelName"] = modelName
            

        # depending on the type of object
        if objectType=="cylinder" or objectType=="ring":
            # color
            area["color"][color_char] += 1
        elif objectType=="face" or objectType=="QR" or objectType=="digits":
            # collection of normals
            area["detectedNormals"].append(normal)
            # average normal
            # print(area["detectedNormals"])
            area["averageNormal"] = np.sum(area["detectedNormals"],axis=0)/len(area["detectedNormals"])
            area["averageNormal"] = area["averageNormal"]/np.sqrt(area["averageNormal"][0]**2 + area["averageNormal"][1]**2 + area["averageNormal"][2]**2)

        # collection of positions
        area["detectedPositions"].append(newPosition.copy())

        # average
        area["averagePostion"] = np.sum(area["detectedPositions"],axis=0)/len(area["detectedPositions"])

        # average marker
        if objectType=="cylinder" or (objectType=="ring" and len(area["detectedPositions"]) >= 3):
            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = area["averagePostion"][0]
            pose.position.y = area["averagePostion"][1]
            pose.position.z = area["averagePostion"][2]+0.1

            # Create a marker used for visualization
            nM += 1
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
                marker.id = nM
                area["avgMarkerId"] = nM
            else:
                marker.id = area["avgMarkerId"]

            best_color = chose_color(area["color"])
            marker.color = rgba_from_char(best_color)

            m_arr.markers.append(marker)
            markers_pub.publish(m_arr)

        elif objectType=="face" and len(area["detectedPositions"]) >= 3:
            nM += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(0)

            if area["avgMarkerId"] == None:
                marker.id = nM
                area["avgMarkerId"] = nM
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

            #change 0.5 => arrow length
            dest_arr = orig_arr - area["averageNormal"]*faceNormalLength
            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

            marker.points = [tail,head]

            m_arr.markers.append(marker)
            markers_pub.publish(m_arr)

        elif objectType=="QR":
            nM += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(0)

            if area["avgMarkerId"] == None:
                marker.id = nM
                area["avgMarkerId"] = nM
            else:
                marker.id = area["avgMarkerId"]

            marker.ns = 'points_arrow'
            marker.type = Marker.ARROW
            marker.pose.orientation.y = 0
            marker.pose.orientation.w = 1
            marker.scale = Vector3(0.5,0.5,0.5)
            marker.color = ColorRGBA(0,0,1,0.5)

            orig_arr = area["averagePostion"].copy()
            orig_arr[2] += 0.1
            dest_arr = orig_arr - area["averageNormal"]*qrNormalLength
            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

            marker.points = [tail,head]

            m_arr.markers.append(marker)
            markers_pub.publish(m_arr)

        elif objectType=="digits":
            nM += 1
            marker = Marker()
            marker.header.stamp = rospy.Time(0)
            marker.header.frame_id = "map"
            marker.action = Marker.ADD
            marker.frame_locked = False
            marker.lifetime = rospy.Duration.from_sec(0)

            if area["avgMarkerId"] == None:
                marker.id = nM
                area["avgMarkerId"] = nM
            else:
                marker.id = area["avgMarkerId"]

            marker.ns = 'points_arrow'
            marker.type = Marker.ARROW
            marker.pose.orientation.y = 0
            marker.pose.orientation.w = 1
            marker.scale = Vector3(0.5,0.5,0.5)
            marker.color = ColorRGBA(0,1,0,0.5)

            orig_arr = area["averagePostion"].copy()
            orig_arr[2] += 0.1
            dest_arr = orig_arr - area["averageNormal"]*digitsNormalLength
            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

            marker.points = [tail,head]

            m_arr.markers.append(marker)
            markers_pub.publish(m_arr)

    if unique_position == True:
        if objectType=="ring":
            colorDict = {"r":0,"g":0,"b":0,"y":0,"w":0,"c":0}
            colorDict[color_char] += 1
            positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "color": colorDict,
                                                "approached": False,
                                                "avgMarkerId": None
                                                })
        elif objectType=="cylinder":
            colorDict = {"r":0,"g":0,"b":0,"y":0,"w":0,"c":0}
            colorDict[color_char] += 1
            positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "color": colorDict,
                                                "approached": False,
                                                "avgMarkerId": None,
                                                "QR_index": None
                                                })
        elif objectType=="face":
            print("\n\nAdding new face\n\n")
            positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "averageNormal": normal.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "detectedNormals":[normal.copy()],
                                                "approached": False,
                                                "avgMarkerId": None,
                                                "QR_index": None,
                                                "digits_index": None,
                                                "mask": {
                                                    "yes":1 if mask==True else 0,
                                                    "no":1 if mask == False else 0
                                                },
                                                "stage": "warning",
                                                "was_vaccinated": None,
                                                "doctor": None,
                                                "hours_exercise": None,
                                                "right_vaccine": None
                                                })
        elif objectType=="QR":
            print("\n\nAdding new QR code\n\n")
            positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "averageNormal": normal.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "detectedNormals":[normal.copy()],
                                                "data": data,
                                                "avgMarkerId": None,
                                                "isAssigned": False,
                                                "modelName": modelName
                                                })

            if data.startswith("https"):
                publisher.publish(f"{data} {modelName}")

        elif objectType=="digits":
            print("\n\nAdding new digits\n\n")
            positions[objectType].append({ "averagePostion": newPosition.copy(),
                                                "averageNormal": normal.copy(),
                                                "detectedPositions":[newPosition.copy()],
                                                "detectedNormals":[normal.copy()],
                                                "data": data,
                                                "avgMarkerId": None,
                                                "isAssigned": False
                                                })
    # Create a Pose object with the same position
    pose = Pose()
    pose.position.x = newPosition[0]
    pose.position.y = newPosition[1]
    pose.position.z = newPosition[2]
    if showEveryDetection==True:
        # Create a marker used for visualization
        nM += 1
        marker = Marker()
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "map"
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(0)
        marker.id = nM

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
            dest_arr = orig_arr - normal*faceNormalLength
            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

            marker.points = [tail,head]
        elif objectType=="QR" or objectType=="digits":
            marker.ns = 'points_arrow'
            marker.type = Marker.ARROW
            marker.pose.orientation.y = 0
            marker.pose.orientation.w = 1
            marker.scale = Vector3(0.5,0.5,0.5)
            marker.color = ColorRGBA(0.5,0.5,0.5,0.1)

            orig_arr = newPosition
            dest_arr = orig_arr - normal*qrNormalLength
            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

            marker.points = [tail,head]

        m_arr.markers.append(marker)
        markers_pub.publish(m_arr)

    # updating connections
    (nM, m_arr, positions) = update_positions(nM, m_arr, positions, markers_pub,faceNormalLength, qrNormalLength, digitsNormalLength)

    return (nM, m_arr, positions)

def update_positions(nM, m_arr, positions, markers_pub, faceNormalLength, qrNormalLength, digitsNormalLength):
    '''
    positions = {
        "ring": [
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None }
        ],
        "cylinder": [
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None  },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None  },
            { "averagePostion": np.array([x,y,z]), "detectedPositions": [ pos0, pos1, pos2, pos3], "color": {"b":0, "g":0, "r":0, ...}, "approached": False,"avgMarkerId": None, "QR_index": None  }
        ],
        "face": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "approached": False,
                "avgMarkerId": None,
                "QR_index": None,
                "digits_index": None
                "mask": {
                    "yes":int,
                    "no": int
                }
            },
            ...
        ],
        "QR": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "data": None,
                "avgMarkerId": None,
                "isAssigned": False
            },
            ...
        ],
        "digits": [
            {
                "averagePostion": np.array([x,y,z]),
                "averageNormal": np.array([x,y,z]),
                "detectedPositions": [ pos0, pos1, pos2, pos3],
                "detectedNormals": [ pos0, pos1, pos2, pos3],
                "data": None,
                "avgMarkerId": None,
                "isAssigned": False
            },
            ...
        ]
    }
    '''

    # check if any seperate objects with same objectType can be joined
    startOver = True
    while startOver == True:
        startOver = False
        # only position dependent
        for objectType in ["ring","cylinder"]:
            for i in range(len(positions[objectType])-1, 0, -1):
                # TODO: consider only those that have 3 or more detections --> maybe could help
                doubleBreak = False
                for j in range(i-1, -1, -1):
                    # TODO: consider only those that have 3 or more detections --> maybe could help
                    area_avg = positions[objectType][i]["averagePostion"]
                    dist_vector = area_avg - positions[objectType][j]["averagePostion"]
                    dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                    if objectType=="ring" and dist>0.35:
                        continue
                    elif objectType=="cylinder" and dist > 0.5:
                        continue

                    else:
                        # merge (i) into location of index (j) + do new average calculations
                        # detectedPositions
                        positions[objectType][j]["detectedPositions"].extend(positions[objectType][i]["detectedPositions"])
                        # color
                        for colorChar in ["r","g","b","y","w","c"]:
                            positions[objectType][j]["color"][colorChar] += positions[objectType][i]["color"][colorChar]
                        # approached
                        positions[objectType][j]["approached"] = positions[objectType][j]["approached"] or positions[objectType][i]["approached"]
                        # avgMarkerId --> do nothing
                        # averagePostion
                        positions[objectType][j]["averagePostion"] = np.sum(positions[objectType][j]["detectedPositions"],axis=0)/len(positions[objectType][j]["detectedPositions"])
                        # update average marker if index (j)
                        # Create a Pose object with the same position
                        pose = Pose()
                        pose.position.x = positions[objectType][j]["averagePostion"][0]
                        pose.position.y = positions[objectType][j]["averagePostion"][1]
                        pose.position.z = positions[objectType][j]["averagePostion"][2]+0.1

                        # Create a marker used for visualization
                        nM += 1
                        marker = Marker()
                        marker.header.stamp = rospy.Time(0)
                        marker.header.frame_id = "map"
                        marker.pose = pose
                        marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.frame_locked = False
                        marker.lifetime = rospy.Duration.from_sec(0)
                        marker.scale = Vector3(0.1, 0.1, 0.1)

                        if positions[objectType][j]["avgMarkerId"] == None:
                            marker.id = nM
                            positions[objectType][j]["avgMarkerId"] = nM
                        else:
                            marker.id = positions[objectType][j]["avgMarkerId"]

                        best_color = chose_color(positions[objectType][j]["color"])
                        marker.color = rgba_from_char(best_color)

                        m_arr.markers.append(marker)
                        markers_pub.publish(m_arr)
                        # delete average marker of index (i) --> or make it transparent
                        if not (positions[objectType][i]["avgMarkerId"] is None):
                            # Create a Pose object with the same position
                            nM += 1
                            pose = Pose()
                            pose.position.x = positions[objectType][i]["averagePostion"][0]
                            pose.position.y = positions[objectType][i]["averagePostion"][1]
                            pose.position.z = positions[objectType][i]["averagePostion"][2]+0.1

                            # Create a marker used for visualization
                            marker = Marker()
                            marker.header.stamp = rospy.Time(0)
                            marker.header.frame_id = "map"
                            marker.pose = pose
                            marker.type = Marker.CUBE if objectType=="ring" else Marker.CYLINDER
                            marker.action = Marker.ADD
                            marker.frame_locked = False
                            marker.lifetime = rospy.Duration.from_sec(0)
                            marker.scale = Vector3(0.1, 0.1, 0.1)
                            marker.id = positions[objectType][i]["avgMarkerId"]

                            marker.color = ColorRGBA(1,1,1,0) # make it white + fully transparent

                            m_arr.markers.append(marker)
                            markers_pub.publish(m_arr)
                        # delete location of index (i)
                        positions[objectType].pop(i)
                        # once we delete, we should start over
                        startOver = True
                        doubleBreak = True
                        break
                if doubleBreak:
                    break

        # normal dependent
        for objectType in ["face","QR","digits"]:
            for i in range(len(positions[objectType])-1, 0, -1):
                # TODO: consider only those faces that have 3 or more detections --> maybe could help
                doubleBreak = False
                for j in range(i-1, -1, -1):
                    # TODO: consider only those faces that have 3 or more detections --> maybe could help
                    c_sim = np.dot(positions[objectType][i]["averageNormal"],positions[objectType][j]["averageNormal"])/(np.linalg.norm(positions[objectType][i]["averageNormal"])*np.linalg.norm(positions[objectType][j]["averageNormal"]))
                    if c_sim<=0.5:
                        continue
                    area_avg = positions[objectType][i]["averagePostion"]
                    dist_vector = area_avg - positions[objectType][j]["averagePostion"]
                    dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                    if dist > 0.5:
                        continue
                    else:
                        # merge (i) into location of index (j) + do new average calculations

                        # detectedPositions
                        positions[objectType][j]["detectedPositions"].extend(positions[objectType][i]["detectedPositions"])
                        # detectedNormals
                        positions[objectType][j]["detectedNormals"].extend(positions[objectType][i]["detectedNormals"])
                        if objectType == "face":
                            # approached
                            positions[objectType][j]["approached"] = positions[objectType][j]["approached"] or positions[objectType][i]["approached"]

                            # mask
                            positions[objectType][j]["mask"]["yes"] += positions[objectType][i]["mask"]["yes"]
                            positions[objectType][j]["mask"]["no"] += positions[objectType][i]["mask"]["no"]
                        # avgMarkerId --> do nothing
                        # QR_index --> do nothing
                        # digits_index --> do nothing
                        # averagePostion
                        positions[objectType][j]["averagePostion"] = np.sum(positions[objectType][j]["detectedPositions"],axis=0)/len(positions[objectType][j]["detectedPositions"])
                        # averageNormal
                        positions[objectType][j]["averageNormal"] = np.sum(positions[objectType][j]["detectedNormals"],axis=0)/len(positions[objectType][j]["detectedNormals"])
                        positions[objectType][j]["averageNormal"] = positions[objectType][j]["averageNormal"]/np.sqrt(positions[objectType][j]["averageNormal"][0]**2 + positions[objectType][j]["averageNormal"][1]**2 + positions[objectType][j]["averageNormal"][2]**2)
                        # update average marker of index (j)
                        nM += 1
                        marker = Marker()
                        marker.header.stamp = rospy.Time(0)
                        marker.header.frame_id = "map"
                        marker.action = Marker.ADD
                        marker.frame_locked = False
                        marker.lifetime = rospy.Duration.from_sec(0)

                        if positions[objectType][j]["avgMarkerId"] == None:
                            marker.id = nM
                            positions[objectType][j]["avgMarkerId"] = nM
                        else:
                            marker.id = positions[objectType][j]["avgMarkerId"]

                        marker.ns = 'points_arrow'
                        marker.type = Marker.ARROW
                        marker.pose.orientation.y = 0
                        marker.pose.orientation.w = 1
                        if objectType == "face":
                            marker.scale = Vector3(1,1,1)
                            marker.color = ColorRGBA(1,0,0,0.5)
                        elif objectType == "QR":
                            marker.scale = Vector3(0.5,0.5,0.5)
                            marker.color = ColorRGBA(0,0,1,0.5)
                        elif objectType == "digits":
                            marker.scale = Vector3(0.5,0.5,0.5)
                            marker.color = ColorRGBA(0,1,0,0.5)


                        orig_arr = positions[objectType][j]["averagePostion"].copy()
                        orig_arr[2] += 0.1

                        if objectType == "face":
                            dest_arr = orig_arr - positions[objectType][j]["averageNormal"]*faceNormalLength
                        elif objectType == "QR":
                            dest_arr = orig_arr - positions[objectType][j]["averageNormal"]*qrNormalLength
                        elif objectType == "digits":
                            dest_arr = orig_arr - positions[objectType][j]["averageNormal"]*digitsNormalLength

                        head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
                        tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

                        marker.points = [tail,head]

                        m_arr.markers.append(marker)
                        markers_pub.publish(m_arr)
                        # delete average marker of index (i) --> or make it transparent
                        if not (positions[objectType][i]["avgMarkerId"] is None):

                            nM += 1
                            marker = Marker()
                            marker.header.stamp = rospy.Time(0)
                            marker.header.frame_id = "map"
                            marker.action = Marker.ADD
                            marker.frame_locked = False
                            marker.lifetime = rospy.Duration.from_sec(0)

                            marker.id = positions[objectType][i]["avgMarkerId"]

                            marker.ns = 'points_arrow'
                            marker.type = Marker.ARROW
                            marker.pose.orientation.y = 0
                            marker.pose.orientation.w = 1
                            if objectType == "face":
                                marker.scale = Vector3(1,1,1)
                                marker.color = ColorRGBA(1,1,1,0)
                            elif objectType == "QR":
                                marker.scale = Vector3(0.5,0.5,0.5)
                                marker.color = ColorRGBA(1,1,1,0)
                            elif objectType == "digits":
                                marker.scale = Vector3(0.5,0.5,0.5)
                                marker.color = ColorRGBA(1,1,1,0)


                            orig_arr = positions[objectType][i]["averagePostion"].copy()
                            orig_arr[2] += 0.1

                            if objectType == "face":
                                dest_arr = orig_arr - positions[objectType][i]["averageNormal"]*faceNormalLength
                            elif objectType == "QR":
                                dest_arr = orig_arr - positions[objectType][i]["averageNormal"]*qrNormalLength
                            elif objectType == "digits":
                                dest_arr = orig_arr - positions[objectType][i]["averageNormal"]*digitsNormalLength

                            head = Point(orig_arr[0],orig_arr[1],orig_arr[2])
                            tail = Point(dest_arr[0],dest_arr[1],dest_arr[2])

                            marker.points = [tail,head]

                            m_arr.markers.append(marker)
                            markers_pub.publish(m_arr)
                        # delete location of index (i)
                        positions[objectType].pop(i)
                        # once we delete, we should start over
                        startOver = True
                        doubleBreak = True
                        break
                if doubleBreak:
                    break

    # reset connections
    for QR_dict in positions["QR"]:
        QR_dict["isAssigned"] = False
    for digits_dict in positions["digits"]:
        digits_dict["isAssigned"] = False
    for cylinder_dict in positions["cylinder"]:
        cylinder_dict["QR_index"] = None
    for face_dict in positions["face"]:
        face_dict["QR_index"] = None
        face_dict["digits_index"] = None

    # povezovanje "digits" s "face"
    for i,digits_dict in enumerate(positions["digits"]):
        closestObjectKey = None
        closestObjectIndx = None
        closestObjectDist = np.inf

        # check all faces
        for j,face_dict in enumerate(positions["face"]):
            if len(face_dict["detectedPositions"]) < 3:
                continue
            # include only those that have correct oriantation
            c_sim = np.dot(digits_dict["averageNormal"],face_dict["averageNormal"])/(np.linalg.norm(digits_dict["averageNormal"])*np.linalg.norm(face_dict["averageNormal"]))
            if c_sim<=0.5:
                continue

            dist_vector = digits_dict["averagePostion"] - face_dict["averagePostion"]
            dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)

            if dist < closestObjectDist:
                closestObjectDist = dist
                closestObjectKey = "face"
                closestObjectIndx = j


        # check if distance is ok
        if closestObjectDist > 0.5:
            continue

        # assign
        digits_dict["isAssigned"] = True
        # assert positions[closestObjectKey][closestObjectIndx]["digits_index"] is None, "Objekt ima že določen svoj digits"
        positions[closestObjectKey][closestObjectIndx]["digits_index"] = i


    # povezovanej "QR" s "face"/"cylinder"
    for i,QR_dict in enumerate(positions["QR"]):
        closestObjectKey = None
        closestObjectIndx = None
        closestObjectDist = np.inf

        if (QR_dict["data"] is None) or QR_dict["data"].startswith("https"):
            # chack all cylinders
            for j,cylinder_dict in enumerate(positions["cylinder"]):
                # if len(cylinder_dict["detectedPositions"]) < 3:
                #     continue
                dist_vector = QR_dict["averagePostion"] - cylinder_dict["averagePostion"]
                dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                if dist < closestObjectDist:
                    closestObjectDist = dist
                    closestObjectKey = "cylinder"
                    closestObjectIndx = j


        if (QR_dict["data"] is None) or (not QR_dict["data"].startswith("https")):
            # check all faces
            for j,face_dict in enumerate(positions["face"]):
                if len(face_dict["detectedPositions"]) < 3:
                    continue
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
        # assert positions[closestObjectKey][closestObjectIndx]["QR_index"] is None, "Objekt ima že določeno svojo QR kodo"
        positions[closestObjectKey][closestObjectIndx]["QR_index"] = i


    # pprint(positions)
    return (nM, m_arr, positions)

def say(sentance):
    # to play a sound !!!!!!!!!!!!!!!!!!!!
    print(f"\n\t--> {sentance}\n")
    subprocess.run(["rosrun" , "sound_play", "say.py", sentance])

def flatten_list(inpt):
    samples = []
    #print(type(inpt))
    #naredimo 4 sample
    for it in range(4):
        tem = random.sample(inpt,50)
        #print("ded")
        #flattenamo sample
        flatten_tem = functools.reduce(operator.iconcat,tem,[])

        #dodamo v li
        samples.append(flatten_tem)
    return samples

def calc_rgbV2(all_points,random_forest,objectType):

    samples = flatten_list(all_points)
    #print(df_x)
    df_x = pd.DataFrame(samples)

    y = random_forest.predict(df_x.to_numpy())
    dict_res = Counter(y)
    #print(dict_res)
    res = dict_res.most_common(1)
    if objectType=="ring" and res[0][0]=="c":
        blue = 150
        green = 200
        red = 150
        temp = [(pixel[0]<blue and pixel[1]>green and pixel[2]<red) for pixel in all_points]
        # print(f"\t\t\t\t\t{sum(temp)}")
        if sum(temp)>0:
            return "g"
    return res[0][0]

def calc_rgb(point,knn_RGB,random_forest_RGB,knn_HSV,random_forest_HSV):
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

    y = knn_RGB.predict(arr)
    c1 = y[0]
    print(f"knn_BGR:")
    print(f"\tprediction: {y[0]}")
    y = random_forest_RGB.predict(arr)
    c2 = y[0]
    print(f"random_forest_RGB:")
    print(f"\tprediction: {y[0]}")

    y = knn_HSV.predict(arr2)
    c3 = y[0]
    print(f"knn_HSV:")
    print(f"\tprediction: {y[0]}")
    y = random_forest_HSV.predict(arr2)
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

def get_pose(xin,yin,dist, depth_im,objectType,depth_stamp,color_char, tf_buf):
    # Calculate the position of the detected ellipse
    print(f"dist: {dist}")
    k_f = 525 # kinect focal length in pixels

    xin = -(xin - depth_im.shape[1]/2)

    angle_to_target = np.arctan2(xin,k_f)

    # if objectType == "cylinder":
    #     dist += 0.125
    #     print("adding depth !!!!!!!!!!!!!!")

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

    # Get the point in the "map" coordinate system
    point_world = tf_buf.transform(point_s, "map")

    #! USELESS CODE
    """
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
    """

    # Create a Pose object with the same position
    pose = Pose()
    pose.position.x = point_world.point.x
    pose.position.y = point_world.point.y
    pose.position.z = point_world.point.z


    return pose

def calc_pnts(el):
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

def calc_thresholds(frame):
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

def bgr2gray(bgr_image):
    return (np.sum(bgr_image.copy().astype(np.float),axis=2)/3).astype(np.uint8)

def gray2bgr(bgr_image):
    newImage = np.zeros((bgr_image.shape[0],bgr_image.shape[1],3))

    newImage[:,:,0] = bgr_image.copy()
    newImage[:,:,1] = bgr_image.copy()
    newImage[:,:,2] = bgr_image.copy()

    return newImage.astype(np.uint8)

def does_it_cross(a_cords_x, a_cords_y, b_cords_x, b_cords_y):
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

def get_normal(depth_image, face_cord, stamp,dist, face_region, tf_buff):
    face_x1, face_y1, face_x2, face_y2 = face_cord
    image = depth_image[face_y1:face_y2, face_x1:face_x2]
    shift_left = face_x1
    shift_top = face_y1


    if min(depth_image.shape) == 0:
        return None

    if min(image.shape) == 0:
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



    v1_cor = get_pose_face((x1,x1,y1,y1),depth_image[y1,x1],stamp, depth_image.shape, tf_buff)

    v2_cor = get_pose_face((x2,x2,y2,y2),depth_image[y2,x2],stamp, depth_image.shape, tf_buff)

    v3_cor = get_pose_face((x3,x3,y3,y3),depth_image[y3,x3],stamp, depth_image.shape, tf_buff)

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

        # print(f"v1:  {v1}")
        # print(f"v2:  {v2}")
        # print(f"v3:  {v3}")

        v12 =v2-v1
        v13 =v3-v1

        v23 = v3-v2

        d12 = np.linalg.norm(v12)
        d13 = np.linalg.norm(v13)
        d23 = np.linalg.norm(v23)
        # print(f"\n\tv12:  {v12}")
        # print(f"\tv13:  {v13}")
        # print(f"\tv13:  {v23}")
        # print("\t----")
        # print(f"\td12:  {d12}")
        # print(f"\td13:  {d13}")
        # print(f"\td23:  {d23}\n")

        if d12>0.2 or d13>0.2 or d23>0.2:
            print("Exiting like a pro!!!!!!!!!!!!")
            return None



        norm = np.cross(v12,v13)
        norm = norm/np.linalg.norm(norm)*0.5
        # print(f"norm: {norm}")

        orig_arr = v1
        dest_arr = v1-norm
        # print("before:",norm)

    except Exception as e:
        print(e)
        norm = None
    return norm

def get_pose_face(coords, dist, stamp, dims, tf_buf):
    # Calculate the position of the detected face

    k_f = 554 # kinect focal length in pixels

    x1, x2, y1, y2 = coords

    face_x = dims[1] / 2 - (x1+x2)/2.
    face_y = dims[0] / 2 - (y1+y2)/2.

    angle_to_target = np.arctan2(face_x,k_f)

    # Get the angles in the base_link relative coordinate system
    x = dist*np.cos(angle_to_target)
    y = dist*np.sin(angle_to_target)


    # Define a stamped message for transformation - in the "camera rgb frame"
    point_s = PointStamped()
    point_s.point.x = -y
    point_s.point.y = 0
    point_s.point.z = x
    point_s.header.frame_id = "camera_rgb_optical_frame"
    point_s.header.stamp = stamp

    # Get the point in the "map" coordinate system
    try:
        point_world = tf_buf.transform(point_s, "map")

        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z
    except Exception as e:
        print(e)
        pose = None

    return pose

def get_arrow_points(scale, head, tail, idn, marker_num):
    marker_num += 1
    m = Marker()
    m.action = Marker.ADD
    m.header.frame_id= 'map'
    m.header.stamp = rospy.Time.now()
    m.ns = 'points_arrow'
    m.id = marker_num
    m.type = Marker.ARROW
    m.pose.orientation.y = 0
    m.pose.orientation.w = 1
    m.scale = scale
    m.color.r = 0.2
    m.color.g = 0.5
    m.color.b = 1.0
    m.color.a = 0.3

    m.points = [tail,head]
    return (marker_num,m)

def has_mask(mask):
    return True if mask["yes"] >= mask["no"] else False