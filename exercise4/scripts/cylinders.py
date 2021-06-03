from itertools import count
from pprint import pprint
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from skimage.color import rgb2hsv
import cv2
import pickle
import json
from pprint import pprint
import os
import random
import functools
import operator
import requests
import rospy
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Twist, Text
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'

def train_classifier(link,name):
    randomForest = RandomForestClassifier(n_estimators=100
                                        , criterion="gini"
                                        , max_depth=None
                                        , min_samples_split=2
                                        , min_samples_leaf=1
                                        , min_weight_fraction_leaf=0.0
                                        , max_features="auto"
                                        , max_leaf_nodes=None
                                        , min_impurity_decrease=0.0
                                        , bootstrap=True
                                        , oob_score=True
                                        , n_jobs=-1
                                        , random_state=None
                                        , verbose=0
                                        , warm_start=False
                                        , class_weight=None
                                        , ccp_alpha=0.0
                                        , max_samples=400
                                        )
    r = requests.get(link, allow_redirects=True)
    f = r.content
    f = f.decode(r.encoding)
    f = f.split('\n')

    for i in range(0,len(f)):
        f[i] = f[i].split(',')
    f.pop()
    datF = pd.DataFrame(f)
    print(datF)
    npF = datF.to_numpy()
    trainX = npF[:,:-1]
    trainY = npF[:,-1]

    random_forest_fit = randomForest.fit(trainX,trainY)
    pickle.dump(random_forest_fit, open(f"{modelsDir}{name}.sav", 'wb'))
    print(f"{name}.sav:\n\tscore: {randomForest_fit.oob_score_}\n")
    #print(f)

def listener():
    try:
        link_class = rospy.wait_for_message("/classifier", Text)
        link =split()
        train_classiffier(link[0],link[1])
    except Exception as e:
        print(e)
        return 0
    return 0


rospy.init_node('cylinders', anonymous=True)

def main():

    # rate = rospy.Rate(1.25)
    rate = rospy.Rate(100)

    #! ch = input("Enter type(color|shape eq. rc - red cylinder)")

    loopTimer = rospy.Time.now().to_sec()
    # print(sleepTimer)
    while not rospy.is_shutdown():
        # print("hello!")

        '''link_class == "<<link>> imefila.sv" '''

        rate.sleep()

    #train_classifier("https://box.vicos.si/rins/13.txt","test1")

if __name__ == '__main__':
    main()
