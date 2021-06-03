#!/usr/bin/python3

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
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'

alreadyExists = True
try:
    with open(f"{modelsDir}colorsDataset.json","r",encoding="utf-8") as f:
        library = json.loads(f.read())

except json.decoder.JSONDecodeError as err:
    alreadyExists = False
    print("JSON error: {err}")
    print("I have created new dictionary")
except Exception as err:
    # alreadyExists = False
    print(err)


#! uncomment ONLY if you want to override a file !!!!!!!!!!!!
# if alreadyExists == False:
#     try:
#         library = {}
#         library["ring"] = {
#             "r" : [],
#             "g" : [],
#             "b" : [],
#             "y" : [],
#             "c" : []
#         }
#         library["cylinder"] = {
#             "r" : [],
#             "g" : [],
#             "b" : [],
#             "y" : [],
#             "c" : []
#         }
#         with open(f"{modelsDir}colorsDataset.json","w",encoding="utf-8") as f:
#             f.write(json.dumps(library))

#     except Exception as err:
#         print(err)

# =======================================================================================================
# =======================================================================================================
# =======================================================================================================
# =======================================================================================================
# =======================================================================================================

"""
library = {
    "ring" : {
        "r" : [ detection0, detection1, detection2, ...],
        "g" : [ detection0, detection1, detection2, ...],
        "b" : [ detection0, detection1, detection2, ...],
        "y" : [ detection0, detection1, detection2, ...],
        "c" : [ detection0, detection1, detection2, ...]
    },

    "cylinder" : {
        "r" : [ detection0, detection1, detection2, ...],
        "g" : [ detection0, detection1, detection2, ...],
        "b" : [ detection0, detection1, detection2, ...],
        "y" : [ detection0, detection1, detection2, ...],
        "c" : [ detection0, detection1, detection2, ...]
    }
}

        r --> red
        g --> green
        b --> blue
        y --> yellow
        c --> crna

        detection? --> list of pixels for 1 detection

        detection? = [ pixel0, pixel1, pixel2, ...]
"""
#! OD TUKAJ NAPREJ POČNEŠ KAR HOČEŠ !!!
"""
def create_list(dicta):
    li = []
    for i in library:
        for color in library[i]:
            print(color)
            for det in library[i][color]:
                #pridobimo 50 samplov
                #print(det)
                for it in range(100):
                    tem = random.sample(det,100)
                    #flattenamo sample
                    flatten_tem = functools.reduce(operator.iconcat,tem,[])
                    #appendamo color
                    flatten_tem.append(color)
                    #dodamo v li
                    li.append(flatten_tem)
    return li
"""

#datF = pd.DataFrame(create_list(library))
#print(datF.head)
#datF.to_pickle("./pixls100.pkl")

datF = pd.read_pickle(f"{modelsDir}pixls.pkl")
#print(datF.count())


df_x = datF.iloc[:,0:-1].to_numpy()
df_y = datF.iloc[:,-1].to_numpy()
#df_x =
#print(df_x)
#print(df_y)

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

randomForest_fit = randomForest.fit(df_x,df_y)
print("\nRandom Forest:")
print(f"\tscore: {randomForest_fit.oob_score_}\n")
filename = "random_forestV2"
pickle.dump(randomForest_fit, open(f"{filename}.sav", 'wb'))
"""
# dictionary je v spremenljivki "library"
allColors = ["r","g","b","y","c"]
numOfAllDetections = 0
numOfAllPixels = 0
for color in allColors:
    temp1 = library["ring"][color]
    temp2 = library["cylinder"][color]
    numOfAllDetections += len(temp1)
    numOfAllDetections += len(temp2)
    numOfAllPixels += sum([len(x) for x in temp1])
    numOfAllPixels += sum([len(x) for x in temp2])
print()
print(f"--------------------- num of detections ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = library[objectType][color]
        print(f"\t{color}: {len(temp)}\t( {round(len(temp)/numOfAllDetections*100)} % )")
print()

print(f"--------------------- num of pixels ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = [len(x) for x in library[objectType][color]]
        print(f"\t{color}: {sum(temp)}\t( {round(sum(temp)/numOfAllPixels*100)} % )")
print()

print(f"--------------------- (min, max) pixlov in 1 detection ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = [len(x) for x in library[objectType][color]]
        print(f"\t{color}: ({min(temp) if len(temp)>0 else 0}\t {max(temp) if len(temp)>0 else 0})")
print()

print(f"--------------------- detections per color ---------------------")
for color in allColors:
    print(f"{color}: {len(library['ring'][color])+len(library['cylinder'][color])}\t( {round((len(library['ring'][color])+len(library['cylinder'][color]))/numOfAllDetections*100)} % )")
print()

print(f"--------------------- pixels per color ---------------------")
for color in allColors:
    temp1 = [len(x) for x in library["ring"][color]]
    temp2 = [len(x) for x in library["cylinder"][color]]
    print(f"{color}: {sum(temp1)+sum(temp2)}\t( {round((sum(temp1)+sum(temp2))/numOfAllPixels*100)} % )")
print()
"""

for objectType in ["ring","cylinder"]:
    print(objectType)
    for color in ["r","g","b","y","c"]:
        blue = 200
        green = 200
        red = 200
        temp = [sum([(pixel[0]<blue and pixel[1]>green and pixel[2]<red) for pixel in detection])>5 for detection in library[objectType][color]]
        print(f"\t{color}: {sum(temp)}\t{round(sum(temp)/len(temp)*100) if len(temp)>0 else 0} %")