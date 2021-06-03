#!/usr/bin/python3

import json
from pprint import pprint
import os
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
# dictionary je v spremenljivki "library"
allColors = ["r","g","b","y","c"]
print()
print(f"--------------------- num of detections ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = library[objectType][color]
        print(f"\t{color}: {len(temp)}")
print()

print(f"--------------------- num of pixels ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = [len(x) for x in library[objectType][color]]
        print(f"\t{color}: {sum(temp)}")
print()

print(f"--------------------- (min, max) pixlov in 1 detection ---------------------")
for objectType in ["ring", "cylinder"]:
    print(f"{objectType}:")
    for color in allColors:
        temp = [len(x) for x in library[objectType][color]]
        print(f"\t{color}: ({min(temp) if len(temp)>0 else 0}\t {max(temp) if len(temp)>0 else 0})")
print()