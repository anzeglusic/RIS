#!/usr/bin/python3

import json
from pprint import pprint
import os
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'


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
alreadyExists = True
try:
    with open(f"{modelsDir}colorsDataset.json","r",encoding="utf-8") as f:
        library = json.loads(f.read())
        print(f"detections")
        print(f"            ring")
        print(f"                  r: {len(library['ring']['r'])}")
        print(f"                  g: {len(library['ring']['g'])}")
        print(f"                  b: {len(library['ring']['b'])}")
        print(f"                  y: {len(library['ring']['y'])}")
        print(f"                  c: {len(library['ring']['c'])}")
        print(f"            cylinder")
        print(f"                  r: {len(library['cylinder']['r'])}")
        print(f"                  g: {len(library['cylinder']['g'])}")
        print(f"                  b: {len(library['cylinder']['b'])}")
        print(f"                  y: {len(library['cylinder']['y'])}")
        print(f"                  c: {len(library['cylinder']['c'])}")
        print()
    
except json.decoder.JSONDecodeError as err:
    alreadyExists = False
    print("JSON error: {err}")
    print("I have created new dictionary")
except Exception as err:
    # alreadyExists = False
    print(err)

if alreadyExists == False:
    try:
        library = {}
        library["ring"] = {
            "r" : [],
            "g" : [],
            "b" : [],
            "y" : [],
            "c" : []
        }
        library["cylinder"] = {
            "r" : [],
            "g" : [],
            "b" : [],
            "y" : [],
            "c" : []
        }
        with open(f"{modelsDir}colorsDataset.json","w",encoding="utf-8") as f:
            f.write(json.dumps(library))
        
    except Exception as err:
        print(err)