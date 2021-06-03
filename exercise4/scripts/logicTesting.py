#!/usr/bin/python3

import os
import sys
import numpy as np
import json
from pprint import pprint
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/')



# /home/sebastjan/Documents/faks/3letnk/ris/ROS_task/src/exercise4/scripts

#modelsDir = "/home/code8master/Desktop/wsROS/src/RIS/exercise4/scripts"
modelsDir = '/'.join(os.path.realpath(__file__).split('/')[0:-1])+'/'

# "face": [
#     {
#         "averagePostion": np.array([x,y,z]),
#         "averageNormal": np.array([x,y,z]),
#         "detectedPositions": [ pos0, pos1, pos2, pos3],
#         "detectedNormals": [ pos0, pos1, pos2, pos3],
#         "approached": False,
#         "avgMarkerId": None,
#         "QR_index": None,
#         "digits_index": None,
#         "has_mask": False
#     },
        
class Brains:
    def __init__(self):
        self.faces = [ 
                        {
                            "averagePostion": np.array([0,0,0]),
                            "averageNormal": np.array([1,0,0]),
                            "digits_index": None,
                            "has_mask": False,
                            "id": 1,
                            "stage": "warning"
                        },
                        {
                            "averagePostion": np.array([0,0.5,0]),
                            "averageNormal": np.array([1,0,0]),
                            "digits_index": None,
                            "has_mask": True,
                            "id": 2,
                            "stage": "warning"
                        }
                     ]
        

    def processStage(self, indx):
        try:
            face = self.faces[indx]
        except Exception as err:
            print(f"ProcessStage error: {err}")
            return

        stage = face["stage"]
        if stage == "warning":
            # TODO: check if it has a mask
            if face["has_mask"]:
                pass
            # TODO: check for social distancing
            
            # TODO: vocalize decision
            pass
        if stage == "talk":
            # TODO: "age"               --> 0 to 100
            # TODO: "was_vaccinated"    --> 0 / 1
            # TODO: "doctor"            --> "red" / "green" / "blue" / "yellow"
            # TODO: "hours_exercise"    --> 0 to 40
            # TODO: "right_vaccine"     --> "Greenzer" / "Rederna" / "StellaBluera" / "BlacknikV"
            
            # TODO: vocalize decision
            pass
        if stage == "cylinder":
            # TODO: check if data has already been processed
            # TODO: choose ring
            pass
        if stage == "ring":
            # TODO: vocalize decision
            pass
        if stage == "vaccinate":
            # TODO: vocalize decision
            pass
        if stage == "done":
            # TODO: nothing
            pass
        print(f"processStage({face['id']}):\t{stage}")
        self.nextStage(face)
        # print()

    def nextStage(self, face):
        
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

        print(f"nextStage({face['id']}):\t\t{stageBefore} --> {face['stage']}")
        print()
            
    def print(self):
        pprint(self.faces)
        print()

    def say(self,statement):
        # TODO: SAY IT !!!
        print()
    
    def run(self):
        for indx in range(len(self.faces)):
            print("----------------------------------------------------------------\n")
            for i in range(6):
                self.processStage(indx)


brain = Brains()

print()
brain.print()
brain.run()
