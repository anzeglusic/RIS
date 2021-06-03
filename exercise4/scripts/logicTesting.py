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
        self.positions = {
                "face": [ 
                            {   "averagePostion": np.array([0,0,0]),
                                "averageNormal": np.array([1,0,0]),
                                "digits_index": 1,
                                "has_mask": False,
                                "stage": "warning",
                                "was_vaccinated": 0,
                                "doctor": "red",
                                "hours_exercise": 25,
                                "right_vaccine": "Greenzer"
                            },
                            {   "averagePostion": np.array([0,0.5,0]), "averageNormal": np.array([1,0,0]),
                                "digits_index": 2, "has_mask": True, "id": 2, "stage": "warning",
                                "was_vaccinated": 1, "doctor": "blue", "hours_exercise": 5,
                                "right_vaccine": "Rederna"
                            },
                            {   "averagePostion": np.array([3,0.5,0]), "averageNormal": np.array([1,0,0]),
                                "digits_index": 0, "has_mask": True, "id": 3, "stage": "warning",
                                "was_vaccinated": 0, "doctor": "yellow", "hours_exercise": 13,
                                "right_vaccine": "StellaBluera"
                            }
                     ],
                "digits": [
                            {   "data": 12 },
                            {   "data": 57 },
                            {   "data": 40 }
                ]
        }
        

    def processStage(self, indx):
        try:
            face = self.positions["face"][indx]
        except Exception as err:
            print(f"ProcessStage error: {err}")
            return

        stage = face["stage"]
        print(f"processStage({face['id']}):\t{stage}")

        if stage == "warning":
            # check if it has a mask
            if face["has_mask"] == False:
                self.say("mask bad")
            # check for social distancing
            for i in range(len(self.positions["face"])):
                if i == indx:
                    continue
                else:
                    area_avg = face["averagePostion"]
                    dist_vector = area_avg - self.positions["face"][i]["averagePostion"]
                    dist = np.sqrt(dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2)
                    if dist < 1:
                        self.say("distancing bad")
                        break
            # vocalize decision
            self.say("no more warnings")
            
        if stage == "talk":
            # "age"                 --> 0 to 100
            age = self.positions["digits"][face["digits_index"]]["data"]
            # TODO: "was_vaccinated"    --> 0 / 1
            # TODO: "doctor"            --> "red" / "green" / "blue" / "yellow"
            # TODO: "hours_exercise"    --> 0 to 40
            # TODO: "right_vaccine"     --> "Greenzer" / "Rederna" / "StellaBluera" / "BlacknikV"

            was_vaccinated = face["was_vaccinated"]
            doctor = face["doctor"]
            hours_exercise = face["hours_exercise"]
            right_vaccine = face["right_vaccine"]
            # vocalize decision
            self.say("data collected")

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
        pprint(self.positions)
        print()

    def say(self,statement):
        # TODO: actualy SAY IT !!!
        print(f"--> {statement}")
    
    def run(self):
        for indx in range(len(self.positions["face"])):
            print("----------------------------------------------------------------\n")
            for i in range(5):
                self.processStage(indx)


brain = Brains()

print()
brain.print()
brain.run()
