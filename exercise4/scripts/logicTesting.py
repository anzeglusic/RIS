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


class Brains:
    def __init__(self):
        self.faces = [
                        {"id": 1, "stage" : "warning" }
                     ]

    def processStage(self, face):
        stage = face["stage"]
        if stage == "warning":
            # TODO: check if it has a mask
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
        # print()


brain = Brains()

print()
brain.print()
for i in range(6):
    brain.processStage(brain.faces[0])
