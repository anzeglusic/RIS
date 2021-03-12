#!/usr/bin/python3

from __future__ import print_function
from homework2.srv import Movement
import rospy
import time

def send_request(movement, secounds):
    pass

def h2_client_node():
    time.sleep(2)
    
    rospy.wait_for_service('h2_service_node')
    call_h2_service_node = rospy.ServiceProxy("h2_service_node", Movement)
    while True:
        print("\n======================== instruct client ========================")
        movement_to_do = input("What movement [circle/rectangle/triangle/random]: ").strip()
        duration = float(input("For how many secounds to run:                     ").strip())
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        response = call_h2_service_node(movement_to_do,duration)
        print(f"Previous movement was \"{response.last_issued_movement}\".")
        print("_________________________________________________________________")
    

if __name__ == "__main__":
    h2_client_node()