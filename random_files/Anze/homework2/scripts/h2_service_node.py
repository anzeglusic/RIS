#!/usr/bin/python3

from __future__ import print_function
from geometry_msgs.msg import Twist
from homework2.srv import Movement
import rospy
import math
import random

def handle_movement(req):
    new_movement = req.what_movement_to_do
    duration = req.how_long
    # print(f"Client instructed to do movement \"{new_movement}\" for {duration} secounds.")
    twist = Twist()

    hz = 4
    r = rospy.Rate(hz)

    step = 1
    final_step = int(duration*hz)
    while step < final_step: # bacouse turtle subscribes 1 per secound
        if new_movement == "circle":
            twist.linear.x = 1
            twist.angular.z = (2.0*math.pi)/8.0
        elif new_movement == "rectangle":
            if (step%10 == 0):
                twist.linear.x = 0.0
                twist.angular.z = (2.0*math.pi)
            else:
                twist.linear.x = 1.0
                twist.angular.z = 0.0
        elif new_movement == "triangle":
            if (step%10 == 0):
                twist.linear.x = 0.0
                twist.angular.z = (2.0*math.pi)/3.0*4.0
            else:
                twist.linear.x = 1.0
                twist.angular.z = 0.0
        elif new_movement == "random":
            twist.linear.x = 1.0 * random.random()
            twist.angular.z = 4.0 * 2 * (random.random()-0.5)
        else:
            # do random
            new_movement = "random"
            twist.linear.x = 1.0 * random.random()
            twist.angular.z = 4.0 * 2 * (random.random()-0.5)

        pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size = 1000)
        pub.publish(twist)

        step += 1
        r.sleep()
    twist.linear.x = 0.0
    twist.angular.z = 0.0

    to_return = rospy.get_param("~last_movement_type")
    rospy.set_param("~last_movement_type",new_movement)
    return to_return

def h2_service_node():
    rospy.init_node("h2_service_node")
    s = rospy.Service("h2_service_node",Movement, handle_movement,)
    rospy.spin()


if __name__ == "__main__":
    h2_service_node()
