#!/usr/bin/python3

import rospy
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

class Arm_Mover():
    def __init__(self):

        rospy.init_node('arm_mover', anonymous=True)
        
        self.arm_movement_pub = rospy.Publisher('/turtlebot_arm/arm_controller/command', JointTrajectory, queue_size=1)
        self.arm_user_command_sub = rospy.Subscriber("/arm_command", String, self.new_user_command)

        # Just for controlling wheter to set the new arm position
        self.user_command = None
        self.send_command = False

        # Pre-defined positions for the arm
        self.retract = JointTrajectory()
        self.retract.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.retract.points = [JointTrajectoryPoint(positions=[0,-1.3,2.2,1],
                                                    time_from_start = rospy.Duration(1))]

        self.extend = JointTrajectory()
        self.extend.joint_names = ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        self.extend.points = [JointTrajectoryPoint(positions=[0.3,1,1,0],
                                                    time_from_start = rospy.Duration(1))]

    def new_user_command(self, data):
        self.user_command = data.data.strip()
        self.send_command = True

    def update_position(self):
        # Only if we had a new command
        if self.send_command:
            if self.user_command == 'retract':
                self.arm_movement_pub.publish(self.retract)
                print('Retracted arm!')
            elif self.user_command == 'extend':
                self.arm_movement_pub.publish(self.extend)
                print('Extended arm!')
            else:
                print('Unknown instruction:', self.user_command)
                return(-1)
            self.send_command = False

if __name__ == "__main__":
    am = Arm_Mover()
    time.sleep(.5)
    am.arm_movement_pub.publish(am.retract)
    print('Retracted arm!')
    
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        am.update_position()
        r.sleep()


#source /home/iletavcioski/ROS/devel/setup.bash
#roslaunch exercise7 rins_world.launch
#roslaunch exercise3 amcl_simulation.launch 
#roslaunch turtlebot_rviz_launchers view_navigation.launch 2>/dev/null
#rostopic echo /camera/rgb/image_raw --noarr
#rosrun exercise4 colorLocalizer.py
#rosrun exercise3 map_goals
#rosrun exercise7 move_arm.py
#rostopic pub /arm_command std_msgs/String "data: "extend""
#Sepravi vektor je od lokacije kjer aprochas in cilindra in potem naredis arcustanges kota do x osi atan2. Js posiljam kr obicajen gole tako da orientation.z =sin(kot/2), orientation.w = cos(kot/2)
