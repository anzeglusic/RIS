#include "ros/ros.h"
#include "exercise2/Typeofmovement.h"
#include <geometry_msgs/Twist.h>

#include <string>
#include <iostream>
#include <algorithm>
#include <stdlib.h>

using namespace std;

void recatngle(int dur) {
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1000);
    ros::Rate rate(1);
    int step = 1;
    while (dur--) {
        geometry_msgs::Twist msg;
        
        msg.linear.x = 0.5;
        step = step % 20;
        if (step % 5 == 0) {
            msg.linear.x = 0;
            msg.angular.z = 1.57;
        }
        pub.publish(msg);
        step++;
        rate.sleep();
    }
    return;
}

void random(int dur) {
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1000);
    ros::Rate rate(1);
    int step = 1;
    srand (time(0));
    while (dur--) {
        geometry_msgs::Twist msg;
        
        msg.linear.x = 0.5;
        if (step%2 == 0) {
            msg.linear.x = 0;
        msg.angular.z = rand() % 6 + 1;
        }
        pub.publish(msg);
        step++;
        rate.sleep();
    }
    return;
}   

void triangle(int dur) {
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1000);
    ros::Rate rate(1);
    int step = 1;
    while (dur--) {
        geometry_msgs::Twist msg;
        
        msg.linear.x = 0.5;
        step = step % 9;
        if (step % 3 == 0) {
            msg.linear.x = 0;
            msg.angular.z = 2.09333333333;
        }
        pub.publish(msg);
        step++;
        rate.sleep();
    }
    return;
}

void circle(int dur) {
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1000);
    ros::Rate rate(1);
    int step = 1;
    while (dur--) {
        geometry_msgs::Twist msg;
        msg.linear.x = 0.1;
        if (step % 2 == 0) {
            msg.linear.x = 0;
            msg.angular.z = 0.2;
        }
        pub.publish(msg);
        step++;
        rate.sleep();
    }
    return;
}

bool manipulate(exercise2::Typeofmovement::Request  &req,
         exercise2::Typeofmovement::Response &res)
{
  res.movement2 = req.movement;
    if (req.movement == "triangle") {
        triangle(req.duration);
    }
    else if (req.movement == "random") {
        random(req.duration);
    }
    else if (req.movement == "circle") {
        circle(req.duration);
    }
    else {
        recatngle(req.duration);
    }
    
  ROS_INFO("Response: %s", res.movement2.c_str());
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "service_node");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("service_node", manipulate);
  ros::spin();

  return 0;
}
