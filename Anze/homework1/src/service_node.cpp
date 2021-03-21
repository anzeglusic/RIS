#include "ros/ros.h"
#include "homework1/Sum.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

bool manipulate(homework1::Sum::Request &req,
                homework1::Sum::Response &res)
{
  
  std::vector<int64_t> numbers = req.numbers;
  
  std::stringstream ss;

  int64_t sum = 0;
  ss << "[ ";
  for (int64_t temp_num : numbers){
    sum += temp_num;
    ss << temp_num << " ";
  }
  ss << "]";
  
  res.sum = sum;
  
  ROS_INFO("request: sum(%s), response: %ld", ss.str().c_str(), res.sum);

  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "our_service_node");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("our_service_node/sum", manipulate);
  ROS_INFO("I am ready to sum up your numbers!");
  ros::spin();

  return 0;
}
