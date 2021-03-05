#include "ros/ros.h"
#include "homework1/Sum.h"

//#include <cstdlib>
#include <sstream>
#include <vector>
#include <time.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "our_client_node");
  ros::NodeHandle n;

  ros::ServiceClient client = n.serviceClient<homework1::Sum>("our_service_node/sum");

  homework1::Sum srv;
  std::stringstream ss;

  std::vector<int64_t> numbers;
  
  // 10 numbers in range [-100 100]
  ss << "[ ";
  // seed is only called set once 
  //    --> therefore if multiple clients are called in same secound they will have the same starting vector
  srand(time(NULL)); 
  for (int i=0; i<10; i++){
    int64_t temp_num = float(std::rand())/float(RAND_MAX)*200-100;
    ss << temp_num << " ";
    numbers.push_back(temp_num);
  }
  ss << "]";


  srv.request.numbers = numbers;

  ros::service::waitForService("our_service_node/sum", 1000);

  ROS_INFO("Sending: %s", ss.str().c_str());

  if (client.call(srv))
  {
    ROS_INFO("The service returned: %ld", srv.response.sum);
  }
  else
  {
    ROS_ERROR("Failed to call service");
    return 1;
  }

  return 0;
}
