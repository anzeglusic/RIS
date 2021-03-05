#include <ros/ros.h>
#include <exercise2/Typeofmovement.h>

#include <stdio.h>
#include <sstream>

using namespace std;
int main(int argc, char **argv) {
    ros::init(argc, argv, "client_node");
    ros::NodeHandle n;

    ros::ServiceClient client = n.serviceClient<exercise2::Typeofmovement>("service_node");

    exercise2::Typeofmovement srv;
    cout << "Insert the type of movement and the duration of the movement" << endl;
    string movem;
    int dur;
    cin >> movem >> dur;

    srv.request.movement = movem;
    srv.request.duration = dur;

    ros::service::waitForService("service_node", 1000);

    if (client.call(srv))
    {
        ROS_INFO("The service returned: %s", srv.response.movement2.c_str());
    }
    else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }

    return 0;
}