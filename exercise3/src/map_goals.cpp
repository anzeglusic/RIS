#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/String.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <cmath>
#include <map>
#include <wait.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

int stevec = 0;

Mat cv_map;
float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
ros::Subscriber map_sub;
ros::Subscriber sub25;
int yy11 = 259;
int xx11 = 255;

string s1 = "-1";
double z[] = {1, 0.7, 0, -0.7};
double w[] = {0, 0.7, 1, 0.7};
map<pair<double, pair<double, int> >,bool> looked;
map<pair<double, pair<double, int> >,bool> went;
map<pair<double, pair<double, int> >,bool> can;
int dirrr=1;


bool look = false;
int mat[1000][1000];
void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg_map) {
    int size_x = msg_map->info.width;
    int size_y = msg_map->info.height;

    if ((size_x < 3) || (size_y < 3) ) {
        ROS_INFO("Map size is only x: %d,  y: %d . Not running map to image conversion", size_x, size_y);
        return;
    }

    // resize cv image if it doesn't have the same dimensions as the map
    if ( (cv_map.rows != size_y) && (cv_map.cols != size_x)) {
        cv_map = cv::Mat(size_y, size_x, CV_8U);
    }

    map_resolution = msg_map->info.resolution;
    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;

    //tf2::poseMsgToTF(msg_map->info.origin, map_transform);

    const std::vector<int8_t>& map_msg_data (msg_map->data);

    unsigned char *cv_map_data = (unsigned char*) cv_map.data;

    //We have to flip around the y axis, y for image starts at the top and y for map at the bottom
    int size_y_rev = size_y-1;
    for (int y = size_y_rev; y >= 0; --y) {
        
        int idx_map_y = size_x * (size_y -y);
        int idx_img_y = size_x * y;

        for (int x = 0; x < size_x; ++x) {

            int idx = idx_img_y + x;
            //cout << y << " " << x << endl;
            switch (map_msg_data[idx_map_y + x])
            {
            case -1:
                cv_map_data[idx] = 127;
                mat[y][x] = 127;
                //cout << 127 << endl;
                break;

            case 0:
                cv_map_data[idx] = 255;
                mat[y][x] = 255;
                //cout << 255 << endl;
                break;

            case 100:
                cv_map_data[idx] = 0;
                mat[y][x] = 0;
                //cout << 0 << endl;
                break;
            }
        }
    }

}
double X = -0.5, Y = 0.5;
void vozi(int y3, int x3, double y, double x, int dir) {
    X = x;
    Y = y;
    dirrr = dir;
    geometry_msgs::PoseStamped goal;
    cout << stevec << endl;     
    goal.header.frame_id = "map";
    goal.pose.orientation.w = w[dir];
    goal.pose.orientation.z = z[dir];
    goal.pose.position.x = x;
    goal.pose.position.y = y;
    goal.header.stamp = ros::Time::now();
    cout << "Moving to:" << y << " " << x << endl;
    stevec++;
    goal_pub.publish(goal);
    if (!look)
    sleep(15);
    else sleep(8);
}

double sy(double y, int dir) {
    if (dir == 1) return y+1;
    else if (dir == 3) return y-1;
    else return y;
}

double sx(double x, int dir) {
    if (dir == 0) return x-1;
    else if(dir == 2) return x+1;
    else return x;
}

void ins() {
    //TO DO implementet
    /*
    can[make_pair(0.5,make_pair(-1.5,0))] = true;
    can[make_pair(0.5,make_pair(-1.5,1))] = true;
    can[make_pair(0.5,make_pair(-1.5,3))] = true;
    can[make_pair(1.5,make_pair(-0.5,0))] = true;
    can[make_pair(1.5,make_pair(-0.5,2))] = true;
    can[make_pair(2.5,make_pair(-1.5,0))] = true;
    can[make_pair(2.5,make_pair(-1.5,1))] = true;
    can[make_pair(2.5,make_pair(-1.5,3))] = true;
    can[make_pair(2.5,make_pair(-0.5,1))] = true;
    can[make_pair(2.5,make_pair(0.5,1))] = true;
    can[make_pair(2.5,make_pair(0.5,3))] = true;
    can[make_pair(2.5,make_pair(1.5,1))] = true;
    can[make_pair(2.5,make_pair(2.5,1))] = true;
    can[make_pair(2.5,make_pair(2.5,2))] = true;
    can[make_pair(1.5,make_pair(2.5,2))] = true;
    can[make_pair(1.5,make_pair(1.5,0))] = true;
    can[make_pair(0.5,make_pair(1.5,0))] = true;
    can[make_pair(0.5,make_pair(1.5,3))] = true;
    can[make_pair(0.5,make_pair(2.5,2))] = true;
    can[make_pair(-0.5,make_pair(3.5,1))] = true;
    can[make_pair(-0.5,make_pair(3.5,2))] = true;
    can[make_pair(-0.5,make_pair(3.5,3))] = true;
    can[make_pair(-0.5,make_pair(2.5,3))] = true;
    can[make_pair(-0.5,make_pair(1.5,1))] = true;
    can[make_pair(-0.5,make_pair(1.5,3))] = true;
    can[make_pair(-0.5,make_pair(0.5,0))] = true;
    can[make_pair(-0.5,make_pair(0.5,1))] = true;
    can[make_pair(-0.5,make_pair(0.5,3))] = true;
    */
   can[make_pair(2.5,make_pair(1.5, 3))] = false;
   can[make_pair(1.5,make_pair(1.5, 1))] = false;
   can[make_pair(0.5,make_pair(1.5, 1))] = false;

}


double y2 = -19999;
double x2 = -19999;

double euc(double x1, double y1, double x2, double y2) {
    double xxx = x1 - x2; //calculating number to square in next step
	double yyy = y1 - y2;
	double dist;

	dist = pow(xxx, 2) + pow(yyy, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}
bool prva = false;
int bro = 0;
void tapa(const geometry_msgs::Twist::ConstPtr &mg) {
    if (bro%2) {
        bro++;
        return;
    }
    double yyy, xxx;
    y2 = mg->angular.y;
    x2 = mg->angular.x;
    yyy = mg->linear.y;
    xxx = mg->linear.x;
    cout << "DOJDEDVA" << " " << yyy << " " << xxx << endl;
    if (isnan(x2) || isnan(y2) || isnan(xxx) || isnan(yyy)) return;
    if (y2 - yyy > 0 && (y2 - yyy) > abs(x2-xxx)) {
        if (!went[make_pair(yyy, make_pair(xxx, 1))]) {
            vozi(yy11, xx11, yyy, xxx,1);
            went[make_pair(yyy, make_pair(xxx, 1))] = true;
        }
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        //system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(10);
    }
    else if (yyy - y2 > 0 && (yyy - y2) > abs(x2-xxx)) {
        if (!went[make_pair(yyy, make_pair(xxx, 3))]) {
            vozi(yy11, xx11, yyy, xxx,3);
            went[make_pair(yyy, make_pair(xxx, 3))] = true;
        }
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        //system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(30);
    }
    else if (x2 - xxx > 0 && (x2 - xxx) > abs(y2-yyy)) {
        if (!went[make_pair(yyy, make_pair(xxx, 2))]) {
            vozi(yy11, xx11, yyy, xxx,2);
            went[make_pair(yyy, make_pair(xxx, 2))] = true;
        }
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        //system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(30);
    }
    else if (xxx - x2 > 0 && (xxx - x2) > abs(y2-yyy)) {
        if (!went[make_pair(yyy, make_pair(xxx, 1))]) {
            vozi(yy11, xx11, yyy, xxx,1);
            went[make_pair(yyy, make_pair(xxx, 1))] = true;
        }
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        //system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(30);
    }
    return;
}

int syy(int my, int dir) {
    if (dir == 1) return my-20;
    else if (dir == 3) return my+20;
    else return my;
}

int sxx(int mx, int dir) {
    if (dir == 0) return mx-20;
    else if(dir == 2) return mx+20;
    else return mx;
}

bool preveri(int my, int mx, int dir) {
    bool cc = false;
    if (dir == 1) {
        for (int i = my - 1; i >= my-25; i--) {
            if (mat[i][mx] != 255) {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 0) {
        for (int i = mx - 1; i >= mx-25; i--) {
            if (mat[my][i] != 255) {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 2) {
        for (int i = mx + 1; i <= mx+25; i++) {
            if (mat[my][i] != 255) {
                cc = true;
                break;
            }
        }
    }
    else {
        for (int i = my+1; i <= my+25; i++) {
            if (mat[i][mx] != 255) {
                cc = true;
                break;
            }
        }
    }
    return cc;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "map_goals");
    ros::NodeHandle n;
    map_sub = n.subscribe("map", 10, &mapCallback);
    goal_pub = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    namedWindow("Map");
    bool cc = true;
    ros::Subscriber sub24 = n.subscribe("/our_pub1/chat1", 100, tapa);
    double y = 0.5;
    double x = -0.5;
    yy11 = 259;
    xx11 = 235;
    int mom = 1;
    
    while(ros::ok()) {
        ros::Subscriber sub24 = n.subscribe("/our_pub1/chat1", 100, tapa);
        ros::spinOnce();
        if (!cv_map.empty()) imshow("Map", cv_map);
       
        if(cc) {
            cc=false;
            cout << "vleze2" << endl;
            s1="Navigation";
            vozi(yy11, xx11,y, x, mom);
            s1="sd";
            waitKey(30);
        }
        cout << y << " " << x << " " << mom << endl;
        if (stevec > 54) return 0;
        int le = (mom + 3) % 4;
        int de = (mom + 1) % 4;
        int naz = (mom + 2) % 4;
        int nap = mom;  
        cout << yy11 << " " << xx11 << endl;
        cout << "le = " << le << " "<< preveri(yy11, xx11, le) << endl;
        cout << "de = " << de << " " << preveri(yy11, xx11, de) << endl;
        cout << "naz = " << naz << " " << preveri(yy11, xx11, naz) << endl;
        cout << "nap = " << nap << " "<<preveri(yy11, xx11, nap) << endl;

        can[make_pair(y, make_pair(x, le))] = preveri(yy11, xx11, le);
        can[make_pair(y, make_pair(x, de))] = preveri(yy11, xx11, de);
        can[make_pair(y, make_pair(x, naz))] = preveri(yy11, xx11, naz);
        can[make_pair(y, make_pair(x, nap))] = preveri(yy11, xx11, nap);
        ins();
        look = true;
        look = true;
        if (!looked[make_pair(y, make_pair(x, le))] && can[make_pair(y, make_pair(x, le))]) looked[make_pair(y, make_pair(x, le))] = true, vozi(yy11, xx11, y, x, le);
        look = false;
        look = true;
        if (!looked[make_pair(y, make_pair(x, de))] && can[make_pair(y, make_pair(x, de))]) looked[make_pair(y, make_pair(x, de))] = true, vozi(yy11, xx11, y, x, de);
        look = false;
        s1="Navigation";
        if (!went[make_pair(y, make_pair(x, le))] && !can[make_pair(y, make_pair(x, le))]) {
            went[make_pair(y, make_pair(x, le))] = true;
            vozi(syy(yy11, le), sxx(xx11,le), sy(y, le), sx(x, le), le);
            mom = le;
            y=sy(y, mom);
            x=sx(x, mom);
            yy11=syy(yy11, mom);
            xx11=sxx(xx11, mom);
        }
        else if (!went[make_pair(y, make_pair(x, nap))] && !can[make_pair(y, make_pair(x, nap))]) {
            went[make_pair(y, make_pair(x, nap))] = true;
            vozi(syy(yy11,nap), sxx(xx11,nap), sy(y, nap), sx(x, nap), nap);
            mom = nap;
            y=sy(y, mom);
            x=sx(x, mom);
            yy11=syy(yy11, mom);
            xx11=sxx(xx11, mom);
        }
        else if (!went[make_pair(y, make_pair(x, de))] && !can[make_pair(y, make_pair(x, de))]) {
            went[make_pair(y, make_pair(x, de))] = true;
            vozi(syy(yy11,de), sxx(xx11,de), sy(y, de), sx(x, de), de);
            mom = de;
            y=sy(y, mom);
            x=sx(x, mom);
            yy11=syy(yy11, mom);
            xx11=sxx(xx11, mom);
        }
        else if (!went[make_pair(y, make_pair(x, naz))] && !can[make_pair(y, make_pair(x, naz))]) {
            went[make_pair(y, make_pair(x, naz))] = true;
            vozi(syy(yy11,naz), sxx(xx11,naz), sy(y, naz), sx(x, naz), naz);
            mom = naz;
            y=sy(y, mom);
            x=sx(x, mom);
            yy11=syy(yy11, mom);
            xx11=sxx(xx11, mom);
        }
        cout << mom << endl;
        waitKey(30);
    }
    return 0;
}
