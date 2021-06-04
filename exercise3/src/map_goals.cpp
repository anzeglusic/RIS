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
#include <actionlib/client/terminal_state.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <wait.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;
bool posiljaj = false;
int stevec = 0;
bool retract = false;
bool streched = false;
Mat cv_map;
float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
ros::Subscriber map_sub;
ros::Subscriber sub25;
ros::Publisher roko;
ros::Publisher semNekaj;
int yy11 = 259;
int xx11 = 255;
double paty[] = {0.5, 1.5, 2.5, 2.5, 2.5, 2.5, 1.5, 0.5, -0.5, -0.5, -0.5};
double patx[] = {-0.5, -0.5, -0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 2.5, 1.5, 0.5};

//proba
string s1 = "-1";
double z[] = {1, 0.7, 0, -0.7, 0.925, 0.3797, -0.3797, -0.925};
double w[] = {0, 0.7, 1, 0.7, 0.3797, 0.925, 0.925, 0.3797};
double momY = 0.5;
double momX = -0.5;
map<pair<double, pair<double, int>>, bool> looked;
map<pair<double, pair<double, int>>, bool> went;
map<pair<double, pair<double, int>>, bool> can;
int dirrr = 1;

bool mozi = false;
bool look = false;
int mat[1000][1000];
int ringe = 0;
int arr[] = {2, 0, 3, 0};
double ary[] = {0.1, 0, 0.25, 0};
double arx[] = {-0.25, 0.25, 0, 0.25};
//proba2
void mapCallback(const nav_msgs::OccupancyGridConstPtr &msg_map)
{
    int size_x = msg_map->info.width;
    int size_y = msg_map->info.height;

    if ((size_x < 3) || (size_y < 3))
    {
        ROS_INFO("Map size is only x: %d,  y: %d . Not running map to image conversion", size_x, size_y);
        return;
    }

    // resize cv image if it doesn't have the same dimensions as the map
    if ((cv_map.rows != size_y) && (cv_map.cols != size_x))
    {
        cv_map = cv::Mat(size_y, size_x, CV_8U);
    }

    map_resolution = msg_map->info.resolution;
    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;

    //tf2::poseMsgToTF(msg_map->info.origin, map_transform);

    const std::vector<int8_t> &map_msg_data(msg_map->data);

    unsigned char *cv_map_data = (unsigned char *)cv_map.data;

    //We have to flip around the y axis, y for image starts at the top and y for map at the bottom
    int size_y_rev = size_y - 1;
    for (int y = size_y_rev; y >= 0; --y)
    {

        int idx_map_y = size_x * (size_y - y);
        int idx_img_y = size_x * y;

        for (int x = 0; x < size_x; ++x)
        {

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
bool konec = false;
void vozi(int y3, int x3, double y, double x, int dir)
{
    if (y == -0.5 && x == 0.5)
    {
        x = 0.3;
        y = -0.75;
    }
    if (y == -0.5 && x == 1.5)
    {
        y = -0.75;
    }
    if (y == 1.5 && x == 1.5)
    {
        y = 1.25;
        x = 1.25;
    }
    if (y == 0.5 && x == 1.5)
    {
        //y = 1.25;
        x = 1.25;
    }
    if (y == 2.5 && x == 1.5)
    {
        y = 2.6;
    }

    mozi = false;
    X = x;
    Y = y;
    dirrr = dir;
    geometry_msgs::PoseStamped goal;
    cout << stevec << endl;
    /* if (stevec % 5 == 0)
    {
        posiljaj = true;
        if (retract == false)
            streched = false, retract = true;
        else
            streched = true, retract = false;
        /// brojcccc = 0;
    }
    */
    goal.header.frame_id = "map";
    goal.pose.orientation.w = w[dir];
    goal.pose.orientation.z = z[dir];
    goal.pose.position.x = x;
    goal.pose.position.y = y;
    goal.header.stamp = ros::Time::now();
    cout << "Moving to:" << y << " " << x << endl;
    stevec++;
    goal_pub.publish(goal);
    if (stevec == 0)
        sleep(5);
    if (look && y == 1.25 && x == 1.25)
        sleep(15);
    else if (look)
        sleep(10);

    if (y == 0.5 && x == -0.5 && dir == 3)
        konec = true;
    return;
}

double sy(double y, int dir)
{
    if (dir == 1)
        return y + 1;
    else if (dir == 3)
        return y - 1;
    else
        return y;
}

double sx(double x, int dir)
{
    if (dir == 0)
        return x - 1;
    else if (dir == 2)
        return x + 1;
    else
        return x;
}

double y2 = -19999;
double x2 = -19999;

double euc(double x1, double y1, double x2, double y2)
{
    double xxx = x1 - x2; //calculating number to square in next step
    double yyy = y1 - y2;
    double dist;

    dist = pow(xxx, 2) + pow(yyy, 2); //calculating Euclidean distance
    dist = sqrt(dist);

    return dist;
}
bool prva = false;
int bro = 0;
int konc = 0;
/*
void tapa(const geometry_msgs::Twist::ConstPtr &mg) {
    //if (bro%2) {
     //   bro++;
      //  return;
   // }
    double yyy, xxx;
    y2 = mg->angular.y;
    x2 = mg->angular.x;
    yyy = mg->linear.y;
    xxx = mg->linear.x;
    cout << "DOJDEDVA" << " " << yyy << " " << xxx << endl;
    if (isnan(x2) || isnan(y2) || isnan(xxx) || isnan(yyy)) return;
    double Y1 = (y2 - yyy);
    int YY = Y1*10;
    double X1 = (x2 - xxx);
    int XX = X1*10;
    if (YY > 0 && YY >= 4 && YY <= 6) {
        vozi(yy11, xx11, y2-0.5, x2,1);
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(20);
    }
    else if (YY < 0 && YY <= -4 && YY >= -6) {
        vozi(yy11, xx11, y2+0.5, x2,3);
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(20);
    }
    else if (XX > 0 && XX >= 4 && XX <= 6) {
        vozi(yy11, xx11, y2, x2-0.5,2);    
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(20);
    }
    else if (XX <  0 && XX >= -6 && XX <= -4){
            vozi(yy11, xx11, y2, x2+0.5,0);
        cout << "POZDRAAAAAAAAAAAAAAV" << endl;
        system("/home/iletavcioski/ROS/src/exercise3/src/pozdrav.sh");
        sleep(20);
    }
    return;
}
*/

bool preveri15(int my, int mx, int dir)
{
    bool cc = false;
    if (dir == 1)
    {
        for (int i = my - 1; i >= my - 10; i--)
        {
            if (mat[i][mx] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 0)
    {
        for (int i = mx - 1; i >= mx - 10; i--)
        {
            if (mat[my][i] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 2)
    {
        for (int i = mx + 1; i <= mx + 10; i++)
        {
            if (mat[my][i] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else
    {
        for (int i = my + 1; i <= my + 10; i++)
        {
            if (mat[i][mx] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    return cc;
}
int cil = 0;
int arc[] = {0, 1, 2, 0};
int brci = 0;
int brrin = 0;

int syy(int my, int dir)
{
    if (dir == 1)
        return my - 20;
    else if (dir == 3)
        return my + 20;
    else
        return my;
}

int sxx(int mx, int dir)
{
    if (dir == 0)
        return mx - 20;
    else if (dir == 2)
        return mx + 20;
    else
        return mx;
}

bool preveri(int my, int mx, int dir)
{
    bool cc = false;
    if (dir == 1)
    {
        for (int i = my - 1; i >= my - 25; i--)
        {
            if (mat[i][mx] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 0)
    {
        for (int i = mx - 1; i >= mx - 25; i--)
        {
            if (mat[my][i] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else if (dir == 2)
    {
        for (int i = mx + 1; i <= mx + 25; i++)
        {
            if (mat[my][i] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    else
    {
        for (int i = my + 1; i <= my + 25; i++)
        {
            if (mat[i][mx] != 255)
            {
                cc = true;
                break;
            }
        }
    }
    return cc;
}

void statusCallback(const actionlib_msgs::GoalStatusArray::ConstPtr &msg)
{
    //cout << "VLEZEEE" << endl;
    //cout << mozi << endl;
    if (msg->status_list.size() == 0)
        return;
    int nn = msg->status_list.size();
    int br = msg->status_list[nn - 1].status;
    //cout << br << endl;
    if (br == 3)
        mozi = true;
    else
        mozi = false;
    //cout << mozi << endl;
    return;
}
int brojcccc = 0;
map<pair<double, pair<double, int>>, bool> can1;
map<pair<double, pair<double, int>>, bool> went1;

queue<pair<double, pair<double, int>>> qi;
void ins()
{
    can[make_pair(2.5, make_pair(1.5, 3))] = false;
    can[make_pair(1.5, make_pair(1.5, 1))] = false;
    can[make_pair(0.5, make_pair(1.5, 1))] = false;
    can[make_pair(0.5, make_pair(2.5, 2))] = true;
    can1[make_pair(2.5, make_pair(1.5, 3))] = false;
    can1[make_pair(1.5, make_pair(1.5, 1))] = false;
    can1[make_pair(0.5, make_pair(1.5, 1))] = false;
    can1[make_pair(0.5, make_pair(2.5, 2))] = true;
}
queue<pair<double, pair<double, int>>> qI;
void approaching(double zY, double zX, int kaj)
{
    cout << "APPROACHING-> " << zY << " " << zX << endl;
    pair<double, double> nearest;
    double dist = 1e9;
    for (map<pair<double, pair<double, int>>, bool>::iterator it = can1.begin(); it != can1.end(); it++)
    {
        pair<double, pair<double, int>> pp = it->first;
        cout << pp.second.first << " " << pp.first << " " << euc(zX, zY, pp.second.first, pp.first) << endl;
        if (pp.first == -0.5 && pp.second.first == 1.5)
        {
            pp.first = -0.75;
            pp.second.first = 1.3;
        }
        if (euc(zX, zY, pp.second.first, pp.first) < dist)
        {
            if (zY > 1 && zY < 2 && zX > 0 && zX < 1 && pp.first == 1.5 && pp.second.first == 1.5)
                continue;

            dist = euc(zX, zY, pp.second.first, pp.first);

            nearest = make_pair(pp.first, pp.second.first);
        }
    }
    cout << nearest.first << " " << nearest.second << endl;
    vector<pair<double, double>> opcii;
    opcii.push_back(make_pair(nearest.first, nearest.second - 0.5));
    opcii.push_back(make_pair(nearest.first + 0.5, nearest.second));
    opcii.push_back(make_pair(nearest.first, nearest.second + 0.5));
    opcii.push_back(make_pair(nearest.first - 0.5, nearest.second));
    if (kaj == 1 || kaj == 3 || kaj == 4)
    {
        double maxi_dist = 1e9;
        int direction;
        for (int i = 0; i < 4; i++)
        {
            if (nearest.first > 0 && nearest.first < 1 && nearest.second > (-2) && nearest.second < (-1) && i == 3)
            {
                cout << "ZABRANA" << endl;
                continue;
            }

            cout << opcii[i].first << " " << opcii[i].second << endl;
            cout << euc(zX, zY, opcii[i].second, opcii[i].first) << endl;
            if (euc(zX, zY, opcii[i].second, opcii[i].first) < maxi_dist)
            {
                direction = i;
                maxi_dist = euc(zX, zY, opcii[i].second, opcii[i].first);
            }
        }
        cout << direction << endl;
        if (kaj == 3 && direction == 2)
            direction = 0;
        //if (kaj == 1 && direction == 0 && zX >= 0 && zX <= 2 && zY >= (-1) && zY <= 1)
        //  direction = 2;
        if (kaj == 1 && zY >= -1 && zY <= 1 && zY >= -2 && zX <= (-1) ) 
            direction = 0;
        if (zY > 1 && zY < 2 && zX > 0 && zX < 1 && kaj == 1)
        {
            cout << "ZABRANA" << endl;
            vozi(0, 0, zY - 0.2, zX - 0.2, 5);
            sleep(20);
            cout << "STASAV" << endl;
        }
        else if (direction == 0)
        {
            vozi(0, 0, zY, zX + 0.3, 0);
            sleep(20);
            int bezveze = 0;
            if (kaj == 3)
            {
                vozi(0, 0, zY + 0.2, zX + 0.3, 0);
                sleep(10);
            }

            cout << "STASAV" << endl;
            //sleep(20);
        }
        else if (direction == 1)
        {
            vozi(0, 0, zY - 0.3, zX - 0.15, 1);
            sleep(20);
            if (kaj == 3)
            {
                vozi(0, 0, zY - 0.3, zX + 0.2, 1);
                sleep(10);
            }
            cout << "STASAV" << endl;
            //sleep(20);
        }
        else if (direction == 2)
        {
            vozi(0, 0, zY, zX - 0.3, 2);
            sleep(20);
            if (kaj == 3)
            {
                vozi(0, 0, zY - 0.2, zX - 0.3, 2);
                sleep(10);
            }
            cout << "STASAV" << endl;
            // sleep(20);
        }
        else if (direction == 3)
        {
            vozi(0, 0, zY + 0.3, zX, 3);
            sleep(20);
            if (kaj == 3)
            {
                vozi(0, 0, zY + 0.3, zX - 0.2, 3);
                sleep(10);
            }
            cout << "STASAV" << endl;
            // sleep(20);
        }
    }
    else
    {
        double maxi_dist = 1e9;
        int direction;
        for (int i = 0; i < 4; i++)
        {
            if (i % 2 == 0)
                continue;
            if (euc(zX, zY, opcii[i].second, opcii[i].first) < maxi_dist)
            {
                direction = i;
                maxi_dist = euc(zX, zY, opcii[i].second, opcii[i].first);
            }
        }
        cout << direction << endl;
        if (direction == 1)
        {
            cout << opcii[1].second << endl;
            if (!can1[make_pair(nearest.first, make_pair(nearest.second, direction))])
            {
                if (zX <= opcii[1].second)
                {
                    vozi(0, 0, zY - 0.3, zX - 0.3, 5);
                    cout << "STASAV" << endl;
                    // sleep(20);
                }
                else
                {
                    vozi(0, 0, zY - 0.3, zX + 0.3, 4);
                    cout << "STASAV" << endl;
                    // sleep(20);
                }
            }
            else
            {
                cout << "VLEZERING" << endl;
                if (zX >= opcii[1].second)
                {
                    vozi(0, 0, zY - 0.3, zX + 0.3, 4);
                    cout << "STASAV" << endl;
                    // sleep(20);
                }
                else
                {
                    vozi(0, 0, zY - 0.3, zX - 0.3, 5);
                    cout << "STASAV" << endl;
                    // sleep(20);
                }
            }
        }
        else
        {
            cout << opcii[3].second << endl;
            if (zX <= opcii[3].second)
            {
                vozi(0, 0, zY + 0.3, zX + 0.3, 7);
                cout << "STASAV" << endl;
                sleep(20);
            }
            else
            {
                vozi(0, 0, zY + 0.3, zX - 0.3, 6);
                cout << "STASAV" << endl;
                sleep(20);
            }
        }
    }
}
int krat = 0;
int isApproaching = 0;
int endSearching;
void prideCel(double y, double x, int kind)
{
    int koje1 = -1;
    int koje2 = -1;
    double dist_maxi = 1e9;
    for (int i = 0; i < 11; i++)
    {
        if (euc(momX, momY, patx[i], paty[i]) < dist_maxi)
        {
            dist_maxi = euc(momX, momY, patx[i], paty[i]);
            koje1 = i;
        }
    }
    if (momY >= -1 && momY <= 1 && momX >= 0 && momX <= 2 && kind == 1)
        koje1 =  6;
    if (momY >= 1 && momY <= 2 && momX >= 0 && momX <= 1 && kind == 4)
        koje1 =  6;
    dist_maxi = 1e9;
    for (int i = 0; i < 11; i++)
    {
        if (euc(x, y, patx[i], paty[i]) < dist_maxi)
        {
            dist_maxi = euc(x, y, patx[i], paty[i]);
            koje2 = i;
        }
    }
    if (y >= -1 && y <= 1 && x >= 0 && x <= 2 && kind == 1)
        koje1 =  6;
    if (y >= 1 && y <= 2 && x >= 0 && x <= 1 && kind == 4)
        koje1 =  6;
    int stevec1 = 0;
    int stevec2 = 0;
    int i = koje1;
    while (i != koje2)
    {
        stevec1++;
        i++;
        if (i == 11)
            i = 0;
    }
    cout << "POMINA1" << endl;
    i = koje1;
    while (i != koje2)
    {
        stevec2++;
        i--;
        if (i == -1)
            i = 10;
    }
    cout << koje1 << " " << koje2 << endl;
    if (stevec1 <= stevec2)
    {
        i = koje1;
        while (1)
        {
            qI.push(make_pair(paty[i], make_pair(patx[i], 0)));
            i++;
            if (i == 11)
                i = 0;
            if (i == koje2)
            {
                qI.push(make_pair(paty[i], make_pair(patx[i], 0)));
                break;
            }
        }
    }
    else
    {
        i = koje1;
        while (1)
        {
            qI.push(make_pair(paty[i], make_pair(patx[i], 0)));
            i--;
            if (i == -1)
                i = 10;
            if (i == koje2)
            {
                qI.push(make_pair(paty[i], make_pair(patx[i], 0)));
                break;
            }
        }
    }
    cout << "POMINA2" << endl;
    momY = y;
    momX = x;
    qI.push(make_pair(y, make_pair(x, kind)));
}
void getObrazCall(const geometry_msgs::Twist::ConstPtr &mg)
{
    // isApproaching = 1;
    double aY = mg->angular.y;
    double aX = mg->angular.x;
    cout << "DOJDEOBRAZ-> " << aY << " " << aX << endl;
    if (endSearching == 1)
    {
        prideCel(aY, aX, 4);
        return;
    }
    else
    {
        qi.push(make_pair(aY, make_pair(aX, 3)));
    }
    //approaching(aY, aX, 3);
    //sleep(5);
}

void getCilinderCall(const geometry_msgs::Point::ConstPtr &mg)
{

    //isApproaching = 1;
    double aY = mg->y;
    double aX = mg->x;
    if (endSearching == 1)
    {
        cout << "DOJDECILINDER-> " << aY << " " << aX << endl;
        prideCel(aY, aX, 1);
        return;
    }
    else
    {
        cout << "DOJDECILINDER-> " << aY << " " << aX << endl;
        qi.push(make_pair(aY, make_pair(aX, 1)));
    }
    //approaching(aY, aX, 1);
    //sleep(5);
}

void getRingCall(const geometry_msgs::Point::ConstPtr &mg)
{
    double aY = mg->y;
    double aX = mg->x;
    cout << "DOJDERING-> " << aY << " " << aX << endl;
    prideCel(aY, aX, 2);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_goals");
    ros::NodeHandle n;
    map_sub = n.subscribe("map", 10, &mapCallback);
    mozi = true;
    goal_pub = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    ros::Subscriber move_base_sub = n.subscribe("move_base/status", 1000, statusCallback);
    ros::Subscriber getCilinder = n.subscribe("cylinder_pt", 1000, getCilinderCall);
    ros::Subscriber getObraz = n.subscribe("face_tw", 1000, getObrazCall);
    ros::Subscriber getRing = n.subscribe("ring_pt", 1000, getRingCall);
    roko = n.advertise<std_msgs::String>("/arm_command", 1000);
    semNekaj = n.advertise<std_msgs::String>("/sem_nekaj", 1000);
    namedWindow("Map");
    //ros::Subscriber sub24 = n.subscribe("/our_pub1/chat1", 100, tapa);
    std_msgs::String mgg;
    mgg.data = "retract";
    roko.publish(mgg);
    double y = 0.5;
    double x = -0.5;
    yy11 = 259;
    xx11 = 235;
    int mom = 1;
    bool cc = true;

    int Kaj = 1;
    int krat1 = 0;
    int krat2 = 0;
    int krat3 = 0;
    int krat4 = 0;
    int krat5 = 0;
    endSearching = 0;
    momY = 0.5;
    momX = -0.5;
    //prideCel(-0.44, 4.06, 1);
    while (ros::ok())
    {
        ros::spinOnce();
        if (!cv_map.empty())
            imshow("Map", cv_map);
        if (endSearching == 0)
        {
            Kaj = 1;
            if (konec)
            {
                std_msgs::String mggg;
                mggg.data = "koncal";
                semNekaj.publish(mggg);
                cout << "THE END OF SEARCHING" << endl;
                endSearching = 1;
                mozi = true;
                momY = 0.5;
                momX = -0.5;
                continue;
            }
            ros::spinOnce();
            //brojcccc++;
            // if (!mozi && stevec > 1)
            // continue;
            //ros::spinOnce();
            //ros::Subscriber sub24 = n.subscribe("/our_pub1/chat1", 100, tapa);
            if (!cv_map.empty())
                imshow("Map", cv_map);
            //cout << "NOVO" << endl;
            double zy = 0.5;
            double zx = -0.5;

            int zy1 = 259;
            int zx1 = 235;
            int mom1 = 1;
            while (krat == 0)
            {
                cout << "VLEZE V KRAT" << endl;
                //cout << zy << " " << zx << " " << mom1 << endl;
                int le1 = (mom1 + 3) % 4;
                int de1 = (mom1 + 1) % 4;
                int naz1 = (mom1 + 2) % 4;
                int nap1 = mom1;

                can1[make_pair(zy, make_pair(zx, le1))] = preveri(zy1, zx1, le1);
                can1[make_pair(zy, make_pair(zx, de1))] = preveri(zy1, zx1, de1);
                can1[make_pair(zy, make_pair(zx, naz1))] = preveri(zy1, zx1, naz1);
                can1[make_pair(zy, make_pair(zx, nap1))] = preveri(zy1, zx1, nap1);
                ins();
                if (!can1[make_pair(zy, make_pair(zx, le1))] && !went1[make_pair(zy, make_pair(zx, le1))])
                {
                    went1[make_pair(zy, make_pair(zx, le1))] = true;
                    mom1 = le1;
                    zy = sy(zy, mom1);
                    zx = sx(zx, mom1);
                    zy1 = syy(zy1, mom1);
                    zx1 = sxx(zx1, mom1);
                }
                else if (!can1[make_pair(zy, make_pair(zx, nap1))] && !went1[make_pair(zy, make_pair(zx, nap1))])
                {
                    went1[make_pair(zy, make_pair(zx, nap1))] = true;
                    mom1 = nap1;
                    zy = sy(zy, mom1);
                    zx = sx(zx, mom1);
                    zy1 = syy(zy1, mom1);
                    zx1 = sxx(zx1, mom1);
                }
                else if (!can1[make_pair(zy, make_pair(zx, de1))] && !went1[make_pair(zy, make_pair(zx, de1))])
                {
                    went1[make_pair(zy, make_pair(zx, de1))] = true;
                    mom1 = de1;
                    zy = sy(zy, mom1);
                    zx = sx(zx, mom1);
                    zy1 = syy(zy1, mom1);
                    zx1 = sxx(zx1, mom1);
                }
                else if (!can1[make_pair(zy, make_pair(zx, naz1))] && !went1[make_pair(zy, make_pair(zx, naz1))])
                {
                    went1[make_pair(zy, make_pair(zx, naz1))] = true;
                    mom1 = naz1;
                    zy = sy(zy, mom1);
                    zx = sx(zx, mom1);
                    zy1 = syy(zy1, mom1);
                    zx1 = sxx(zx1, mom1);
                }
                else
                {
                    krat++;
                    break;
                }
            }
            //return 0;

            /*if (krat1 < 3 && y == 0.5 && x == -0.5)
        {
            isApproaching = 1;
            krat1++;
            if (krat1 == 1)
            {
                approaching(0.3, -1.79, 1);
                continue;
            }
            else if (krat1 == 2)
            {
                approaching(1.35, 0.29, 1);
                continue;
            }
            else
            {
                approaching(1.01, -1.2, 2);
                Kaj = 2;
                continue;
            }
        }
        if (krat4 == 0 && y == 2.5 && x == 1.5)
        {
            krat4++;
            Kaj = 2;
            approaching(2.00, 1.45, 2);
            isApproaching = 1;
            continue;
        }
        if (krat2 < 2 && y == 1.5 && x == 2.5)
        {
            isApproaching = 1;
            krat2++;
            if (krat2 == 1)
            {
                approaching(1.28, 2.94, 1);
                continue;
            }
            else
            {
                approaching(2.00, 2.00, 2);
                continue;
            }
            Kaj = 2;
        }
        if (krat5 == 0 && y == 0.5 && x == 2.5)
        {
            krat5++;
            Kaj = 2;
            approaching(-0.87, 2.89, 2);
            isApproaching = 1;
            continue;
        }
        if (krat3 == 0 && y == -0.5 && x == 2.5)
        {
            krat3++;
            approaching(-0.44, 4.06, 1);
            isApproaching = 1;
            continue;
        }
        */

            //approaching(1.32, 0.39, 1);
            //approaching(1.28, 2.94, 1);
            //approaching(-0.44, 4.06, 1);

            if (cc)
            {
                cc = false;
                cout << "vleze2" << endl;
                s1 = "Navigation";
                vozi(yy11, xx11, y, x, mom);
                //ros::Rate r(10); // 10 hz
                cout << "ZAVRSENO" << endl;
                s1 = "sd";
                waitKey(30);
                continue;
            }
            if (!qi.empty() && isApproaching == 0)
            {
                isApproaching = 1;
                approaching(qi.front().first, qi.front().second.first, qi.front().second.second);
                Kaj = qi.front().second.second;
                qi.pop();
                cout << "VLEZEAPP" << endl;

                if (Kaj == 4 || Kaj == 2)
                {
                    Kaj = 1;
                    std_msgs::String mgg1;
                    mgg1.data = "extend";
                    roko.publish(mgg1);
                    sleep(5);
                    std_msgs::String mgg;
                    mgg.data = "retract";
                    roko.publish(mgg);
                    sleep(5);
                }
                //sleep(5);
                std_msgs::String mggg;
                mggg.data = "pridel";
                semNekaj.publish(mggg);
                sleep(3);
                // sleep(5);

                continue;
            }
            if (mozi && stevec > 1)
            {
                cout << "VLEZE v MOZI" << endl;
                isApproaching = 0;
            }
            if ((!mozi && stevec > 1) || isApproaching == 1)
            {
                //cout << "VLEZE v !MOZI" << endl;
                continue;
            }

            cout << y << " " << x << " " << mom << endl;
            if (stevec > 100)
                return 0;
            int le = (mom + 3) % 4;
            int de = (mom + 1) % 4;
            int naz = (mom + 2) % 4;
            int nap = mom;
            cout << yy11 << " " << xx11 << endl;
            cout << "le = " << le << " " << preveri(yy11, xx11, le) << endl;
            cout << "de = " << de << " " << preveri(yy11, xx11, de) << endl;
            cout << "naz = " << naz << " " << preveri(yy11, xx11, naz) << endl;
            cout << "nap = " << nap << " " << preveri(yy11, xx11, nap) << endl;

            can[make_pair(y, make_pair(x, le))] = preveri(yy11, xx11, le);
            can[make_pair(y, make_pair(x, de))] = preveri(yy11, xx11, de);
            can[make_pair(y, make_pair(x, naz))] = preveri(yy11, xx11, naz);
            can[make_pair(y, make_pair(x, nap))] = preveri(yy11, xx11, nap);
            ins();
            look = true;
            look = true;
            if (!looked[make_pair(y, make_pair(x, le))] && can[make_pair(y, make_pair(x, le))])
                looked[make_pair(y, make_pair(x, le))] = true, vozi(yy11, xx11, y, x, le);
            look = false;
            //look = true;
            //if (!looked[make_pair(y, make_pair(x, nap))] && can[make_pair(y, make_pair(x, nap))]) looked[make_pair(y, make_pair(x, nap))] = true, vozi(yy11, xx11, y, x, nap);
            //look = false;
            look = true;
            if (!looked[make_pair(y, make_pair(x, de))] && can[make_pair(y, make_pair(x, de))])
                looked[make_pair(y, make_pair(x, de))] = true, vozi(yy11, xx11, y, x, de);
            look = false;
            s1 = "Navigation";
            if (!went[make_pair(y, make_pair(x, le))] && !can[make_pair(y, make_pair(x, le))])
            {
                went[make_pair(y, make_pair(x, le))] = true;
                vozi(syy(yy11, le), sxx(xx11, le), sy(y, le), sx(x, le), le);

                mom = le;
                y = sy(y, mom);
                x = sx(x, mom);
                yy11 = syy(yy11, mom);
                xx11 = sxx(xx11, mom);
            }
            else if (!went[make_pair(y, make_pair(x, nap))] && !can[make_pair(y, make_pair(x, nap))])
            {
                went[make_pair(y, make_pair(x, nap))] = true;
                vozi(syy(yy11, nap), sxx(xx11, nap), sy(y, nap), sx(x, nap), nap);

                mom = nap;
                y = sy(y, mom);
                x = sx(x, mom);
                yy11 = syy(yy11, mom);
                xx11 = sxx(xx11, mom);
            }
            else if (!went[make_pair(y, make_pair(x, de))] && !can[make_pair(y, make_pair(x, de))])
            {
                went[make_pair(y, make_pair(x, de))] = true;
                vozi(syy(yy11, de), sxx(xx11, de), sy(y, de), sx(x, de), de);

                mom = de;
                y = sy(y, mom);
                x = sx(x, mom);
                yy11 = syy(yy11, mom);
                xx11 = sxx(xx11, mom);
            }
            else if (!went[make_pair(y, make_pair(x, naz))] && !can[make_pair(y, make_pair(x, naz))])
            {
                went[make_pair(y, make_pair(x, naz))] = true;
                vozi(syy(yy11, naz), sxx(xx11, naz), sy(y, naz), sx(x, naz), naz);
                mom = naz;
                y = sy(y, mom);
                x = sx(x, mom);
                yy11 = syy(yy11, mom);
                xx11 = sxx(xx11, mom);
            }
            cout << mom << endl;
            waitKey(30);
        }
        else
        {
            ros::spinOnce();
           // cout << "VLEZE" << endl;
           // cout << mozi << endl;
            if (!mozi)
            {
                continue;
            }
            if (!qI.empty())
            {
                cout << qI.front().first << " " << qI.front().second.first << endl;
                if (qI.front().second.second == 0)
                {

                    vozi(0, 0, qI.front().first, qI.front().second.first, 0);
                }
                else
                {
                    isApproaching = 1;
                    approaching(qI.front().first, qI.front().second.first, qI.front().second.second);
                    Kaj = qI.front().second.second;
                    qi.pop();
                    cout << "VLEZEAPP" << endl;
                    isApproaching = 0;
                    if (Kaj == 4 || Kaj == 2)
                    {
                        Kaj = 1;
                        std_msgs::String mgg1;
                        mgg1.data = "extend";
                        roko.publish(mgg1);
                        sleep(5);
                        std_msgs::String mgg;
                        mgg.data = "retract";
                        roko.publish(mgg);
                        sleep(5);
                    }
                    //sleep(5);
                    std_msgs::String mggg;
                    mggg.data = "pridel";
                    semNekaj.publish(mggg);
                    sleep(5);
                    //sleep(5);
                }
                qI.pop();
            }
            waitKey(30);
        }
    }
    return 0;
}

/*
source /home/iletavcioski/ROS/devel/setup.bash
roslaunch exercise7 rins_world.launch
roslaunch exercise3 amcl_simulation.launch 
roslaunch turtlebot_rviz_launchers view_navigation.launch 2>/dev/null
rostopic echo /camera/rgb/image_raw --noarr
rosrun exercise4 colorLocalizer.py
rosrun exercise3 map_goals
rosrun exercise7 move_arm.py

Sepravi vektor je od lokacije kjer aprochas in cilindra in potem naredis arcustanges kota do x osi atan2. Js posiljam kr obicajen gole tako da orientation.z =sin(kot/2), orientation.w = cos(kot/2)
*/
