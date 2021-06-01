cmake_minimum_required(VERSION 2.8.3)
project(turtlebot_custom_navigation)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS tf roscpp sensor_msgs)

# What other packages will need to use this package
catkin_package(
    CATKIN_DEPENDS tf roscpp sensor_msgs
)


###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS})

# Add_executables



#############
## Install ##
#############

# Mark anything (useful) else for installation
