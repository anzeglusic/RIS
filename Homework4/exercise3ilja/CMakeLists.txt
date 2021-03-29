cmake_minimum_required(VERSION 2.8.3)
project(exercise3ilja)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	nav_msgs
	std_msgs
	geometry_msgs
	tf2
	tf2_geometry_msgs
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

find_package(OpenCV)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_executable(map_goalsilja src/map_goalsilja.cpp)
target_link_libraries(map_goalsilja ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

