cmake_minimum_required(VERSION 2.8.3)
project(rotors_control)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mav_msgs
  nav_msgs
  roscpp
  sensor_msgs
  cmake_modules
)

find_package(Eigen REQUIRED)

find_package(TinyXML)
include_directories(${TINYXML_INCLUDE_DIR})
set(UTIL_LIBS /usr/lib/x86_64-linux-gnu/libGLU.so;/usr/lib/x86_64-linux-gnu/libGL.so;/usr/lib/x86_64-linux-gnu/libglut.so;/usr/lib/x86_64-linux-gnu/libXmu.so;/usr/lib/x86_64-linux-gnu/libXi.so;gcop_utils;/usr/lib/x86_64-linux-gnu/libtinyxml.so)

message("UTIL ${UTIL_LIBS}")


catkin_package(
  INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS}
  LIBRARIES lee_position_controller roll_pitch_yawrate_thrust_controller
  CATKIN_DEPENDS geometry_msgs mav_msgs nav_msgs roscpp sensor_msgs
  DEPENDS Eigen
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
        ${TINYXML_INCLUDE_DIR}
  $ENV{MY_GCOP}~/optimalcontrol/gcop
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/algos
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/systems
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/utils
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/views
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/systems/manifolds
        $ENV{MY_GCOP}~/optimalcontrol/gcop/lib/systems/costs
)


FIND_LIBRARY(GCOP_LIBS gcop_algos /home/charris/optimalcontrol/gcop/build/lib/)


add_library(gcop_algos SHARED IMPORTED)
add_library(gcop_systems SHARED IMPORTED)
add_library(gcop_est SHARED IMPORTED)
add_library(gcop_utils SHARED IMPORTED)
add_library(gcop_views SHARED IMPORTED)
add_library(util_libs SHARED IMPORTED)


SET(ALL LIBS "/usr/lib/x86_64-linux-gnu/libGLU.so;/usr/lib/x86_64-linux-gnu/libGL.so;/usr/lib/x86_64-linux-gnu/libglut.so;/usr/lib/x86_64-linux-gnu/libXmu.so;/usr/lib/x86_64-linux-gnu/libXi.so;gcop_utils;/usr/lib/x86_64-linux-gnu/libtinyxml.so
")


set_target_properties(gcop_algos PROPERTIES
        IMPORTED_LOCATION /home/charris/optimalcontrol/gcop/build/lib/libgcop_algos.so)
set_target_properties(gcop_systems PROPERTIES
        IMPORTED_LOCATION /home/charris/optimalcontrol/gcop/build/lib/libgcop_systems.so)
set_target_properties(gcop_est PROPERTIES
        IMPORTED_LOCATION /home/charris/optimalcontrol/gcop/build/lib/libgcop_est.so)
set_target_properties(gcop_utils PROPERTIES
        IMPORTED_LOCATION /home/charris/optimalcontrol/gcop/build/lib/libgcop_utils.so)
set_target_properties(gcop_views PROPERTIES
        IMPORTED_LOCATION /home/charris/optimalcontrol/gcop/build/lib/libgcop_views.so)
set_target_properties(util_libs PROPERTIES
        IMPORTED_LOCATION //usr/lib/x86_64-linux-gnu/libtinyxml.so)
add_dependencies(gcop_utils UTIL_LIBS)
add_dependencies(gcop_systems UTIL_LIBS)
add_dependencies(gcop_views UTIL_LIBS)


#set(GCOP_LIBS gcop_algos gcop_systems gcop_est gcop_utils gcop_views UTIL_LIBS)

#add_library(gcop_libs SHARED IMPORTED)
#set_target_properties(gcop_libs PROPERTIES
#        IMPORTED LOCATION GCOP_LIBS)


add_library(lee_position_controller
  src/library/lee_position_controller.cpp
)

add_library(roll_pitch_yawrate_thrust_controller
  src/library/roll_pitch_yawrate_thrust_controller.cpp
)

add_library(ddp_controller
        src/library/lee_position_controller.cpp)

target_link_libraries(lee_position_controller ${catkin_LIBRARIES})
add_dependencies(lee_position_controller ${catkin_EXPORTED_TARGETS})

target_link_libraries(roll_pitch_yawrate_thrust_controller ${catkin_LIBRARIES})
add_dependencies(roll_pitch_yawrate_thrust_controller ${catkin_EXPORTED_TARGETS})

target_link_libraries(ddp_controller ${catkin_LIBRARIES})
add_dependencies(ddp_controller ${catkin_EXPORTED_TARGETS})

add_executable(lee_position_controller_node src/nodes/lee_position_controller_node.cpp)
add_dependencies(lee_position_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(lee_position_controller_node
  lee_position_controller ${catkin_LIBRARIES})

add_executable(roll_pitch_yawrate_thrust_controller_node
  src/nodes/roll_pitch_yawrate_thrust_controller_node.cpp)
add_dependencies(roll_pitch_yawrate_thrust_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(roll_pitch_yawrate_thrust_controller_node
  roll_pitch_yawrate_thrust_controller ${catkin_LIBRARIES})

add_executable(ddp_controller_node
        src/nodes/ddp_controller_node.cpp)
add_dependencies(ddp_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(ddp_controller_node
        ddp_controller ${catkin_LIBRARIES} gcop_algos gcop_views gcop_systems gcop_utils gcop_est ${ALL_LIBS} util_libs)



install(TARGETS lee_position_controller roll_pitch_yawrate_thrust_controller ddp_controller
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS lee_position_controller_node roll_pitch_yawrate_thrust_controller_node ddp_controller_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
