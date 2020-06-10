/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iostream>
#include <tf/transform_broadcaster.h> 
#include <angles/angles.h>
#include <std_msgs/String.h>
#include <sstream>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

using namespace std;
double original_yaw, original_yaw_deg;
double new_x, new_y, new_z, new_yaw;
string img_proc_status;

class WaypointWithTime {
 public:
  WaypointWithTime()
      : yaw(0.0) {
  }

  WaypointWithTime(float x, float y, float z, float _yaw)
      : position(x, y, z), yaw(_yaw) {
  }

  Eigen::Vector3d position;
  double yaw;
};

void procCallback (const std_msgs::String::ConstPtr& msg) {

  if (msg->data == "finished") {
    
    img_proc_status = "finished";
    ROS_INFO("image process: %s", img_proc_status.c_str());
    
  }
}


// main code
int main(int argc, char** argv) {

  ros::init(argc, argv, "waypoint_landing_detection_publisher");
  ros::NodeHandle nh;
  ros::Publisher trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);
  ros::Publisher detector_switch_pub = nh.advertise<std_msgs::String>("/switch_update", 10);
  ros::Subscriber img_proc_status_sub;

  ROS_INFO("Started waypoint_publisher_new_automation.");

  ros::V_string args;
  ros::removeROSArgs(argc, argv, args);
  std::vector<WaypointWithTime> waypoints;
  const float DEG_2_RAD = M_PI / 180.0;
  double input_pose_x, input_pose_y, input_pose_z;
  double desired_yaw;

  trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
  trajectory_msg.header.stamp = ros::Time::now();

  input_pose_x = std::stof(args.at(1));
  input_pose_y = std::stof(args.at(2));
  input_pose_z = std::stof(args.at(3));

  desired_yaw = std::stof(args.at(4)) * DEG_2_RAD;

  std::ifstream wp_file(args.at(5).c_str());

  if (wp_file.is_open()) {
    double x, y, z, yaw;

    // Only read complete waypoints.
    while (wp_file >> x >> y >> z >> yaw) {

        waypoints.push_back(WaypointWithTime(x, y, z, yaw * DEG_2_RAD));

    }
    wp_file.close();
    ROS_INFO("Read %d waypoints.", (int) waypoints.size());
  } else {
    ROS_ERROR_STREAM("Unable to open poses file: " << args.at(5));
    return -1;
  }

  original_yaw = desired_yaw;
  original_yaw_deg = std::stof(args.at(4));

  Eigen::Vector3d desired_position(input_pose_x, input_pose_y, input_pose_z);
  mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(desired_position, desired_yaw, &trajectory_msg);

  while (trajectory_pub.getNumSubscribers() == 0 && ros::ok()) {
    ROS_INFO("There is no subscriber available, trying again in 1 second.");
    ros::Duration(1.0).sleep();
  }

  ROS_INFO("Publishing waypoint ON NAMESPACE %s: [%f, %f, %f].", nh.getNamespace().c_str(), desired_position.x(), desired_position.y(), desired_position.z(), original_yaw_deg);
  trajectory_pub.publish(trajectory_msg);

  ROS_INFO("Begin autonomous operation.");  

  // turn OFF the landing mark detector 
  std_msgs::String msg_2;
  std::stringstream ss_2;
  ss_2 << "off";
  msg_2.data = ss_2.str();
  detector_switch_pub.publish(msg_2);    
  ros::Duration(1.0).sleep();

  ROS_INFO("Detection switch: OFF.");

  for (size_t i = 0; i < waypoints.size(); ++i) {
  
    ROS_INFO("Start the Image Processing.");
    ROS_INFO("Detection switch: ON.");

    // turn ON the landing mark detector switch
    std_msgs::String msg_3;
    std::stringstream ss_3;
    ss_3 << "on";
    msg_3.data = ss_3.str();
    detector_switch_pub.publish(msg_3);
    ros::Duration(1.0).sleep();

    bool img_proc = true;
    
    while (img_proc) {

	img_proc_status_sub = nh.subscribe("/process_status", 1, procCallback);
        ros::spinOnce();
        
	if (img_proc_status == "finished") {
          img_proc = false;
        }
    }

    // update the waypoints to the new x, y, z, and yaw
    WaypointWithTime& wp = waypoints[i];
    new_x = wp.position.x();
    new_y = wp.position.y();
    new_z = wp.position.z();
    new_yaw = wp.yaw;

    ros::Duration(1.0).sleep();

    // turn OFF the landing mark detector switch
    std_msgs::String msg_4;
    std::stringstream ss_4;
    ss_4 << "off";
    msg_4.data = ss_4.str();
    detector_switch_pub.publish(msg_4);
    ros::Duration(1.0).sleep();

    ROS_INFO("Detection switch: OFF.");

    Eigen::Vector3d desired_position_new(new_x, new_y, new_z);
    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(desired_position_new, new_yaw, &trajectory_msg);

    // Wait for some time to create the ros publisher.
    ros::Duration(1.0).sleep();

    ROS_INFO("Publishing waypoint on namespace %s: [%f, %f, %f, %f].", nh.getNamespace().c_str(),desired_position_new.x(),desired_position_new.y(),desired_position_new.z(),new_yaw);
    trajectory_pub.publish(trajectory_msg);
    
    double delay_time = 1.0;
    ROS_INFO("Wait for %f seconds for drone fly to the new waypoint.", delay_time);
    ros::Duration(1.0).sleep();
  }
  // ros::spinOnce();
  ros::shutdown();
  return 0;
}
