#!/usr/bin/env python

from __future__ import print_function

import os.path
import roslib

from datetime import datetime
import sys
import csv
import rospy
import cv2
import math
import numpy as np
import tf
from image_geometry import PinholeCameraModel

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped, \
    PoseWithCovariance, TwistWithCovariance, Point
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import landing_sign_detection as landing_op

roslib.load_manifest('rotors_gazebo')


class DataCollector():

    def __init__(self):

        # initializing some parameters
        self.output_filename = 'training_data_obstacle_avoidance.csv'
        self.detector_switch = 'off'
        self.img_proc_status = 'processing'
        self.count_image = 0
        self.count_title = 0
        self.output_image = []
        self.feature_point = []
        self.major_group = []
        self.other_group = []
        self.pixel_center = []
        self.f_min = []
        self.f_num_point = []
        self.mj_min = []
        self.mj_num_point = []
        self.current_x = []
        self.current_y = []
        self.current_z = []
        self.current_yaw = []
        self.decent_rate = 25     # in %
        self.wait_time = 5
        self.RAD_to_DEG = 180.0 / math.pi
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []

        # Initialize data to collect
        self.imagename = []
        self.pose_x = []
        self.pose_y = []
        self.pose_z = []
        self.pose_q1 = []
        self.pose_q2 = []
        self.pose_q3 = []
        self.pose_q4 = []
        self.xdot = []
        self.ydot = []
        self.zdot = []
        self.p = []
        self.q = []
        self.r = []
        self.astar_x = []
        self.astar_y = []
        self.astar_z = []
        self.mpc_roll = []
        self.mpc_pitch = []
        self.mpc_yaw = []
        self.mpc_thr = []
        # Update this to working folder
        self.foldername = os.path.join("/home/charris/Desktop/data_collector/",
                                       datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.img_detection_fnc = False
        self.datacollected = False


        self.image_sub = rospy.Subscriber("/DJI/vi_sensor/camera_depth/camera/image_raw", Image, self.callback_image)
        self.odometry_sub = rospy.Subscriber("/DJI/ground_truth/odometry", Odometry, self.callback_odom)
        self.astar_sub = rospy.Subscriber("/DJI/astar/nextpoint", Point, self.callback_astar_point)
        # self.mpc_sub = rospy.Subscriber("DJI/command/rol")

        # initializing the subscribers and publishers
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.detector_switch_sub = rospy.Subscriber("/switch_update", String, self.callback_detector_switch)
        self.image_pub = rospy.Publisher("/detected_sign", Image, queue_size=10)
        self.pos_pub = rospy.Publisher("/target_location", Pose, queue_size=10)
        self.proc_done_pub = rospy.Publisher("/process_status", String, queue_size=10)




    def reset_data(self):
        self.imagename = []
        self.pose_x = []
        self.pose_y = []
        self.pose_z = []
        self.pose_q1 = []
        self.pose_q2 = []
        self.pose_q3 = []
        self.pose_q4 = []
        self.xdot = []
        self.ydot = []
        self.zdot = []
        self.p = []
        self.q = []
        self.r = []
        self.astar_x = []
        self.astar_y = []
        self.astar_z = []
        self.mpc_roll = []
        self.mpc_pitch = []
        self.mpc_yaw = []
        self.mpc_thr = []

        self.datacollected = False

        self.count_image = 0
        self.feature_point = []
        self.major_group = []
        self.other_group = []
        self.pixel_center = []
        self.f_min = []
        self.f_num_point = []
        self.mj_min = []
        self.mj_num_point = []
        self.current_x = []
        self.current_y = []
        self.current_z = []
        self.current_yaw = []
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []


    def callback_detector_switch(self, msg):

        if msg.data == 'on':
            self.detector_switch = 'on'
            self.img_proc_status = 'collecting_data'
        else:
            self.detector_switch = 'off'
            self.img_proc_status = 'traveling'
            self.reset_data()


    def callback_astar_point(self, msg):

        if self.detector_switch == 'on':

            self.astar_x = msg.x
            self.astar_y = msg.y
            self.astar_z = msg.z

            print("Collected Astar solution")


    def callback_image(self, data):
        cv_image = []

        try:
            self.output_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        if self.detector_switch == 'off':

            text_for_image = 'Initializing data collection...'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, text_for_image, (10, 25), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        else:

            # collect the image for storage and run Astar calculation
            if self.count_title < 1:
                if os.path.exists(self.foldername) is False:
                    os.makedirs(self.foldername)

                csv_save_complete_name = os.path.join(self.foldername, self.output_filename)
                with open(csv_save_complete_name, 'a') as csvfile:
                    infowriter = csv.writer(csvfile)
                    # infowriter.writerow(("current_x", "current_y", "current_z", "current_yaw",
                    #                      "image_center_x", "image_center_y", "f_min", "f_num_point",
                    #                      "mj_min", "mj_num_point", "mj_x_min", "mj_y_min", "mj_x_max", "mj_y_max"))
                    infowriter.writerow(("IMAGENAME", "POSE_x", "POSE_y", "POSE_z", "POSE_q1", "POSE_q2", "POSE_q3", "POSE_q4",
                                         "POSE_xdot", "POSE_ydot", "POSE_zdot", "POSE_p", "POSE_q", "POSE_r",
                                         "ASTAR_x", "ASTAR_y", "ASTAR_z", "MPC_roll", "MPC_pitch", "MPC_yaw", "MPC_thr"))

                self.count_title = self.count_title + 1

            else:
                pass

            if self.count_image == self.wait_time + 1:

                # set the file name for the output image (for future deep learning training)
                name_world = 'obstacle'
                tag_x = str(np.round(self.pose_x,3))
                tag_y = str(np.round(self.pose_y,3))
                tag_z = str(np.round(self.pose_z,3))
                tag_yaw = str(np.round(self.pose_q4,0))
                name_output_image = 'output_image_x_'  + tag_x + '_y_' + tag_y + '_z_' + tag_z + '_w_' + tag_yaw + '_' + name_world + '.jpg'

                image_save_complete_name = os.path.join(self.foldername, name_output_image)

                self.imagename = image_save_complete_name

                # output the image of current altitude, yaw angle, and world
                cv2.imwrite(image_save_complete_name, self.output_image)

                # FIRST DO ASTAR AND MPC!!!

                # please modify the output file name accordingly
                csv_save_complete_name = os.path.join(self.foldername, self.output_filename)
                with open(csv_save_complete_name, 'a') as csvfile:
                    infowriter = csv.writer(csvfile)
                    infowriter.writerow((str(self.imagename), str(self.pose_x), str(self.pose_y), str(self.pose_z), str(self.pose_q1),
                                         str(self.pose_q2), str(self.pose_q3), str(self.pose_q4), str(self.xdot),
                                         str(self.ydot), str(self.zdot), str(self.p), str(self.q), str(self.r),
                                         str(self.astar_x), str(self.astar_y), str(self.astar_z), str(self.mpc_roll),
                                         str(self.mpc_pitch), str(self.mpc_yaw), str(self.mpc_thr)))

                self.img_proc_status = 'finished'

            self.count_image = self.count_image + 1

        cv2.imshow("Result of Landing Mark Detection", cv_image)
        cv2.waitKey(3)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        # WAIT FOR A* SOLUTION -> MAYBE CALL IT TO MAKE SOLUTIN SO IT ISNT ALWAYS CALCULATING


        self.proc_done_pub.publish(self.img_proc_status)

    def callback_odom(self, msg):

        if self.detector_switch == 'on':
            self.pose_x = msg.pose.pose.position.x
            self.pose_y = msg.pose.pose.position.y
            self.pose_z = msg.pose.pose.position.z
            self.pose_q1 = msg.pose.pose.orientation.x
            self.pose_q2 = msg.pose.pose.orientation.y
            self.pose_q3 = msg.pose.pose.orientation.z
            self.pose_q4 = msg.pose.pose.orientation.w
            self.xdot = msg.twist.twist.linear.x
            self.ydot = msg.twist.twist.linear.y
            self.zdot = msg.twist.twist.linear.z
            self.p = msg.twist.twist.angular.x
            self.q = msg.twist.twist.angular.y
            self.r = msg.twist.twist.angular.z

        else:
            pass

def main(args):

    rospy.init_node('Data_Collector')
    DataCollector()
    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass

