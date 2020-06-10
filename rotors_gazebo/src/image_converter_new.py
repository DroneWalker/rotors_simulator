#!/usr/bin/env python

from __future__ import print_function

import os.path
import roslib
roslib.load_manifest('rotors_gazebo')

import sys
import csv
import rospy
import cv2
import math
import numpy as np
import tf
from image_geometry import PinholeCameraModel

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import landing_sign_detection as landing_op

class image_converter():

    def __init__(self):

        # initializing the subscribers and publishers
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.detector_switch_sub = rospy.Subscriber("/switch_update", String, self.callback_detector_switch)
        self.image_sub = rospy.Subscriber("/firefly/vi_sensor/camera_depth/camera/image_raw", Image, self.callback_image)
        self.pos_sub = rospy.Subscriber("/firefly/ground_truth/pose", Pose, self.callback_pos)
        self.image_pub = rospy.Publisher("/detected_sign", Image, queue_size=10)
        self.pos_pub = rospy.Publisher("/tinstancearget_location", Pose, queue_size=10)
        self.proc_done_pub = rospy.Publisher("/process_status", String, queue_size=10)

        # initializing some parameters
        self.output_filename = 'training_data_basic_1000.csv'
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
        self.wait_time = 100
        self.RAD_to_DEG = 180.0 / math.pi
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []

        self.img_detection_fnc = False

    def callback_detector_switch(self, msg):

        if msg.data == 'on':
            self.detector_switch = 'on'
        else:
            self.detector_switch = 'off'
            self.img_proc_status = 'processing'
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

    def callback_image(self, data):
        cv_image = []

        try:
            self.output_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        if self.detector_switch == 'off':

            text_for_image = 'Initialing landing mark detector...Standing by...'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, text_for_image, (10, 25), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        else:

            # process the image to detect the landing mark
            if self.count_title < 1 and self.img_detection_fnc:

                # please modify the output file name accordingly
		
                csv_save_complete_name = os.path.join('/home/charris/Desktop/data_collector/', self.output_filename)
                with open(csv_save_complete_name, 'a') as csvfile:
                    infowriter = csv.writer(csvfile)
                    infowriter.writerow(("current_x", "current_y", "current_z", "current_yaw",
                                         "image_center_x", "image_center_y", "f_min", "f_num_point",
                                         "mj_min", "mj_num_point", "mj_x_min", "mj_y_min", "mj_x_max", "mj_y_max"))

                self.count_title = self.count_title + 1

            else:
                pass

            if self.count_image == self.wait_time + 1:

                # set the file name for the output image (for future deep learning training)
                name_world = 'basic'
                tag_x = str(np.round(self.current_x,2))
                tag_y = str(np.round(self.current_y,2))
                tag_z = str(np.round(self.current_z,2))
                tag_yaw = str(np.round(self.current_yaw,0))
                name_output_image = 'output_image_x_'  + tag_x + '_y_' + tag_y + '_z_' + tag_z + '_yaw_' + tag_yaw + '_world_name_' + name_world + '.jpg'

                image_save_complete_name = os.path.join('/home/charris/Desktop/data_collector/', name_output_image)

                # output the image of current altitude, yaw angle, and world
                cv2.imwrite(image_save_complete_name, self.output_image)

                if self.img_detection_fnc:
                    # please modify the output file name accordingly
                    csv_save_complete_name = os.path.join('/home/charris/Desktop/data_collector/', self.output_filename)
                    with open(csv_save_complete_name, 'a') as csvfile:
                        infowriter = csv.writer(csvfile)
                        infowriter.writerow((str(self.output_filename), str(self.current_x), str(self.current_y), str(self.current_z), str(self.current_yaw),
                                             str(self.pixel_center[0]), str(self.pixel_center[1]), str(self.f_min), str(self.f_num_point),
                                             str(self.mj_min), str(self.mj_num_point), str(self.x_min), str(self.y_min), str(self.x_max), str(self.y_max)
                                             ))

                    self.img_proc_status = 'finished'

                else:
                    self.img_proc_status = 'finished'
                    pass

            if self.img_detection_fnc:

                if self.count_image == self.wait_time:

                    image_processing = landing_op.landing_sign_detection(cv_image)

                    self.pixel_center = image_processing[0]
                    self.major_group = image_processing[1]
                    self.feature_point = image_processing[2]
                    self.f_min = image_processing[3]
                    self.f_num_point = image_processing[4]
                    self.mj_min = image_processing[5]
                    self.mj_num_point = image_processing[6]
                    self.other_group = image_processing[7]
                    self.x_min = image_processing[8]
                    self.x_max = image_processing[9]
                    self.y_min = image_processing[10]
                    self.y_max = image_processing[11]

                if self.pixel_center != []:

                    cv2.circle(cv_image, (self.pixel_center[0], self.pixel_center[1]), 4, (0, 0, 255), -1)
                    text_for_image = 'Center of the landing mark: ' + 'x = ' + str(self.pixel_center[0]) + ', ' + 'y = ' + str(self.pixel_center[1]) + ' (in pixel)'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(cv_image, text_for_image, (10, 25), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    # print(self.count_image)

                    # indicating all the feature points in the output image
                    for point in self.feature_point:
                        cv2.circle(cv_image, (point[1], point[0]), 2, (0,255, 0), -1)

                    # indicating all the major feature points in the output image
                    for index in self.major_group:
                        cv2.circle(cv_image, (self.feature_point[index][1], self.feature_point[index][0]), 2, (255, 100, 0), -1)

                    # indicating all the major feature points in the output image
                    for index in self.other_group:
                        cv2.circle(cv_image, (self.feature_point[index][1], self.feature_point[index][0]), 2, (63, 99, 255), -1)

                else:
                    if self.count_image < self.wait_time:
                        text_for_image = 'Landing mark detection is under process...'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(cv_image, text_for_image, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # print(self.count_image)

                    if self.count_image > self.wait_time:
                        text_for_image = 'No major feature points founded, unable to locate the center of landing mark!'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(cv_image, text_for_image, (10, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # print(self.count_image)

            self.count_image = self.count_image + 1

        cv2.imshow("Result of Landing Mark Detection", cv_image)
        cv2.waitKey(3)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        self.proc_done_pub.publish(self.img_proc_status)

    def callback_pos(self, msg):


        if self.detector_switch == 'on':

            x_c = 0.0
            y_c = 0.0
            z_c = 0.0
            x_org = 0.0
            y_org = 0.0
            z_org = 0.0

            if self.img_detection_fnc:
                # original drone location:
                x_org = msg.position.x
                y_org = msg.position.y
                z_org = msg.position.z

                self.current_x = x_org
                self.current_y = y_org
                self.current_z = z_org

                # need to transform back to the Eular angle
                orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                self.current_yaw = round(self.RAD_to_DEG * euler_from_quaternion(orientation_list)[2])

                # camera location relative to drone:
                x_c = 0.0
                y_c = 0.0
                z_c = 0.0
            else:
                pass

            if self.pixel_center != []:

                pose = Pose()
                point_msg = PoseStamped()

                # create vectors to scale the normalized coordinates in camera frame
                t_1 = np.array([[x_org, y_org, z_org]])
                t_2 = np.array([[x_c, y_c, z_c]])

                # translation matrix from world to camera
                t_W2C = np.add(t_1, t_2)

                # transformation from image coordinate to camera coordinate
                cam_info = rospy.wait_for_message("/firefly/vi_sensor/camera_depth/camera/camera_info", CameraInfo, timeout=None)
                img_proc = PinholeCameraModel()
                img_proc.fromCameraInfo(cam_info)
                cam_model_point = img_proc.projectPixelTo3dRay(self.pixel_center)
                cam_model_point = np.round(np.multiply(np.divide(cam_model_point, cam_model_point[2]), t_W2C[0, 2]), decimals=0)

                # creating a place holder to save the location info
                point_msg.pose.position.x = cam_model_point[0]
                point_msg.pose.position.y = cam_model_point[1]
                point_msg.pose.position.z = cam_model_point[2]
                point_msg.pose.orientation.x = 0
                point_msg.pose.orientation.y = 0
                point_msg.pose.orientation.z = 0
                point_msg.pose.orientation.w = 1
                point_msg.header.stamp = rospy.Time.now()
                point_msg.header.frame_id = img_proc.tfFrame()

                # initiate the coordinate (frame) transformation
                self.tf_listener.waitForTransform(img_proc.tfFrame(), "world", rospy.Time.now(), rospy.Duration(1.0))

                # calculate the transformed coordinates
                tf_point = self.tf_listener.transformPose("world", point_msg)

                x_new = 0.0 # round(tf_point.pose.position.x, 2)
                y_new = 0.0 # round(tf_point.pose.position.y, 2)
                z_new = msg.position.z - 0.01
                # z_new = (1 - 0.01 * self.decent_rate) * msg.position.z

                pose.position.x = x_new
                pose.position.y = y_new
                pose.position.z = z_new

                pose.orientation.x = msg.orientation.x
                pose.orientation.y = msg.orientation.y
                pose.orientation.z = msg.orientation.z
                pose.orientation.w = msg.orientation.w

                self.pos_pub.publish(pose)

            else:
                # If there is no landing mark detected
                pose = Pose()

                # original drone location:
                x_org = msg.position.x
                y_org = msg.position.y
                z_org = msg.position.z

                self.current_x = x_org
                self.current_y = y_org
                self.current_z = z_org

                # need to transform back to the Eular angle
                orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                self.current_yaw = round(self.RAD_to_DEG * euler_from_quaternion(orientation_list)[2])

                pose.position.x = msg.position.x
                pose.position.y = msg.position.y
                pose.position.z = msg.position.z

                pose.orientation.x = msg.orientation.x
                pose.orientation.y = msg.orientation.y
                pose.orientation.z = msg.orientation.z
                pose.orientation.w = msg.orientation.w

                self.pos_pub.publish(pose)

        else:
            # if the detection switch doesn't turn on, skip the coordinate computation
            pass

def main(args):

    rospy.init_node('Image_Processer')
    image_converter()
    rospy.spin()

if __name__ == '__main__':

    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
    
 
