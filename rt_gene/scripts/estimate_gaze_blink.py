#! /usr/bin/env python

"""
Convolutional Neural Network (CNN) for eye gaze estimation
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Hyung Jin Chang (hj.chang@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import collections
import os

import numpy as np
import rospkg
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rt_gene.msg import MSG_SubjectImagesList, MSG_Gaze, MSG_GazeList, MSG_BlinkList, MSG_Blink
from sensor_msgs.msg import Image
from tf import transformations
from tqdm import tqdm
import cv2

import rt_gene.gaze_tools as gaze_tools
import rt_gene.ros_tools as ros_tools
from rt_gene.subject_ros_bridge import SubjectListBridge


class GazeBlinkEstimatorROS(object):

    def __init__(self, device_id_gaze, device_id_blink, gaze_model_files, blink_model_files, blink_threshold):
        self.bridge = CvBridge()
        self.subjects_bridge = SubjectListBridge()

        self.tf2_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.tf_prefix = rospy.get_param("~tf_prefix", default="gaze")
        self.headpose_frame = self.tf_prefix + "/head_pose_estimated"
        self.gaze_backend = rospy.get_param("~gaze_backend", default="pytorch")
        self.blink_backend = rospy.get_param("~blink_backend", default="pytorch")
        self.visualise_eyepose = rospy.get_param("~visualise_eyepose", default=True)
        self.visualise_blink = rospy.get_param("~visualise_blink", default=True)

        assert self.gaze_backend == "pytorch" and self.blink_backend == "pytorch", "Only PyTorch is suitable for combined gaze and blink estimation; this is owing to the blink detector in Tensorflow expecting a different image to the gaze detector"

        if self.gaze_backend == "tensorflow":
            from rt_gene.estimate_gaze_tensorflow import GazeEstimator
            self._gaze_estimator = GazeEstimator(device_id_gaze, gaze_model_files)
        elif self.gaze_backend == "pytorch":
            from rt_gene.estimate_gaze_pytorch import GazeEstimator
            self._gaze_estimator = GazeEstimator(device_id_gaze, gaze_model_files)
        else:
            raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

        if self.blink_backend == "tensorflow":
            from rt_bene.estimate_blink_tensorflow import BlinkEstimatorTensorflow
            self._blink_estimator = BlinkEstimatorTensorflow(device_id_blink, blink_model_files, blink_threshold)
        elif self.blink_backend == "pytorch":
            from rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch
            self._blink_estimator = BlinkEstimatorPytorch(device_id_blink, blink_model_files, blink_threshold)

        self.blink_publisher = rospy.Publisher("/subjects/blink", MSG_BlinkList, queue_size=3)
        if self.visualise_blink:
            self.subjects_blink_img = rospy.Publisher(rospy.get_param("~viz_topic", "/subjects/blink_images"), Image, queue_size=3)

        if self.visualise_eyepose:
            self.subjects_gaze_img = rospy.Publisher("/subjects/gazeimages", Image, queue_size=3)
        self.gaze_publishers = rospy.Publisher("/subjects/gaze", MSG_GazeList, queue_size=3)

        self._last_time = rospy.Time().now()
        self._freq_deque = collections.deque(maxlen=30)  # average frequency statistic over roughly one second
        self._latency_deque = collections.deque(maxlen=30)

        rospy.Subscriber("/subjects/images", MSG_SubjectImagesList, self.image_callback, queue_size=3, buff_size=2 ** 24)

    def publish_image(self, image, image_publisher, timestamp):
        """This image publishes the `image` to the `image_publisher` with the given `timestamp`."""
        image_ros = self.bridge.cv2_to_imgmsg(image, "rgb8")
        image_ros.header.stamp = timestamp
        image_publisher.publish(image_ros)

    def image_callback(self, subject_image_list):
        """This method is called whenever new input arrives. The input is first converted in a format suitable
        for the gaze estimation network (see :meth:`input_from_image`), then the gaze is estimated (see
        :meth:`estimate_gaze`. The estimated gaze is overlaid on the input image (see :meth:`visualize_eye_result`),
        and this image is published along with the estimated gaze vector (see :meth:`publish_image` and
        :func:`publish_gaze`)"""
        timestamp = subject_image_list.header.stamp
        camera_frame = subject_image_list.header.frame_id

        subjects_dict = self.subjects_bridge.msg_to_images(subject_image_list)
        input_r_list = []
        input_l_list = []
        input_head_list = []
        valid_subject_list = []
        for subject_id, s in subjects_dict.items():
            try:
                transform_msg = self.tf2_buffer.lookup_transform(camera_frame, self.headpose_frame + str(subject_id), timestamp)
                rot_head = transform_msg.transform.rotation
                _m = transformations.quaternion_matrix([rot_head.x, rot_head.y, rot_head.z, rot_head.w])
                euler_angles_head = list(transformations.euler_from_matrix(np.dot(ros_tools.camera_to_ros, _m)))

                euler_angles_head = gaze_tools.limit_yaw(euler_angles_head)

                phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(euler_angles_head)
                input_head_list.append([theta_head, phi_head])
                input_r_list.append(self._gaze_estimator.input_from_image(s.right))
                input_l_list.append(self._gaze_estimator.input_from_image(s.left))
                valid_subject_list.append(subject_id)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException):
                pass

        if len(valid_subject_list) == 0:
            return

        gaze_est = self._gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                              inference_input_right_list=input_r_list,
                                                              inference_headpose_list=input_head_list)

        blink_est = self._blink_estimator.predict(input_l_list, input_r_list)

        subject_subset = dict((k, subjects_dict[k]) for k in valid_subject_list if k in subjects_dict)
        self.publish_gaze_msg(subject_image_list.header, subject_subset, gaze_est.tolist())
        self.publish_blink_msg(subject_image_list.header, subject_subset, blink_est)

        subjects_gaze_img_list = []
        blink_image_list = []
        for subject_id, gaze, blink_p in zip(valid_subject_list, gaze_est.tolist(), blink_est.tolist()):
            subjects_dict[subject_id].gaze = gaze
            self.publish_gaze(gaze, timestamp, subject_id)

            if self.visualise_eyepose:
                s = subjects_dict[subject_id]
                r_gaze_img = self._gaze_estimator.visualize_eye_result(s.right, gaze)
                l_gaze_img = self._gaze_estimator.visualize_eye_result(s.left, gaze)
                s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)
                subjects_gaze_img_list.append(s_gaze_img)

            if self.visualise_blink:
                resized_face = cv2.resize(subjects_dict[subject_id].face, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                blink_image_list.append(self._blink_estimator.overlay_prediction_over_img(resized_face, blink_p))

        if len(subjects_gaze_img_list) > 0:
            gaze_img_msg = self.bridge.cv2_to_imgmsg(np.hstack(subjects_gaze_img_list).astype(np.uint8), "bgr8")
            gaze_img_msg.header.stamp = timestamp
            self.subjects_gaze_img.publish(gaze_img_msg)

        if len(blink_image_list) > 0:
            blink_viz_img = self.bridge.cv2_to_imgmsg(np.hstack(blink_image_list), encoding="bgr8")
            blink_viz_img.header.stamp = subject_image_list.header.stamp
            self.subjects_blink_img.publish(blink_viz_img)

        _now = rospy.Time().now()
        _freq = 1.0 / (_now - self._last_time).to_sec()
        self._freq_deque.append(_freq)
        self._latency_deque.append(_now.to_sec() - timestamp.to_sec())
        self._last_time = _now
        tqdm.write(
            '\033[2K\033[1;32mTime now: {:.2f} message color: {:.2f} latency: {:.2f}s for {} subject(s) {:.0f}Hz\033[0m'.format(
                (_now.to_sec()), timestamp.to_sec(), np.mean(self._latency_deque), len(valid_subject_list), np.mean(self._freq_deque)), end="\r")

    def publish_gaze_msg(self, header, subjects, gazes):
        gaze_msg_list = MSG_GazeList()
        gaze_msg_list.header = header
        for subjects_id, gaze in zip(subjects.keys(), gazes):
            gaze_msg = MSG_Gaze()
            gaze_msg.subject_id = subjects_id
            gaze_msg.theta = gaze[0]
            gaze_msg.phi = gaze[1]
            gaze_msg_list.subjects.append(gaze_msg)

        self.gaze_publishers.publish(gaze_msg_list)

    def publish_blink_msg(self, header, subjects, probabilities):
        blink_msg_list = MSG_BlinkList()
        blink_msg_list.header = header
        for subject_id, p in zip(subjects.keys(), probabilities):
            blink_msg = MSG_Blink()
            blink_msg.subject_id = str(subject_id)
            blink_msg.blink = bool(p >= self._blink_estimator.threshold)
            blink_msg.probability = p
            blink_msg_list.subjects.append(blink_msg)

        self.blink_publisher.publish(blink_msg_list)

    def publish_gaze(self, est_gaze, msg_stamp, subject_id):
        """Publish the gaze vector as a PointStamped."""
        theta_gaze = est_gaze[0]
        phi_gaze = est_gaze[1]
        euler_angle_gaze = gaze_tools.get_euler_from_phi_theta(phi_gaze, theta_gaze)
        quaternion_gaze = transformations.quaternion_from_euler(*euler_angle_gaze)

        t = TransformStamped()
        t.header.stamp = msg_stamp
        t.header.frame_id = self.headpose_frame + str(subject_id)
        t.child_frame_id = self.tf_prefix + "/world_gaze" + str(subject_id)
        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0.05  # publish it 5cm above the head pose's origin (nose tip)
        t.transform.rotation.x = quaternion_gaze[0]
        t.transform.rotation.y = quaternion_gaze[1]
        t.transform.rotation.z = quaternion_gaze[2]
        t.transform.rotation.w = quaternion_gaze[3]

        try:
            self.tf2_broadcaster.sendTransform([t])
        except rospy.ROSException as exc:
            if str(exc) == "publish() to a closed topic":
                pass
            else:
                raise exc


if __name__ == "__main__":
    try:
        rospy.init_node("estimate_gaze_blink")
        gaze_estimator = GazeBlinkEstimatorROS(device_id_gaze=rospy.get_param("~device_id_gazeestimation", default="/gpu:0"),
                                               device_id_blink=rospy.get_param("~device_id_blinkestimation", default="/gpu:0"),
                                               gaze_model_files=[os.path.join(rospkg.RosPack().get_path("rt_gene"), model_file) for model_file in
                                                                 rospy.get_param("~gaze_model_files")],
                                               blink_model_files=[os.path.join(rospkg.RosPack().get_path("rt_gene"), model_file) for model_file in
                                                                  rospy.get_param("~blink_model_files")],
                                               blink_threshold=rospy.get_param("~blink_threshold", default=0.425))
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
    except rospy.ROSException as e:
        if str(e) == "publish() to a closed topic":
            pass
        else:
            raise e
    except KeyboardInterrupt:
        print("Shutting down")
