#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point


COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'

TARGET_FRAME = 'base_link'

# Ore model height = 0.0762 m.
# Camera depth measures the visible top face; Task 1A expects the ore middle.
ORE_MIDDLE_OFFSET_Z = 0.0381


class OreDetector(Node):

    def __init__(self):
        super().__init__('ore_detector')

        self.bridge = CvBridge()

        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.color_frame_id = None
        self.depth_frame_id = None

        # -----------------------------------------------------
        # TF2 listener
        # -----------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # -----------------------------------------------------
        # TF broadcaster
        # -----------------------------------------------------

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(
            self
        )

        # -----------------------------------------------------
        # Camera subscriptions
        # -----------------------------------------------------

        self.color_sub = self.create_subscription(
            Image,
            COLOR_TOPIC,
            self.color_callback,
            qos_profile_sensor_data
        )

        self.depth_sub = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.info_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(
            0.5,
            self.process
        )

        self.get_logger().info(
            'Starting Task 1A perception + TF publisher...'
        )

    # =========================================================
    # CAMERA CALLBACKS
    # =========================================================

    def color_callback(self, msg):

        try:
            self.color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            self.color_frame_id = msg.header.frame_id

        except CvBridgeError as e:

            self.get_logger().error(
                f'Color conversion failed: {e}'
            )

    def depth_callback(self, msg):

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

            self.depth_frame_id = msg.header.frame_id

        except CvBridgeError as e:

            self.get_logger().error(
                f'Depth conversion failed: {e}'
            )

    def info_callback(self, msg):

        self.camera_info = msg

    # =========================================================
    # ORE DETECTION
    # =========================================================

    def detect_ores(self, image):

        center_ore_list = []
        ore_type_list = []

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        color_ranges = {

            'azurite_ore': (
                np.array([95, 180, 100]),
                np.array([115, 255, 255])
            ),

            'malachite_ore': (
                np.array([60, 180, 100]),
                np.array([85, 255, 255])
            ),

            'vanadinite_ore': (
                np.array([5, 200, 150]),
                np.array([15, 255, 255])
            )
        }

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        for ore_type, (lower, upper) in color_ranges.items():

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                area = cv2.contourArea(
                    contour
                )

                if area < 100 or area > 10000:
                    continue

                moments = cv2.moments(
                    contour
                )

                if moments['m00'] == 0:
                    continue

                cX = int(
                    moments['m10'] /
                    moments['m00']
                )

                cY = int(
                    moments['m01'] /
                    moments['m00']
                )

                center_ore_list.append(
                    (cX, cY)
                )

                ore_type_list.append(
                    ore_type
                )

        return center_ore_list, ore_type_list

    # =========================================================
    # DEPTH
    # =========================================================

    def get_depth(self, u, v):

        h, w = self.depth_image.shape[:2]

        x0 = max(0, u - 4)
        x1 = min(w, u + 5)

        y0 = max(0, v - 4)
        y1 = min(h, v + 5)

        region = self.depth_image[
            y0:y1,
            x0:x1
        ]

        valid = region[
            np.isfinite(region)
        ]

        valid = valid[
            valid > 0
        ]

        if len(valid) == 0:
            return None

        return float(
            np.median(valid)
        )

    # =========================================================
    # CAMERA XYZ
    # =========================================================

    def camera_xyz(self, u, v, z):

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]

        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return x, y, z

    # =========================================================
    # CAMERA → BASE_LINK
    # =========================================================

    def transform_to_base(
        self,
        x,
        y,
        z,
        source_frame
    ):

        point_camera = PointStamped()

        point_camera.header.frame_id = source_frame
        point_camera.header.stamp = Time().to_msg()

        point_camera.point.x = x
        point_camera.point.y = y
        point_camera.point.z = z

        try:

            transform = self.tf_buffer.lookup_transform(
                TARGET_FRAME,
                source_frame,
                Time()
            )

            point_base = do_transform_point(
                point_camera,
                transform
            )

            return (
                point_base.point.x,
                point_base.point.y,
                point_base.point.z
            )

        except Exception as e:

            self.get_logger().warn(
                f'TF transform failed '
                f'{source_frame} -> {TARGET_FRAME}: {e}'
            )

            return None

    # =========================================================
    # PUBLISH ORE TF
    # =========================================================

    def publish_ore_tf(
        self,
        child_frame,
        x,
        y,
        z
    ):
        z -= ORE_MIDDLE_OFFSET_Z

        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()

        transform.header.frame_id = TARGET_FRAME

        transform.child_frame_id = child_frame

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        # Ore frame has no special orientation yet.
        # Use identity rotation.
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(
            transform
        )

    # =========================================================
    # MAIN PROCESSING
    # =========================================================

    def process(self):

        if (
            self.color_image is None
            or self.depth_image is None
            or self.camera_info is None
        ):
            return

        display_image = self.color_image.copy()

        centers, ore_types = self.detect_ores(
            display_image
        )

        # -----------------------------------------------------
        # Draw detections
        # -----------------------------------------------------

        for (u, v), ore_type in zip(
            centers,
            ore_types
        ):

            cv2.circle(
                display_image,
                (u, v),
                8,
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display_image,
                ore_type,
                (u + 10, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        cv2.imshow(
            'Task 1A - Ore Detection',
            display_image
        )

        cv2.waitKey(1)

        if not centers:

            self.get_logger().warn(
                'No ores detected.'
            )

            return

        # -----------------------------------------------------
        # Convert every detection to base_link
        # -----------------------------------------------------

        detected_ores = []

        for (u, v), ore_type in zip(
            centers,
            ore_types
        ):

            z = self.get_depth(
                u,
                v
            )

            if z is None:

                self.get_logger().warn(
                    f'{ore_type}: '
                    f'no valid depth at '
                    f'pixel=({u},{v})'
                )

                continue

            x, y, z = self.camera_xyz(
                u,
                v,
                z
            )

            base_xyz = self.transform_to_base(
                x,
                y,
                z,
                self.color_frame_id
            )

            if base_xyz is None:
                continue

            bx, by, bz = base_xyz

            detected_ores.append(
                {
                    'type': ore_type,
                    'pixel': (u, v),
                    'camera_xyz': (x, y, z),
                    'base_xyz': (bx, by, bz)
                }
            )

        # -----------------------------------------------------
        # Group by ore type
        # -----------------------------------------------------

        ore_groups = {
            'azurite_ore': [],
            'malachite_ore': [],
            'vanadinite_ore': []
        }

        for ore in detected_ores:

            ore_groups[
                ore['type']
            ].append(ore)

        # -----------------------------------------------------
        # Assign _1 and _2
        #
        # Sort by base_link X.
        # Smaller X → _1
        # Larger X → _2
        # -----------------------------------------------------

        for ore_type, ores in ore_groups.items():

            ores.sort(
                key=lambda item:
                item['base_xyz'][0]
            )

            for index, ore in enumerate(
                ores[:2],
                start=1
            ):

                bx, by, bz = ore['base_xyz']

                child_frame = (
                    f'{ore_type}_{index}'
                )

                self.publish_ore_tf(
                    child_frame,
                    bx,
                    by,
                    bz
                )

                self.get_logger().info(
                    f'Published {child_frame}: '
                    f'({bx:.3f}, '
                    f'{by:.3f}, '
                    f'{bz - ORE_MIDDLE_OFFSET_Z:.3f}) m'
                )

            if len(ores) != 2:

                self.get_logger().warn(
                    f'{ore_type}: '
                    f'expected 2 detections, '
                    f'got {len(ores)}'
                )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = OreDetector()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
