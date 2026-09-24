import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.pub = self.create_publisher(Image, 'camera/image_raw', 10)

        self.bridge = CvBridge()  # translator: OpenCV frame <-> ROS Image message

        self.cap = cv2.VideoCapture(0)  # 0 = first webcam; swap for a video file path if none available
        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera source')

        # call self.tick() automatically, ~30 times a second
        self.timer = self.create_timer(1 / 30, self.tick)

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return  # skip this cycle if no frame came through

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
