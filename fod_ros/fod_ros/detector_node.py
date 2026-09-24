import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.bridge = CvBridge()

        # resolve model path relative to the installed package share directory
        model_path = os.path.join(
            get_package_share_directory('fod_ros'), 'models', 'best.onnx'
        )
        self.get_logger().info(f'Loading model from {model_path}')
        self.model = YOLO(model_path)

        self.sub = self.create_subscription(
            Image, 'camera/image_raw', self.callback, 10
        )
        self.det_pub = self.create_publisher(Detection2DArray, 'fod/detections', 10)
        self.img_pub = self.create_publisher(Image, 'fod/annotated', 10)

    def callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, verbose=False)[0]

        det_array = Detection2DArray()
        det_array.header = msg.header

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_id)
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            det_array.detections.append(det)

        self.det_pub.publish(det_array)

        annotated = results.plot()
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
