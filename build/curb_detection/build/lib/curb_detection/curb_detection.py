import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import pcl

class CurbDetector(Node):

    def __init__(self):
        super().__init__('curb_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/your_pointcloud_topic',  # 修改为你实际的点云话题
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/curb_points', 10)

    def pointcloud_callback(self, msg):
        # 将 ROS2 点云消息转换为 Numpy 数组
        point_cloud_numpy = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"))))

        # 使用 PCL 进行处理
        pc = pcl.PointCloud(point_cloud_numpy)
        # 你可以在这里添加一些降噪滤波算法
        # 执行路沿检测算法
        curb_points_np = self.detect_curb(pc).to_array()
        
        # 将路沿点云从 Numpy 数组转换为 ROS2 消息
        curb_msg = self.numpy_to_pointcloud2(curb_points_np, msg.header.frame_id)
        self.publisher.publish(curb_msg)
        self.get_logger().info(f'Published {len(curb_points_np)} curb points.')


    def detect_curb(self, cloud):
       #  基于简单的Z轴阈值过滤来进行路沿检测
        z_threshold = 0.1 # 假设路沿z值高度高于0.1m
        filtered_cloud = cloud.extract(np.where(cloud.to_array()[:, 2] > z_threshold))
        # TODO: 这里可以替换成更复杂的算法，如平面拟合，或其他基于机器学习的方法
        return filtered_cloud
    
    def numpy_to_pointcloud2(self, numpy_array, frame_id):
        points = []
        for point in numpy_array:
            points.append((point[0], point[1], point[2]))
        msg = pc2.create_cloud_xyz32(rclpy.Time().to_msg(), frame_id, points)
        return msg
        


def main(args=None):
    rclpy.init(args=args)
    curb_detector = CurbDetector()
    rclpy.spin(curb_detector)
    curb_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()