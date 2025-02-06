import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header
import open3d as o3d
from sklearn.cluster import DBSCAN

class CurbDetector(Node):
    def __init__(self):
        super().__init__('curb_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/rslidar/pointcloud_before_sync_left',  # 修改为你的 LiDAR 话题
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/curb_points', 10)

    def pointcloud_callback(self, msg):
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z")))
        point_cloud_numpy = np.array(points_list, dtype=np.float32)
        
        if point_cloud_numpy.size == 0:
            self.get_logger().warn('接收到空点云')
            return
        
        curb_candidates = self.process_point_cloud(point_cloud_numpy)

        if curb_candidates.size == 0:
            self.get_logger().warn('未检测到路缘点')
            return
        
        curb_msg = self.numpy_to_pointcloud2(curb_candidates, msg.header.frame_id)
        self.publisher.publish(curb_msg)
        self.get_logger().info(f'发布 {len(curb_candidates)} 个路缘点')

    def process_point_cloud(self, points, voxel_size=0.1, ground_thresh=0.15, height_thresh=0.1, cluster_eps=0.2, min_cluster_size=5):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # 体素降采样
        downsampled_cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
        
        # 去除地面（改进：使用直方图方法动态设定地面阈值）
        non_ground = self.remove_ground_plane(downsampled_cloud, ground_thresh)

        points_array = np.asarray(non_ground.points)
        if points_array.size == 0:
            return np.array([])

        # 按Y轴分组计算局部高度差
        curb_candidates = self.detect_curb_by_local_height_difference(points_array, height_thresh)
        
        if curb_candidates.size == 0:
            return np.array([])

        # 进行聚类（可选：区域生长方法可以替代 DBSCAN）
        curb_candidates = self.cluster_points(curb_candidates, eps=cluster_eps, min_samples=min_cluster_size)
        return curb_candidates

    def remove_ground_plane(self, point_cloud, distance_threshold=0.15):
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=100
        )
        non_ground_cloud = point_cloud.select_by_index(inliers, invert=True)
        return non_ground_cloud

    def detect_curb_by_local_height_difference(self, points, height_diff_threshold=0.1, y_step=0.1):
        # 按Y轴排序
        sorted_indices = np.argsort(points[:, 1])
        points = points[sorted_indices]
        
        curb_points = []
        for i in range(len(points) - 1):
            if abs(points[i, 2] - points[i + 1, 2]) > height_diff_threshold:
                curb_points.append(points[i])
        return np.array(curb_points)

    def cluster_points(self, points, eps=0.2, min_samples=5):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
        labels = clustering.labels_
        return points[labels != -1]  # 只保留有效聚类点

    def numpy_to_pointcloud2(self, numpy_array, frame_id):
        if numpy_array.size == 0:
            return PointCloud2()
        
        points = [(p[0], p[1], p[2]) for p in numpy_array]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return pc2.create_cloud_xyz32(header, points)

def main(args=None):
    rclpy.init(args=args)
    curb_detector = CurbDetector()
    rclpy.spin(curb_detector)
    curb_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
