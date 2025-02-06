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
            '/sensing/lidar/rslidar/pointcloud_before_sync_left',  # Modify to your topic
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/curb_points', 10)

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z")))
        point_cloud_numpy = np.array([(p[0], p[1], p[2]) for p in points_list], dtype=np.float32)

        # Process point cloud
        curb_candidates, labels = self.process_point_cloud(point_cloud_numpy)

        # Convert curb points back to PointCloud2
        curb_msg = self.numpy_to_pointcloud2(curb_candidates, msg.header.frame_id)
        self.publisher.publish(curb_msg)
        self.get_logger().info(f'Published {len(curb_candidates)} curb points.')

    def process_point_cloud(self, points, voxel_size=0.1, ground_dist_thresh=0.1, height_diff_thresh=0.1, cluster_eps=0.2, min_cluster_size=10):
        # Convert numpy array to Open3D point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # Downsample point cloud
        downsampled_cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

        # Remove ground plane
        non_ground, _ = self.remove_ground_plane(downsampled_cloud, distance_threshold=ground_dist_thresh)

        # Detect curbs using height differences
        points_array = np.asarray(non_ground.points)
        curb_candidates = self.detect_curb_by_height_difference(points_array, height_diff_thresh)

        # Cluster curb points
        labels = self.cluster_points(curb_candidates, eps=cluster_eps, min_samples=min_cluster_size)
        return curb_candidates, labels

    def remove_ground_plane(self, point_cloud, distance_threshold=0.1):
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=100
        )
        ground_cloud = point_cloud.select_by_index(inliers)
        non_ground_cloud = point_cloud.select_by_index(inliers, invert=True)
        return non_ground_cloud, ground_cloud

    def detect_curb_by_height_difference(self, points, height_diff_threshold=0.1):
        curb_points = []
        for i in range(len(points) - 1):
            if abs(points[i, 2] - points[i + 1, 2]) > height_diff_threshold:
                curb_points.append(points[i])
        return np.array(curb_points)

    def cluster_points(self, points, eps=0.2, min_samples=10):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
        labels = clustering.labels_
        return labels

    def numpy_to_pointcloud2(self, numpy_array, frame_id):
        points = [(p[0], p[1], p[2]) for p in numpy_array]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        msg = pc2.create_cloud_xyz32(header, points)
        return msg


def main(args=None):
    rclpy.init(args=args)
    curb_detector = CurbDetector()
    rclpy.spin(curb_detector)
    curb_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
