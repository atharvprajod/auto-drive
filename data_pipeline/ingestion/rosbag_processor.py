import rospy
import rosbag
import numpy as np
import torch
from typing import Dict, List, Any, Generator
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

@dataclass
class ROSTopicConfig:
    """Configuration for ROS topic processing"""
    name: str
    msg_type: str
    fields: List[str]
    frequency: float

@dataclass
class SensorData:
    """Container for synchronized sensor data"""
    timestamp: float
    lidar: np.ndarray = None  # [N, 4] points (x, y, z, intensity)
    camera: np.ndarray = None  # [H, W, 3] RGB image
    imu: np.ndarray = None    # [6] (acc_xyz, gyro_xyz)
    pose: np.ndarray = None   # [7] (position_xyz, quaternion_wxyz)

class RosbagProcessor:
    def __init__(self, 
                 topic_configs: List[ROSTopicConfig],
                 sync_tolerance: float = 0.1):
        """
        Initialize ROS bag processor
        
        Args:
            topic_configs: List of topic configurations
            sync_tolerance: Time tolerance for message synchronization
        """
        self.topic_configs = {config.name: config for config in topic_configs}
        self.sync_tolerance = sync_tolerance
        
    def process_bag(self, 
                   bag_path: Path,
                   output_dir: Path = None) -> Generator[SensorData, None, None]:
        """
        Process ROS bag file and yield synchronized sensor data
        
        Args:
            bag_path: Path to ROS bag file
            output_dir: Optional path to save processed data
            
        Yields:
            SensorData objects containing synchronized sensor measurements
        """
        # Open bag file
        bag = rosbag.Bag(bag_path)
        
        # Create message buffers for each topic
        msg_buffers = {topic: [] for topic in self.topic_configs.keys()}
        
        # Read messages
        for topic, msg, t in bag.read_messages():
            if topic in self.topic_configs:
                msg_buffers[topic].append((t.to_sec(), msg))
                
                # Try to create synchronized data packet
                sync_data = self._synchronize_messages(msg_buffers)
                if sync_data is not None:
                    if output_dir is not None:
                        self._save_data(sync_data, output_dir)
                    yield sync_data
                    
        bag.close()
        
    def _synchronize_messages(self, 
                            msg_buffers: Dict[str, List[tuple]]) -> SensorData:
        """
        Attempt to synchronize messages from different topics
        
        Args:
            msg_buffers: Dictionary of message buffers for each topic
            
        Returns:
            Synchronized SensorData object if successful, None otherwise
        """
        # Find earliest message
        earliest_times = []
        for topic, msgs in msg_buffers.items():
            if msgs:
                earliest_times.append(msgs[0][0])
                
        if not earliest_times:
            return None
            
        sync_time = min(earliest_times)
        
        # Check if messages can be synchronized
        sync_msgs = {}
        for topic, msgs in msg_buffers.items():
            if not msgs:
                return None
                
            # Find closest message in time
            closest_idx = min(range(len(msgs)), 
                            key=lambda i: abs(msgs[i][0] - sync_time))
            
            # Check time tolerance
            if abs(msgs[closest_idx][0] - sync_time) > self.sync_tolerance:
                return None
                
            sync_msgs[topic] = msgs[closest_idx][1]
            
            # Remove older messages
            msg_buffers[topic] = msgs[closest_idx:]
            
        # Create synchronized data packet
        return self._create_sensor_data(sync_time, sync_msgs)
        
    def _create_sensor_data(self,
                           timestamp: float,
                           messages: Dict[str, Any]) -> SensorData:
        """
        Create SensorData object from synchronized messages
        
        Args:
            timestamp: Synchronized timestamp
            messages: Dictionary of synchronized messages
            
        Returns:
            SensorData object
        """
        data = SensorData(timestamp=timestamp)
        
        # Process each message type
        for topic, msg in messages.items():
            config = self.topic_configs[topic]
            
            if config.msg_type == 'sensor_msgs/PointCloud2':
                data.lidar = self._process_pointcloud(msg)
            elif config.msg_type == 'sensor_msgs/Image':
                data.camera = self._process_image(msg)
            elif config.msg_type == 'sensor_msgs/Imu':
                data.imu = self._process_imu(msg)
            elif config.msg_type == 'geometry_msgs/PoseStamped':
                data.pose = self._process_pose(msg)
                
        return data
        
    def _process_pointcloud(self, msg) -> np.ndarray:
        """Convert PointCloud2 message to numpy array"""
        # Implementation depends on point cloud format
        # This is a simplified example
        points = []
        for p in msg.data:
            points.append([p.x, p.y, p.z, p.intensity])
        return np.array(points)
        
    def _process_image(self, msg) -> np.ndarray:
        """Convert Image message to numpy array"""
        from cv_bridge import CvBridge
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
    def _process_imu(self, msg) -> np.ndarray:
        """Convert IMU message to numpy array"""
        return np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
    def _process_pose(self, msg) -> np.ndarray:
        """Convert PoseStamped message to numpy array"""
        return np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
        
    def _save_data(self, data: SensorData, output_dir: Path):
        """Save synchronized data to disk"""
        timestamp_str = f"{data.timestamp:.6f}"
        save_dir = output_dir / timestamp_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if data.lidar is not None:
            np.save(save_dir / 'lidar.npy', data.lidar)
        if data.camera is not None:
            np.save(save_dir / 'camera.npy', data.camera)
        if data.imu is not None:
            np.save(save_dir / 'imu.npy', data.imu)
        if data.pose is not None:
            np.save(save_dir / 'pose.npy', data.pose) 