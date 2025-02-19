import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CalibrationParams:
    """Sensor calibration parameters"""
    # LiDAR to camera transform
    T_lidar_camera: np.ndarray  # [4, 4] homogeneous transformation
    camera_matrix: np.ndarray   # [3, 3] intrinsic matrix
    distortion_coeffs: np.ndarray  # [5] distortion parameters
    
    # Radar to LiDAR transform
    T_radar_lidar: np.ndarray  # [4, 4] homogeneous transformation

class SensorFusionModule(nn.Module):
    def __init__(self, 
                 calib: CalibrationParams,
                 device: Optional[torch.device] = None):
        """
        Initialize sensor fusion module
        
        Args:
            calib: Calibration parameters
            device: Torch device
        """
        super().__init__()
        self.calib = calib
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert calibration to tensors
        self.T_lidar_camera = torch.from_numpy(calib.T_lidar_camera).float().to(self.device)
        self.camera_matrix = torch.from_numpy(calib.camera_matrix).float().to(self.device)
        self.T_radar_lidar = torch.from_numpy(calib.T_radar_lidar).float().to(self.device)
        
    def project_lidar_to_camera(self,
                               points: torch.Tensor,
                               image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project LiDAR points to camera image
        
        Args:
            points: LiDAR points [N, 4] (x, y, z, intensity)
            image_size: (height, width) of camera image
            
        Returns:
            projected_points: [M, 2] projected pixel coordinates
            valid_mask: [M] boolean mask for valid projections
        """
        # Transform points to camera frame
        points_hom = torch.cat([points[:, :3], torch.ones_like(points[:, :1])], dim=1)
        points_cam = (self.T_lidar_camera @ points_hom.T).T
        
        # Project to image plane
        points_2d = (self.camera_matrix @ points_cam[:, :3].T).T
        pixels = points_2d[:, :2] / points_2d[:, 2:3]
        
        # Check valid projections
        valid_mask = (points_cam[:, 2] > 0) & \
                    (pixels[:, 0] >= 0) & (pixels[:, 0] < image_size[1]) & \
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size[0])
                    
        return pixels, valid_mask
        
    def transform_radar_to_lidar(self, radar_points: torch.Tensor) -> torch.Tensor:
        """
        Transform radar detections to LiDAR frame
        
        Args:
            radar_points: [N, 7] (x, y, z, vx, vy, vz, rcs)
            
        Returns:
            Transformed points in LiDAR frame
        """
        # Transform positions
        points_hom = torch.cat([radar_points[:, :3], torch.ones_like(radar_points[:, :1])], dim=1)
        points_lidar = (self.T_radar_lidar @ points_hom.T).T
        
        # Transform velocities
        vel_hom = torch.cat([radar_points[:, 3:6], torch.zeros_like(radar_points[:, :1])], dim=1)
        vel_lidar = (self.T_radar_lidar @ vel_hom.T).T
        
        # Combine with RCS
        return torch.cat([points_lidar[:, :3], vel_lidar[:, :3], radar_points[:, 6:]], dim=1)
        
    def fuse_measurements(self,
                         lidar_points: torch.Tensor,
                         camera_image: torch.Tensor,
                         radar_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fuse LiDAR, camera and radar measurements
        
        Args:
            lidar_points: [N, 4] LiDAR points
            camera_image: [H, W, 3] RGB image
            radar_points: [M, 7] radar detections
            
        Returns:
            Dictionary of fused data
        """
        # Project LiDAR to image
        image_size = camera_image.shape[:2]
        pixels, valid_mask = self.project_lidar_to_camera(lidar_points, image_size)
        
        # Get colors for valid LiDAR points
        valid_pixels = pixels[valid_mask].long()
        point_colors = camera_image[valid_pixels[:, 1], valid_pixels[:, 0]]
        
        # Transform radar to LiDAR frame
        radar_lidar = self.transform_radar_to_lidar(radar_points)
        
        # Combine point clouds
        lidar_features = torch.cat([
            lidar_points[valid_mask],
            point_colors,
            torch.zeros_like(lidar_points[valid_mask, :3])  # zero velocity
        ], dim=1)
        
        fused_points = torch.cat([lidar_features, radar_lidar], dim=0)
        
        return {
            'fused_points': fused_points,
            'valid_lidar_mask': valid_mask,
            'projected_pixels': pixels
        }
        
class TemporalFusion(nn.Module):
    def __init__(self,
                 window_size: int = 10,
                 motion_compensation: bool = True):
        """
        Initialize temporal fusion module
        
        Args:
            window_size: Number of frames to fuse
            motion_compensation: Whether to compensate for ego-motion
        """
        super().__init__()
        self.window_size = window_size
        self.motion_compensation = motion_compensation
        
        # Buffer for past measurements
        self.measurement_buffer = []
        self.pose_buffer = []
        
    def update(self, 
               current_points: torch.Tensor,
               current_pose: torch.Tensor):
        """
        Update fusion with new measurement
        
        Args:
            current_points: [N, D] current point cloud
            current_pose: [4, 4] current pose matrix
        """
        if self.motion_compensation:
            # Transform points to world frame
            points_hom = torch.cat([
                current_points[:, :3],
                torch.ones_like(current_points[:, :1])
            ], dim=1)
            points_world = (current_pose @ points_hom.T).T
            
            # Store world frame points
            self.measurement_buffer.append(torch.cat([
                points_world[:, :3],
                current_points[:, 3:]
            ], dim=1))
        else:
            self.measurement_buffer.append(current_points)
            
        self.pose_buffer.append(current_pose)
        
        # Remove old measurements
        if len(self.measurement_buffer) > self.window_size:
            self.measurement_buffer.pop(0)
            self.pose_buffer.pop(0)
            
    def get_fused_points(self) -> torch.Tensor:
        """Get temporally fused point cloud"""
        if not self.measurement_buffer:
            return None
            
        if self.motion_compensation:
            # Transform all points to current frame
            current_pose_inv = torch.inverse(self.pose_buffer[-1])
            
            transformed_points = []
            for points in self.measurement_buffer:
                points_hom = torch.cat([
                    points[:, :3],
                    torch.ones_like(points[:, :1])
                ], dim=1)
                points_current = (current_pose_inv @ points_hom.T).T
                
                transformed_points.append(torch.cat([
                    points_current[:, :3],
                    points[:, 3:]
                ], dim=1))
                
            return torch.cat(transformed_points, dim=0)
        else:
            return torch.cat(self.measurement_buffer, dim=0) 