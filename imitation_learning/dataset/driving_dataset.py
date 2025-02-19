import torch
from torch.utils.data import Dataset, DataLoader
from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
from webdataset import WebDataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

class DrivingDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 sequence_length: int = 50,
                 transform: Optional[callable] = None):
        """
        Initialize driving dataset
        
        Args:
            data_path: Path to dataset
            split: Data split ('train', 'val', 'test')
            sequence_length: Length of trajectory sequences
            transform: Optional transform function
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load dataset index
        self.index = self._load_index()
        
    def _load_index(self) -> List[Dict]:
        """Load dataset index file"""
        index_path = self.data_path / f'{self.split}_index.json'
        with open(index_path, 'r') as f:
            return json.load(f)
            
    def __len__(self) -> int:
        return len(self.index)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - states: Vehicle states [sequence_length, state_dim]
                - actions: Control inputs [sequence_length, control_dim]
                - images: Camera images [sequence_length, C, H, W]
                - point_clouds: LiDAR point clouds [sequence_length, N, 4]
        """
        # Load sequence data
        sequence_info = self.index[idx]
        sequence_path = self.data_path / sequence_info['path']
        
        # Load states and actions
        states = np.load(sequence_path / 'states.npy')
        actions = np.load(sequence_path / 'actions.npy')
        
        # Load sensor data
        images = np.load(sequence_path / 'images.npy')
        point_clouds = np.load(sequence_path / 'lidar.npy')
        
        # Convert to tensors
        data = {
            'states': torch.from_numpy(states).float(),
            'actions': torch.from_numpy(actions).float(),
            'images': torch.from_numpy(images).float(),
            'point_clouds': torch.from_numpy(point_clouds).float()
        }
        
        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
            
        return data

class PetastormDrivingDataset:
    def __init__(self,
                 parquet_path: str,
                 batch_size: int = 32,
                 sequence_length: int = 50,
                 num_workers: int = 4):
        """
        Initialize Petastorm dataset for efficient data loading
        
        Args:
            parquet_path: Path to Parquet dataset
            batch_size: Batch size
            sequence_length: Length of trajectory sequences
            num_workers: Number of worker processes
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        
    def make_dataloader(self) -> PetastormDataLoader:
        """Create Petastorm dataloader"""
        with make_reader(self.parquet_path) as reader:
            return PetastormDataLoader(
                reader,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
            
class WebDrivingDataset(WebDataset):
    def __init__(self,
                 urls: str,
                 batch_size: int = 32,
                 sequence_length: int = 50,
                 transform: Optional[callable] = None):
        """
        Initialize WebDataset for cloud storage
        
        Args:
            urls: URLs/paths to dataset shards
            batch_size: Batch size
            sequence_length: Length of trajectory sequences
            transform: Optional transform function
        """
        super().__init__(urls)
        
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Define preprocessing pipeline
        self.pipeline = (
            self.shuffle(1000)
            .decode('torch')
            .to_tuple('states.pth', 'actions.pth', 'images.pth', 'lidar.pth')
        )
        
        if transform is not None:
            self.pipeline = self.pipeline.map(transform)
            
    def make_dataloader(self) -> DataLoader:
        """Create WebDataset dataloader"""
        return DataLoader(
            self.pipeline,
            batch_size=self.batch_size,
            num_workers=4
        )

class DataTransform:
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 num_points: int = 2048,
                 normalize: bool = True):
        """
        Initialize data transform
        
        Args:
            image_size: Target image size
            num_points: Number of points to sample from point cloud
            normalize: Whether to normalize data
        """
        self.image_size = image_size
        self.num_points = num_points
        self.normalize = normalize
        
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to data
        
        Args:
            data: Dictionary of tensors
            
        Returns:
            Transformed data
        """
        # Resize images
        B, T, C, H, W = data['images'].shape
        images = data['images'].view(-1, C, H, W)
        images = torch.nn.functional.interpolate(
            images,
            size=self.image_size,
            mode='bilinear',
            align_corners=True
        )
        data['images'] = images.view(B, T, C, *self.image_size)
        
        # Sample point clouds
        if data['point_clouds'].shape[2] > self.num_points:
            indices = torch.randperm(data['point_clouds'].shape[2])[:self.num_points]
            data['point_clouds'] = data['point_clouds'][:, :, indices]
            
        # Normalize data
        if self.normalize:
            data['states'] = (data['states'] - data['states'].mean()) / data['states'].std()
            data['actions'] = (data['actions'] - data['actions'].mean()) / data['actions'].std()
            data['images'] = data['images'] / 255.0
            
        return data 