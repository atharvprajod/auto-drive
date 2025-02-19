import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import interp1d

@dataclass
class TimeSeriesConfig:
    """Configuration for time series alignment"""
    target_frequency: float = 10.0  # Hz
    max_time_diff: float = 0.1     # seconds
    interpolation_method: str = 'linear'
    extrapolation_enabled: bool = False

class TemporalAlignment:
    def __init__(self, config: TimeSeriesConfig):
        """
        Initialize temporal alignment module
        
        Args:
            config: Time series configuration
        """
        self.config = config
        self.dt = 1.0 / config.target_frequency
        
    def align_time_series(self,
                         timestamps: Dict[str, np.ndarray],
                         values: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Align multiple time series to common time base
        
        Args:
            timestamps: Dictionary of timestamps for each sensor
            values: Dictionary of sensor measurements
            
        Returns:
            common_timestamps: Array of aligned timestamps
            aligned_values: Dictionary of aligned measurements
        """
        # Find common time range
        start_time = max(ts[0] for ts in timestamps.values())
        end_time = min(ts[-1] for ts in timestamps.values())
        
        # Create common time base
        num_steps = int((end_time - start_time) / self.dt) + 1
        common_timestamps = np.linspace(start_time, end_time, num_steps)
        
        # Align each time series
        aligned_values = {}
        for sensor_name, sensor_values in values.items():
            aligned_values[sensor_name] = self._interpolate_time_series(
                timestamps[sensor_name],
                sensor_values,
                common_timestamps
            )
            
        return common_timestamps, aligned_values
        
    def _interpolate_time_series(self,
                                timestamps: np.ndarray,
                                values: np.ndarray,
                                target_timestamps: np.ndarray) -> np.ndarray:
        """
        Interpolate time series to target timestamps
        
        Args:
            timestamps: Original timestamps
            values: Original values
            target_timestamps: Target timestamps
            
        Returns:
            Interpolated values
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)
            
        # Create interpolator for each dimension
        interpolated = np.zeros((len(target_timestamps), values.shape[1]))
        
        for i in range(values.shape[1]):
            interpolator = interp1d(
                timestamps,
                values[:, i],
                kind=self.config.interpolation_method,
                bounds_error=not self.config.extrapolation_enabled,
                fill_value='extrapolate' if self.config.extrapolation_enabled else np.nan
            )
            interpolated[:, i] = interpolator(target_timestamps)
            
        return interpolated.squeeze()

class OnlineTemporalAlignment:
    def __init__(self, config: TimeSeriesConfig):
        """
        Initialize online temporal alignment module
        
        Args:
            config: Time series configuration
        """
        self.config = config
        self.dt = 1.0 / config.target_frequency
        
        # Buffer for recent measurements
        self.buffer = {}
        self.latest_timestamp = None
        
    def update(self,
              sensor_name: str,
              timestamp: float,
              value: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Update with new measurement and attempt alignment
        
        Args:
            sensor_name: Name of sensor
            timestamp: Measurement timestamp
            value: Sensor measurement
            
        Returns:
            Dictionary of aligned measurements if successful, None otherwise
        """
        # Initialize buffer for new sensor
        if sensor_name not in self.buffer:
            self.buffer[sensor_name] = {
                'timestamps': [],
                'values': []
            }
            
        # Add measurement to buffer
        self.buffer[sensor_name]['timestamps'].append(timestamp)
        self.buffer[sensor_name]['values'].append(value)
        
        # Update latest timestamp
        if self.latest_timestamp is None or timestamp > self.latest_timestamp:
            self.latest_timestamp = timestamp
            
        # Try to align measurements
        return self._align_buffered_measurements()
        
    def _align_buffered_measurements(self) -> Optional[Dict[str, np.ndarray]]:
        """Attempt to align buffered measurements"""
        if not self.latest_timestamp:
            return None
            
        # Check if all sensors have recent measurements
        target_time = self.latest_timestamp - self.config.max_time_diff
        aligned_values = {}
        
        for sensor_name, sensor_buffer in self.buffer.items():
            # Find measurements within time window
            recent_mask = np.array(sensor_buffer['timestamps']) >= target_time
            if not np.any(recent_mask):
                return None
                
            # Get most recent measurement
            recent_timestamps = np.array(sensor_buffer['timestamps'])[recent_mask]
            recent_values = np.array(sensor_buffer['values'])[recent_mask]
            
            # Interpolate to target time
            interpolator = interp1d(
                recent_timestamps,
                recent_values,
                kind=self.config.interpolation_method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            aligned_values[sensor_name] = interpolator(target_time)
            
            # Remove old measurements
            old_mask = np.array(sensor_buffer['timestamps']) < target_time
            if np.any(old_mask):
                sensor_buffer['timestamps'] = list(np.array(sensor_buffer['timestamps'])[~old_mask])
                sensor_buffer['values'] = list(np.array(sensor_buffer['values'])[~old_mask])
                
        # Check if all sensors are aligned
        if any(np.any(np.isnan(v)) for v in aligned_values.values()):
            return None
            
        return aligned_values

class BatchTemporalAlignment(nn.Module):
    def __init__(self, config: TimeSeriesConfig):
        """
        Initialize batch temporal alignment module
        
        Args:
            config: Time series configuration
        """
        super().__init__()
        self.config = config
        self.dt = 1.0 / config.target_frequency
        
    def forward(self,
                timestamps: torch.Tensor,
                values: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align batch of time series
        
        Args:
            timestamps: [batch_size, num_steps] timestamps
            values: [batch_size, num_steps, feature_dim] values
            mask: [batch_size, num_steps] boolean mask for valid values
            
        Returns:
            aligned_timestamps: [batch_size, num_aligned_steps] aligned timestamps
            aligned_values: [batch_size, num_aligned_steps, feature_dim] aligned values
        """
        batch_size = timestamps.shape[0]
        
        # Find common time range for each sequence
        start_times = []
        end_times = []
        
        for i in range(batch_size):
            valid_timestamps = timestamps[i, mask[i]]
            start_times.append(valid_timestamps[0])
            end_times.append(valid_timestamps[-1])
            
        # Create aligned time base
        max_start = torch.max(torch.tensor(start_times))
        min_end = torch.min(torch.tensor(end_times))
        num_steps = int((min_end - max_start) / self.dt) + 1
        
        aligned_timestamps = torch.linspace(max_start, min_end, num_steps)
        aligned_values = torch.zeros(batch_size, num_steps, values.shape[-1])
        
        # Interpolate each sequence
        for i in range(batch_size):
            valid_timestamps = timestamps[i, mask[i]]
            valid_values = values[i, mask[i]]
            
            # Interpolate each feature dimension
            for j in range(values.shape[-1]):
                aligned_values[i, :, j] = torch.tensor(
                    np.interp(
                        aligned_timestamps.numpy(),
                        valid_timestamps.numpy(),
                        valid_values[:, j].numpy()
                    )
                )
                
        return aligned_timestamps, aligned_values 