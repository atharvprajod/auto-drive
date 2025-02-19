import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from scipy import stats

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42
    z_score_threshold: float = 3.0
    mad_threshold: float = 3.5
    temporal_window: int = 10

class SensorAnomalyDetector:
    def __init__(self, config: AnomalyConfig):
        """
        Initialize sensor anomaly detector
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        
        # Initialize detectors
        self.isolation_forest = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state
        )
        
        self.elliptic_envelope = EllipticEnvelope(
            contamination=config.contamination,
            random_state=config.random_state
        )
        
        # Buffer for temporal analysis
        self.buffer = []
        
    def detect_statistical_anomalies(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using statistical methods
        
        Args:
            data: [N, D] sensor measurements
            
        Returns:
            Dictionary of anomaly masks
        """
        # Z-score based detection
        z_scores = stats.zscore(data, axis=0)
        z_score_mask = np.abs(z_scores) > self.config.z_score_threshold
        
        # MAD-based detection
        median = np.median(data, axis=0)
        mad = stats.median_abs_deviation(data, axis=0)
        mad_scores = np.abs(data - median) / mad
        mad_mask = mad_scores > self.config.mad_threshold
        
        # Isolation Forest detection
        if_scores = self.isolation_forest.fit_predict(data)
        if_mask = if_scores == -1
        
        # Robust covariance detection
        try:
            ee_scores = self.elliptic_envelope.fit_predict(data)
            ee_mask = ee_scores == -1
        except:
            ee_mask = np.zeros_like(if_mask)
        
        return {
            'z_score_anomalies': z_score_mask,
            'mad_anomalies': mad_mask,
            'isolation_forest_anomalies': if_mask,
            'elliptic_envelope_anomalies': ee_mask
        }
        
    def detect_temporal_anomalies(self, measurement: np.ndarray) -> Dict[str, bool]:
        """
        Detect anomalies in temporal sequence
        
        Args:
            measurement: Current sensor measurement
            
        Returns:
            Dictionary of temporal anomaly flags
        """
        self.buffer.append(measurement)
        if len(self.buffer) > self.config.temporal_window:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.config.temporal_window:
            return {'is_temporal_anomaly': False}
            
        # Convert buffer to array
        sequence = np.array(self.buffer)
        
        # Compute temporal statistics
        velocity = np.diff(sequence, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Check for sudden changes
        max_velocity = np.max(np.abs(velocity), axis=0)
        max_acceleration = np.max(np.abs(acceleration), axis=0)
        
        # Compute historical statistics
        historical_vel_mean = np.mean(np.abs(velocity[:-1]), axis=0)
        historical_vel_std = np.std(np.abs(velocity[:-1]), axis=0)
        
        historical_acc_mean = np.mean(np.abs(acceleration[:-1]), axis=0)
        historical_acc_std = np.std(np.abs(acceleration[:-1]), axis=0)
        
        # Detect anomalies
        vel_anomaly = np.any(
            np.abs(velocity[-1]) > historical_vel_mean + 3 * historical_vel_std
        )
        
        acc_anomaly = np.any(
            np.abs(acceleration[-1]) > historical_acc_mean + 3 * historical_acc_std
        )
        
        return {
            'is_temporal_anomaly': vel_anomaly or acc_anomaly,
            'velocity_anomaly': vel_anomaly,
            'acceleration_anomaly': acc_anomaly
        }

class DeepAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize deep learning based anomaly detector
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Reconstructed input and latent representation
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
        
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Anomaly scores [batch_size]
        """
        # Get reconstruction
        reconstructed, _ = self.forward(x)
        
        # Compute reconstruction error
        reconstruction_error = torch.mean((x - reconstructed)**2, dim=1)
        
        return reconstruction_error
        
class MultiModalAnomalyDetector:
    def __init__(self,
                 config: AnomalyConfig,
                 feature_dims: Dict[str, int]):
        """
        Initialize multi-modal anomaly detector
        
        Args:
            config: Anomaly detection configuration
            feature_dims: Dictionary of feature dimensions for each sensor
        """
        self.config = config
        
        # Initialize statistical detector
        self.statistical_detector = SensorAnomalyDetector(config)
        
        # Initialize deep detectors for each sensor
        self.deep_detectors = nn.ModuleDict({
            sensor_name: DeepAnomalyDetector(dim)
            for sensor_name, dim in feature_dims.items()
        })
        
        # Initialize temporal detectors
        self.temporal_detectors = {
            sensor_name: SensorAnomalyDetector(config)
            for sensor_name in feature_dims.keys()
        }
        
    def detect_anomalies(self,
                        measurements: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Detect anomalies in multi-modal measurements
        
        Args:
            measurements: Dictionary of sensor measurements
            
        Returns:
            Dictionary of anomaly detection results
        """
        results = {}
        
        for sensor_name, measurement in measurements.items():
            # Statistical detection
            statistical_results = self.statistical_detector.detect_statistical_anomalies(
                measurement.numpy()
            )
            
            # Deep detection
            deep_scores = self.deep_detectors[sensor_name].compute_anomaly_score(
                measurement
            )
            
            # Temporal detection
            temporal_results = self.temporal_detectors[sensor_name].detect_temporal_anomalies(
                measurement.numpy()
            )
            
            results[sensor_name] = {
                **statistical_results,
                'deep_anomaly_scores': deep_scores,
                **temporal_results
            }
            
        return results 