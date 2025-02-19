import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class ValidationConfig:
    """Configuration for statistical validation"""
    min_samples: int = 1000
    significance_level: float = 0.05
    distribution_tests: List[str] = None
    correlation_threshold: float = 0.8
    stationarity_window: int = 100

class DataDistributionValidator:
    def __init__(self, config: ValidationConfig):
        """
        Initialize data distribution validator
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.config.distribution_tests = config.distribution_tests or [
            'normality',
            'stationarity',
            'correlation'
        ]
        
        # Initialize statistics buffer
        self.statistics_buffer = {}
        
    def validate_distribution(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Validate data distribution properties
        
        Args:
            data: [N, D] data array
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check sample size
        if len(data) < self.config.min_samples:
            return {'error': 'Insufficient samples'}
            
        # Perform requested tests
        if 'normality' in self.config.distribution_tests:
            results['normality'] = self._test_normality(data)
            
        if 'stationarity' in self.config.distribution_tests:
            results['stationarity'] = self._test_stationarity(data)
            
        if 'correlation' in self.config.distribution_tests:
            results['correlation'] = self._test_correlation(data)
            
        return results
        
    def _test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """Test for normality using multiple methods"""
        results = {}
        
        # Shapiro-Wilk test
        for i in range(data.shape[1]):
            _, p_value = stats.shapiro(data[:, i])
            results[f'shapiro_p_value_{i}'] = p_value
            
        # Anderson-Darling test
        for i in range(data.shape[1]):
            stat, crit_vals, sig_level = stats.anderson(data[:, i])
            results[f'anderson_statistic_{i}'] = stat
            results[f'anderson_critical_values_{i}'] = crit_vals
            
        # D'Agostino's K^2 test
        for i in range(data.shape[1]):
            stat, p_value = stats.normaltest(data[:, i])
            results[f'dagostino_p_value_{i}'] = p_value
            
        return results
        
    def _test_stationarity(self, data: np.ndarray) -> Dict[str, float]:
        """Test for stationarity using rolling statistics"""
        results = {}
        
        # Convert to pandas for rolling statistics
        df = pd.DataFrame(data)
        
        for col in df.columns:
            # Compute rolling statistics
            rolling_mean = df[col].rolling(window=self.config.stationarity_window).mean()
            rolling_std = df[col].rolling(window=self.config.stationarity_window).std()
            
            # Augmented Dickey-Fuller test
            adf_stat, p_value, *_ = stats.adfuller(df[col].dropna())
            
            results[f'adf_statistic_{col}'] = adf_stat
            results[f'adf_p_value_{col}'] = p_value
            
            # Test for trend
            trend_stat, trend_p_value = stats.kendalltau(
                np.arange(len(df[col])),
                df[col].values
            )
            
            results[f'trend_statistic_{col}'] = trend_stat
            results[f'trend_p_value_{col}'] = trend_p_value
            
        return results
        
    def _test_correlation(self, data: np.ndarray) -> Dict[str, float]:
        """Test for correlation between features"""
        results = {}
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
                    
        results['correlation_matrix'] = corr_matrix
        results['high_correlations'] = high_corr_pairs
        
        return results

class DataQualityChecker:
    def __init__(self, config: ValidationConfig):
        """
        Initialize data quality checker
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def check_quality(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive data quality checks
        
        Args:
            data: [N, D] data array
            
        Returns:
            Dictionary of quality metrics
        """
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'missing_values': np.isnan(data).sum(axis=0).tolist()
        }
        
        # Outlier detection
        scaled_data = self.scaler.fit_transform(data)
        z_scores = np.abs(scaled_data)
        results['outliers'] = {
            'num_outliers': (z_scores > 3).sum(axis=0).tolist(),
            'outlier_fraction': ((z_scores > 3).sum(axis=0) / len(data)).tolist()
        }
        
        # Distribution metrics
        results['distribution'] = {
            'skewness': stats.skew(data, axis=0).tolist(),
            'kurtosis': stats.kurtosis(data, axis=0).tolist()
        }
        
        # Entropy and information content
        results['information'] = {}
        for i in range(data.shape[1]):
            hist, _ = np.histogram(data[:, i], bins='auto', density=True)
            results['information'][f'entropy_{i}'] = stats.entropy(hist + 1e-10)
            
        return results
        
    def generate_report(self, data: np.ndarray) -> str:
        """Generate human-readable quality report"""
        quality_results = self.check_quality(data)
        
        report = []
        report.append("Data Quality Report")
        report.append("=" * 50)
        
        # Basic statistics
        report.append("\nBasic Statistics:")
        for stat, values in quality_results['basic_stats'].items():
            report.append(f"{stat}: {values}")
            
        # Outlier information
        report.append("\nOutlier Information:")
        for metric, values in quality_results['outliers'].items():
            report.append(f"{metric}: {values}")
            
        # Distribution characteristics
        report.append("\nDistribution Characteristics:")
        for metric, values in quality_results['distribution'].items():
            report.append(f"{metric}: {values}")
            
        # Information content
        report.append("\nInformation Content:")
        for feature, entropy in quality_results['information'].items():
            report.append(f"{feature}: {entropy:.4f}")
            
        return "\n".join(report)

class TimeSeriesValidator:
    def __init__(self, config: ValidationConfig):
        """
        Initialize time series validator
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
    def validate_time_series(self,
                           timestamps: np.ndarray,
                           values: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Validate time series properties
        
        Args:
            timestamps: [N] timestamp array
            values: [N, D] value array
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check timestamp properties
        time_diffs = np.diff(timestamps)
        results['temporal'] = {
            'mean_sampling_rate': 1.0 / np.mean(time_diffs),
            'std_sampling_rate': np.std(1.0 / time_diffs),
            'max_gap': np.max(time_diffs),
            'missing_timestamps': np.sum(time_diffs > 2 * np.median(time_diffs))
        }
        
        # Check for seasonality
        results['seasonality'] = {}
        for i in range(values.shape[1]):
            f, Pxx = stats.periodogram(values[:, i])
            peak_freq = f[np.argmax(Pxx)]
            results['seasonality'][f'dominant_frequency_{i}'] = peak_freq
            
        # Check for change points
        results['change_points'] = {}
        for i in range(values.shape[1]):
            # Detect mean shifts
            change_points = self._detect_change_points(values[:, i])
            results['change_points'][f'feature_{i}'] = change_points
            
        return results
        
    def _detect_change_points(self, series: np.ndarray) -> List[int]:
        """Detect change points in time series using CUSUM"""
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            return []
            
        # Standardize series
        standardized = (series - mean) / std
        
        # CUSUM calculation
        cusums = np.cumsum(standardized)
        change_points = []
        
        # Detect significant changes
        threshold = 2 * np.sqrt(len(series))
        for i in range(1, len(cusums)-1):
            if abs(cusums[i] - cusums[i-1]) > threshold:
                change_points.append(i)
                
        return change_points 