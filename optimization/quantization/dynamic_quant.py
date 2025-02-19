import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DynamicQuantConfig:
    """Configuration for dynamic quantization"""
    dtype: torch.dtype = torch.qint8
    qscheme: torch.qscheme = torch.per_tensor_affine
    reduce_range: bool = True
    calibration_batches: int = 100
    observer_type: str = 'minmax'  # ['minmax', 'histogram']

class DynamicQuantizer:
    def __init__(self,
                 model: nn.Module,
                 config: DynamicQuantConfig):
        """
        Initialize dynamic quantizer
        
        Args:
            model: Neural network model
            config: Quantization configuration
        """
        self.model = model
        self.config = config
        
        # Prepare model for quantization
        self.prepare_model()
        
    def prepare_model(self):
        """Prepare model for dynamic quantization"""
        # Specify quantization configuration
        self.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=self.config.dtype,
                qscheme=self.config.qscheme,
                reduce_range=self.config.reduce_range
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=self.config.dtype,
                qscheme=self.config.qscheme,
                reduce_range=self.config.reduce_range
            )
        )
        
        # Prepare model
        self.quantized_model = torch.quantization.prepare(
            self.model,
            self.qconfig
        )
        
    def calibrate(self, 
                 calibration_loader: torch.utils.data.DataLoader):
        """
        Calibrate quantization parameters
        
        Args:
            calibration_loader: DataLoader for calibration data
        """
        self.quantized_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= self.config.calibration_batches:
                    break
                    
                # Forward pass to collect statistics
                if isinstance(batch, dict):
                    self.quantized_model(**batch)
                else:
                    self.quantized_model(batch)
                    
    def quantize(self) -> nn.Module:
        """
        Convert model to quantized version
        
        Returns:
            Quantized model
        """
        self.quantized_model.eval()
        return torch.quantization.convert(self.quantized_model)
        
    def get_size_stats(self) -> Dict[str, float]:
        """Get model size statistics"""
        def get_model_size(model: nn.Module) -> float:
            """Get model size in MB"""
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024
            
        original_size = get_model_size(self.model)
        quantized_size = get_model_size(self.quantized_model)
        
        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size
        }

class DynamicQuantWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 config: DynamicQuantConfig):
        """
        Wrapper for dynamic quantization during inference
        
        Args:
            model: Neural network model
            config: Quantization configuration
        """
        super().__init__()
        self.model = model
        self.config = config
        
        # Initialize quantizer
        self.quantizer = DynamicQuantizer(model, config)
        
    def calibrate_and_quantize(self,
                             calibration_loader: torch.utils.data.DataLoader):
        """
        Calibrate and quantize model
        
        Args:
            calibration_loader: DataLoader for calibration data
        """
        self.quantizer.calibrate(calibration_loader)
        self.quantized_model = self.quantizer.quantize()
        
    def forward(self, *args, **kwargs):
        """Forward pass with quantized model"""
        if hasattr(self, 'quantized_model'):
            return self.quantized_model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
            
class QuantizationProfiler:
    def __init__(self, model: nn.Module):
        """
        Initialize quantization profiler
        
        Args:
            model: Neural network model
        """
        self.model = model
        
    def profile_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """Profile statistics for each layer"""
        stats = {}
        
        def hook_fn(module, input, output, name):
            if isinstance(input[0], torch.Tensor):
                stats[name] = {
                    'input_range': (input[0].min().item(), input[0].max().item()),
                    'output_range': (output.min().item(), output.max().item()),
                    'input_mean': input[0].mean().item(),
                    'input_std': input[0].std().item(),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item()
                }
                
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )
                
        return stats
        
    def suggest_quantization_config(self,
                                  stats: Dict[str, Dict[str, float]]) -> DynamicQuantConfig:
        """
        Suggest quantization configuration based on statistics
        
        Args:
            stats: Layer statistics
            
        Returns:
            Suggested quantization configuration
        """
        # Analyze value ranges
        max_range = max(
            max(abs(s['input_range'][0]), abs(s['input_range'][1]))
            for s in stats.values()
        )
        
        # Suggest configuration
        config = DynamicQuantConfig()
        
        # Suggest dtype based on range
        if max_range < 127:
            config.dtype = torch.qint8
        else:
            config.dtype = torch.qint32
            
        # Suggest scheme based on distribution
        if any(s['input_std'] > 5.0 for s in stats.values()):
            config.qscheme = torch.per_channel_affine
        else:
            config.qscheme = torch.per_tensor_affine
            
        return config 