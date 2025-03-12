import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import os
import sys
import logging
from pathlib import Path

# Import base inference engine
sys.path.append(str(Path(__file__).parent.parent))
from inference_engine import InferenceEngine, InferenceConfig

# Import quantization module
sys.path.append(str(Path(__file__).parent.parent.parent))
from optimization.quantization.dynamic_quant import DynamicQuantConfig, DynamicQuantizer, DynamicQuantWrapper

@dataclass
class QuantizedInferenceConfig(InferenceConfig):
    """Configuration for quantized inference engine"""
    quant_config: DynamicQuantConfig = DynamicQuantConfig()
    backend: str = "pytorch"  # ["pytorch", "tensorrt", "triton"]
    cache_dir: str = "quantized_models"
    enable_fusion: bool = True
    enable_jit: bool = True

class QuantizedInferenceEngine(InferenceEngine):
    """Inference engine for quantized models"""
    
    def __init__(self, model: nn.Module, config: QuantizedInferenceConfig):
        """
        Initialize quantized inference engine
        
        Args:
            model: PyTorch model
            config: Quantized inference configuration
        """
        super().__init__(model, config)
        self.config = config  # Override with quantized specific config
        self.quantized_model = None
        self.backend_engine = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
    def optimize(self) -> None:
        """Optimize model for inference using quantization"""
        # First, apply dynamic quantization
        self._apply_quantization()
        
        # Then, apply backend-specific optimizations
        if self.config.backend == "tensorrt":
            self._optimize_tensorrt()
        elif self.config.backend == "triton":
            self._optimize_triton()
        else:  # pytorch
            self._optimize_pytorch()
            
        self.is_optimized = True
        
    def _apply_quantization(self) -> None:
        """Apply dynamic quantization to the model"""
        # Create quantization wrapper
        quant_wrapper = DynamicQuantWrapper(self.model, self.config.quant_config)
        
        # Check if we have a cached quantized model
        model_name = type(self.model).__name__
        quant_model_path = os.path.join(
            self.config.cache_dir, 
            f"{model_name}_quant_{self.config.quant_config.dtype}.pt"
        )
        
        if os.path.exists(quant_model_path):
            logging.info(f"Loading quantized model from {quant_model_path}")
            quant_wrapper.load_state_dict(torch.load(quant_model_path))
            self.quantized_model = quant_wrapper
        else:
            logging.info("Quantized model not found in cache. Please calibrate the model first.")
            self.quantized_model = quant_wrapper
            
    def calibrate(self, calibration_loader: torch.utils.data.DataLoader) -> None:
        """
        Calibrate and quantize the model
        
        Args:
            calibration_loader: DataLoader for calibration data
        """
        if self.quantized_model is None:
            self._apply_quantization()
            
        # Calibrate and quantize
        self.quantized_model.calibrate_and_quantize(calibration_loader)
        
        # Save quantized model
        model_name = type(self.model).__name__
        quant_model_path = os.path.join(
            self.config.cache_dir, 
            f"{model_name}_quant_{self.config.quant_config.dtype}.pt"
        )
        
        torch.save(self.quantized_model.state_dict(), quant_model_path)
        logging.info(f"Saved quantized model to {quant_model_path}")
        
    def _optimize_pytorch(self) -> None:
        """Apply PyTorch-specific optimizations"""
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before optimization")
            
        model = self.quantized_model
        
        # Apply fusion optimizations if enabled
        if self.config.enable_fusion:
            model = torch.quantization.fuse_modules(model, [["conv", "bn", "relu"]])
            
        # Apply JIT compilation if enabled
        if self.config.enable_jit:
            # Create example inputs for tracing
            example_inputs = self._create_example_inputs()
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_inputs)
                traced_model = torch.jit.freeze(traced_model)
                
            self.backend_engine = traced_model
        else:
            self.backend_engine = model
            
    def _optimize_tensorrt(self) -> None:
        """Apply TensorRT optimizations to the quantized model"""
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before optimization")
            
        # Import TensorRT engine
        from inference.tensorrt.tensorrt_engine import TensorRTEngine, TensorRTConfig
        
        # Create TensorRT config
        trt_config = TensorRTConfig(
            batch_size=self.config.batch_size,
            device=self.config.device,
            precision=self.config.precision,
            int8_mode=self.config.quant_config.dtype == torch.qint8,
            fp16_mode=self.config.precision == "fp16",
            max_workspace_size=self.config.max_workspace_size or (1 << 30),
            cache_dir=os.path.join(self.config.cache_dir, "tensorrt")
        )
        
        # Create TensorRT engine
        trt_engine = TensorRTEngine(self.quantized_model, trt_config)
        trt_engine.optimize()
        
        self.backend_engine = trt_engine
        
    def _optimize_triton(self) -> None:
        """Apply Triton optimizations to the quantized model"""
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before optimization")
            
        # Import Triton engine
        from inference.triton.triton_engine import TritonEngine, TritonConfig
        
        # Create Triton config
        triton_config = TritonConfig(
            batch_size=self.config.batch_size,
            device=self.config.device,
            precision=self.config.precision,
            use_fp16=self.config.precision == "fp16",
            cache_dir=os.path.join(self.config.cache_dir, "triton")
        )
        
        # Create Triton engine
        triton_engine = TritonEngine(self.quantized_model, triton_config)
        triton_engine.optimize()
        
        self.backend_engine = triton_engine
        
    def _create_example_inputs(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Create example inputs for tracing/optimization
        
        Returns:
            Example inputs
        """
        # This is a placeholder - in a real implementation, you would need to
        # determine the input shape based on the model's expected input
        return torch.randn(self.config.batch_size, 3, 224, 224, device=self.config.device)
        
    def infer(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """
        Run inference using the quantized model
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before inference")
            
        # If we're using a backend engine, delegate to it
        if self.backend_engine is not None:
            if isinstance(self.backend_engine, InferenceEngine):
                return self.backend_engine.infer(inputs)
            else:
                with torch.no_grad():
                    if isinstance(inputs, dict):
                        return self.backend_engine(**inputs)
                    else:
                        return self.backend_engine(inputs)
        else:
            # Fall back to the quantized model
            with torch.no_grad():
                if isinstance(inputs, dict):
                    return self.quantized_model(**inputs)
                else:
                    return self.quantized_model(inputs)
                    
    def get_size_stats(self) -> Dict[str, float]:
        """
        Get model size statistics
        
        Returns:
            Dictionary with size statistics
        """
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before getting size stats")
            
        if hasattr(self.quantized_model, "quantizer"):
            return self.quantized_model.quantizer.get_size_stats()
        else:
            # Fallback implementation
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
        
    def profile_performance(self, 
                          sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
                          num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile model performance including latency, throughput, and memory usage
        
        Args:
            sample_input: Sample input for profiling
            num_iterations: Number of iterations to profile
            
        Returns:
            Dictionary with profiling results
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before profiling")
            
        results = {
            'latencies_ms': [],
            'throughputs': [],
            'memory_usage_mb': []
        }
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.infer(sample_input)
                
        # Profile
        for _ in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record memory before
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            
            start_event.record()
            with torch.no_grad():
                _ = self.infer(sample_input)
            end_event.record()
            
            # Wait for completion
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            mem_after = torch.cuda.memory_allocated()
            
            results['latencies_ms'].append(latency)
            results['throughputs'].append(1000 / latency)  # Convert to items/second
            results['memory_usage_mb'].append((mem_after - mem_before) / (1024 * 1024))
            
        # Compute statistics
        return {
            'mean_latency_ms': float(np.mean(results['latencies_ms'])),
            'p95_latency_ms': float(np.percentile(results['latencies_ms'], 95)),
            'mean_throughput': float(np.mean(results['throughputs'])),
            'peak_throughput': float(np.max(results['throughputs'])),
            'mean_memory_mb': float(np.mean(results['memory_usage_mb'])),
            'peak_memory_mb': float(np.max(results['memory_usage_mb']))
        }
        
    def export_model(self, 
                    format: str = 'torchscript',
                    example_inputs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
                    optimize: bool = True) -> None:
        """
        Export the quantized model to various formats
        
        Args:
            format: Export format ('torchscript', 'onnx')
            example_inputs: Example inputs for tracing
            optimize: Whether to optimize the exported model
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before exporting")
            
        example_inputs = example_inputs or self._create_example_inputs()
        model_name = type(self.model).__name__
        
        if format == 'torchscript':
            with torch.no_grad():
                # Trace the model
                traced_model = torch.jit.trace(self.quantized_model, example_inputs)
                
                if optimize:
                    # Apply TorchScript optimizations
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    
                # Save the model
                export_path = os.path.join(
                    self.config.cache_dir,
                    f"{model_name}_quantized_scripted.pt"
                )
                traced_model.save(export_path)
                logging.info(f"Exported TorchScript model to {export_path}")
                
        elif format == 'onnx':
            export_path = os.path.join(
                self.config.cache_dir,
                f"{model_name}_quantized.onnx"
            )
            
            # Export to ONNX
            torch.onnx.export(
                self.quantized_model,
                example_inputs,
                export_path,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            if optimize:
                try:
                    import onnxoptimizer
                    import onnx
                    
                    # Load and optimize model
                    model = onnx.load(export_path)
                    optimized_model = onnxoptimizer.optimize(model)
                    
                    # Save optimized model
                    onnx.save(optimized_model, export_path)
                    logging.info(f"Exported optimized ONNX model to {export_path}")
                except ImportError:
                    logging.warning("onnxoptimizer not found. Skipping ONNX optimization.")
                    logging.info(f"Exported ONNX model to {export_path}")
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def profile_memory(self) -> Dict[str, float]:
        """
        Get detailed memory statistics
        
        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {}
            
        torch.cuda.synchronize()
        
        stats = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'max_cached_mb': torch.cuda.max_memory_reserved() / (1024 * 1024)
        }
        
        # Get per-device statistics
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                stats[f'device_{device}_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats[f'device_{device}_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                
        return stats
        
    def profile_layers(self) -> Dict[str, Dict[str, float]]:
        """
        Profile performance of individual layers
        
        Returns:
            Dictionary with layer-wise performance statistics
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before profiling")
            
        layer_stats = {}
        hooks = []
        
        def hook_fn(name):
            def _hook(module, inputs, outputs):
                if name not in layer_stats:
                    layer_stats[name] = []
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = module(*inputs)
                end.record()
                torch.cuda.synchronize()
                layer_stats[name].append(start.elapsed_time(end))
                return result
            return _hook
            
        # Register hooks
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        # Run profiling
        example_input = self._create_example_inputs()
        for _ in range(100):  # Number of profiling iterations
            with torch.no_grad():
                _ = self.infer(example_input)
                
        # Process results
        processed_stats = {}
        for name, timings in layer_stats.items():
            processed_stats[name] = {
                'mean_time_ms': float(np.mean(timings)),
                'std_time_ms': float(np.std(timings)),
                'min_time_ms': float(np.min(timings)),
                'max_time_ms': float(np.max(timings)),
                'p95_time_ms': float(np.percentile(timings, 95))
            }
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return processed_stats
        
    def optimize_memory_format(self) -> None:
        """Optimize memory format for better performance"""
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before memory format optimization")
            
        # Convert to channels_last memory format for better performance on modern GPUs
        self.quantized_model = self.quantized_model.to(memory_format=torch.channels_last)
        
        # If using CUDA, pin memory for faster CPU->GPU transfer
        if self.config.device.startswith('cuda'):
            for param in self.quantized_model.parameters():
                if param.data.pin_memory:
                    param.data = param.data.pin_memory()
                    
    def fuse_modules(self) -> None:
        """Fuse modules for better performance"""
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before module fusion")
            
        # Fuse batch norm with convolution
        self.quantized_model = torch.nn.utils.fusion.fuse_conv_bn_eval(self.quantized_model)
        
        # Fuse other common patterns
        patterns = [
            ['conv', 'bn', 'relu'],
            ['conv', 'relu'],
            ['linear', 'relu']
        ]
        
        for pattern in patterns:
            try:
                self.quantized_model = torch.quantization.fuse_modules(
                    self.quantized_model, 
                    [pattern]
                )
            except Exception as e:
                logging.warning(f"Failed to fuse pattern {pattern}: {e}")
                
    def enable_cudagraphs(self) -> None:
        """Enable CUDA Graphs for static input shapes"""
        if not torch.cuda.is_available():
            return
            
        if self.quantized_model is None:
            raise RuntimeError("Model must be quantized before enabling CUDA Graphs")
            
        # Wrap the forward method with CUDA Graphs
        original_forward = self.quantized_model.forward
        
        def cuda_graphs_forward(*args, **kwargs):
            # Static input shapes are required for CUDA Graphs
            static_input = True
            for arg in args:
                if isinstance(arg, torch.Tensor) and not hasattr(self, 'static_input_size'):
                    self.static_input_size = arg.size()
                elif isinstance(arg, torch.Tensor) and arg.size() != self.static_input_size:
                    static_input = False
                    
            if static_input and hasattr(self, 'cuda_graph'):
                # Replay the CUDA Graph if input shapes match
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        self.static_inputs[i].copy_(arg)
                torch.cuda.synchronize()
                self.cuda_graph.replay()
                return self.static_outputs[0]
            else:
                # Capture a new CUDA Graph or run without it
                if static_input:
                    self.static_inputs = [arg.clone() for arg in args if isinstance(arg, torch.Tensor)]
                    
                    # Warmup
                    for _ in range(3):
                        original_forward(*args, **kwargs)
                        
                    # Capture
                    self.cuda_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.cuda_graph):
                        self.static_outputs = [original_forward(*args, **kwargs)]
                    return self.static_outputs[0]
                else:
                    return original_forward(*args, **kwargs)
                    
        # Replace the forward method
        self.quantized_model.forward = cuda_graphs_forward 