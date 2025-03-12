import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import os
import sys
import logging
from pathlib import Path

# Import base inference engine
sys.path.append(str(Path(__file__).parent.parent))
from inference_engine import InferenceEngine, InferenceConfig

@dataclass
class TensorRTConfig(InferenceConfig):
    """Configuration for TensorRT inference engine"""
    fp16_mode: bool = False
    int8_mode: bool = False
    cache_dir: str = "tensorrt_engines"
    max_workspace_size: int = 1 << 30  # 1GB
    strict_type_constraints: bool = False
    builder_optimization_level: int = 3
    dla_core: int = -1  # -1 means not using DLA
    onnx_opset_version: int = 13
    calibrator: Optional[Any] = None  # For INT8 calibration

class TensorRTEngine(InferenceEngine):
    """TensorRT inference engine"""
    
    def __init__(self, model: nn.Module, config: TensorRTConfig):
        """
        Initialize TensorRT inference engine
        
        Args:
            model: PyTorch model
            config: TensorRT configuration
        """
        super().__init__(model, config)
        self.config = config  # Override with TensorRT specific config
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.bindings = []
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
    def optimize(self) -> None:
        """Optimize model for inference using TensorRT"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT and/or PyCUDA not found. Please install them with: "
                "pip install tensorrt pycuda"
            )
            
        logger = trt.Logger(trt.Logger.WARNING)
        
        # Generate a unique engine name based on model architecture and config
        model_name = type(self.model).__name__
        precision = "fp16" if self.config.fp16_mode else "fp32"
        precision = "int8" if self.config.int8_mode else precision
        engine_name = f"{model_name}_{precision}_batch{self.config.batch_size}.engine"
        engine_path = os.path.join(self.config.cache_dir, engine_name)
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            logging.info(f"Loading TensorRT engine from {engine_path}")
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
                
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        else:
            logging.info(f"Building TensorRT engine and saving to {engine_path}")
            
            # Convert model to ONNX
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                self._convert_to_onnx(tmp.name)
                
                # Build TensorRT engine from ONNX
                builder = trt.Builder(logger)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, logger)
                
                with open(tmp.name, "rb") as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            logging.error(f"TensorRT ONNX parser error: {parser.get_error(error)}")
                        raise RuntimeError("Failed to parse ONNX model")
                
                # Configure builder
                config = builder.create_builder_config()
                config.max_workspace_size = self.config.max_workspace_size
                
                if self.config.fp16_mode and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    
                if self.config.int8_mode and builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    if self.config.calibrator:
                        config.int8_calibrator = self.config.calibrator
                        
                if self.config.dla_core >= 0:
                    config.default_device_type = trt.DeviceType.DLA
                    config.DLA_core = self.config.dla_core
                    
                # Set optimization level
                config.builder_optimization_level = self.config.builder_optimization_level
                
                # Build and save engine
                self.engine = builder.build_engine(network, config)
                
                with open(engine_path, "wb") as f:
                    f.write(self.engine.serialize())
                    
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input and output names
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                
        self.is_optimized = True
        
    def _convert_to_onnx(self, onnx_path: str) -> None:
        """
        Convert PyTorch model to ONNX
        
        Args:
            onnx_path: Path to save ONNX model
        """
        import torch.onnx
        
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        # Export model to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
    def _create_dummy_input(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Create dummy input for ONNX export
        
        Returns:
            Dummy input tensor or dictionary of tensors
        """
        # This is a placeholder - in a real implementation, you would need to
        # determine the input shape based on the model's expected input
        return torch.randn(self.config.batch_size, 3, 224, 224, device=self.config.device)
        
    def infer(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """
        Run inference using TensorRT
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before inference")
            
        # Prepare inputs
        if isinstance(inputs, torch.Tensor):
            input_dict = {self.input_names[0]: inputs}
        else:
            input_dict = inputs
            
        # Allocate output buffers
        output_dict = {}
        bindings = []
        
        # Process inputs
        for name in self.input_names:
            tensor = input_dict[name].contiguous()
            if tensor.device.type != "cuda":
                tensor = tensor.cuda()
            bindings.append(tensor.data_ptr())
            
        # Allocate output buffers
        for name in self.output_names:
            shape = tuple(self.context.get_binding_shape(self.engine.get_binding_index(name)))
            dtype = torch.float32  # Default dtype
            output = torch.empty(shape, dtype=dtype, device="cuda")
            output_dict[name] = output
            bindings.append(output.data_ptr())
            
        # Run inference
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        
        # Return outputs
        if len(self.output_names) == 1:
            return output_dict[self.output_names[0]]
        else:
            return output_dict 