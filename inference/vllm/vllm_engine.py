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

@dataclass
class vLLMConfig(InferenceConfig):
    """Configuration for vLLM inference engine"""
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB
    enforce_eager: bool = False
    trust_remote_code: bool = False
    dtype: str = "auto"  # ["auto", "half", "float16", "bfloat16", "float", "float32"]
    quantization: Optional[str] = None  # ["awq", "gptq", "squeezellm", None]
    seed: int = 0
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    block_size: int = 16

class vLLMEngine(InferenceEngine):
    """vLLM inference engine for efficient LLM inference"""
    
    def __init__(self, model: nn.Module, config: vLLMConfig):
        """
        Initialize vLLM inference engine
        
        Args:
            model: PyTorch model or model name/path
            config: vLLM configuration
        """
        super().__init__(model, config)
        self.config = config  # Override with vLLM specific config
        self.llm_engine = None
        self.sampling_params = None
        
        # Try importing vLLM
        try:
            import vllm
        except ImportError:
            raise ImportError(
                "vLLM not found. Please install it with: "
                "pip install vllm"
            )
            
    def optimize(self) -> None:
        """Initialize and optimize the vLLM engine"""
        import vllm
        from vllm import LLM, SamplingParams
        
        # Get model name or path
        if isinstance(self.model, str):
            model_name_or_path = self.model
        else:
            # If it's a PyTorch model, we need to save it first
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                
                # Save model
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(model_path)
                else:
                    torch.save(self.model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
                    
                model_name_or_path = model_path
                
        # Initialize vLLM engine
        self.llm_engine = LLM(
            model=model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            swap_space=self.config.swap_space,
            enforce_eager=self.config.enforce_eager,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            seed=self.config.seed,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            block_size=self.config.block_size,
        )
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=256
        )
        
        self.is_optimized = True
        
    def infer(self, 
             inputs: Union[str, List[str], Dict[str, Any]], 
             sampling_params: Optional[Any] = None) -> Any:
        """
        Run inference using vLLM
        
        Args:
            inputs: Input prompts or dictionary with prompts and other parameters
            sampling_params: Optional sampling parameters to override defaults
            
        Returns:
            Generated outputs
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before inference")
            
        from vllm import SamplingParams
        
        # Process inputs
        if isinstance(inputs, dict) and "prompts" in inputs:
            prompts = inputs["prompts"]
            custom_sampling_params = inputs.get("sampling_params", None)
        else:
            prompts = inputs
            custom_sampling_params = sampling_params
            
        # Use provided sampling parameters or defaults
        params = custom_sampling_params or self.sampling_params
        
        # Run inference
        outputs = self.llm_engine.generate(prompts, params)
        
        # Process outputs
        if isinstance(prompts, str):
            return outputs[0].outputs[0].text
        else:
            return [output.outputs[0].text for output in outputs]
            
    def stream_infer(self, 
                   inputs: Union[str, List[str], Dict[str, Any]], 
                   sampling_params: Optional[Any] = None) -> Any:
        """
        Stream inference results using vLLM
        
        Args:
            inputs: Input prompts or dictionary with prompts and other parameters
            sampling_params: Optional sampling parameters to override defaults
            
        Returns:
            Generator yielding partial outputs
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before inference")
            
        from vllm import SamplingParams
        
        # Process inputs
        if isinstance(inputs, dict) and "prompts" in inputs:
            prompts = inputs["prompts"]
            custom_sampling_params = inputs.get("sampling_params", None)
        else:
            prompts = inputs
            custom_sampling_params = sampling_params
            
        # Use provided sampling parameters or defaults
        params = custom_sampling_params or self.sampling_params
        
        # Run streaming inference
        for outputs in self.llm_engine.generate(prompts, params, stream=True):
            if isinstance(prompts, str):
                yield outputs[0].outputs[0].text
            else:
                yield [output.outputs[0].text for output in outputs]
                
    def batch_infer(self, 
                  prompts: List[str], 
                  sampling_params: Optional[Any] = None) -> List[str]:
        """
        Run batch inference using vLLM
        
        Args:
            prompts: List of input prompts
            sampling_params: Optional sampling parameters to override defaults
            
        Returns:
            List of generated outputs
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before inference")
            
        # Use provided sampling parameters or defaults
        params = sampling_params or self.sampling_params
        
        # Run batch inference
        outputs = self.llm_engine.generate(prompts, params)
        
        return [output.outputs[0].text for output in outputs]
        
    def set_sampling_params(self, **kwargs) -> None:
        """
        Set default sampling parameters
        
        Args:
            **kwargs: Sampling parameters
        """
        from vllm import SamplingParams
        self.sampling_params = SamplingParams(**kwargs) 