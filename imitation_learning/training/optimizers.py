import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import math

@dataclass
class OptimizerConfig:
    """Configuration for optimizers"""
    optimizer: str = 'adam'  # ['adam', 'adamw', 'lamb', 'adafactor']
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    lr_schedule: str = 'cosine'  # ['constant', 'cosine', 'linear', 'exponential']
    warmup_steps: int = 1000
    decay_steps: int = 10000
    min_lr_ratio: float = 0.1
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None

class LAMB(torch.optim.Optimizer):
    """
    CUDA-optimized Layer-wise Adaptive Moments optimizer
    Designed for better scaling with large batch sizes
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 adam: bool = False):
        """
        Initialize LAMB optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
            adam: Whether to use standard Adam update
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(LAMB, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moments
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute adaptive learning rate
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Compute update
                update = exp_avg / bias_correction1 / denom
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                    
                if self.adam:
                    # Standard Adam update
                    p.data.add_(update, alpha=-group['lr'])
                else:
                    # LAMB update
                    # Compute trust ratio
                    w_norm = p.data.norm(p=2).clamp(min=1e-6)
                    g_norm = update.norm(p=2).clamp(min=1e-6)
                    trust_ratio = w_norm / g_norm
                    
                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)
                    
        return loss

class Adafactor(torch.optim.Optimizer):
    """
    CUDA-optimized Adafactor optimizer
    Memory-efficient alternative to Adam
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 eps: tuple = (1e-30, 1e-3),
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 beta1: Optional[float] = None,
                 weight_decay: float = 0.0,
                 scale_parameter: bool = True,
                 relative_step: bool = True,
                 warmup_init: bool = False):
        """
        Initialize Adafactor optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            eps: Regularization constants for moment estimates
            clip_threshold: Threshold for clipping gradient norms
            decay_rate: Decay rate for second moment estimator
            beta1: Coefficient for computing running averages of gradient
            weight_decay: Weight decay coefficient
            scale_parameter: If True, learning rate is scaled by root of parameter size
            relative_step: If True, time-dependent learning rate is used
            warmup_init: If True, warm up learning rate from 0
        """
        if not 0.0 <= clip_threshold:
            raise ValueError("Invalid clip_threshold value: {}".format(clip_threshold))
        if not 0.0 <= eps[0]:
            raise ValueError("Invalid eps value: {}".format(eps[0]))
            
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init
        )
        super(Adafactor, self).__init__(params, defaults)
        
    @staticmethod
    def _get_lr(param_group, param_state):
        """Get learning rate based on parameters"""
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0/math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz
        
    @staticmethod
    def _get_options(param_group, param_shape):
        """Get factored and use first moment options"""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment
        
    @staticmethod
    def _rms(tensor):
        """Root mean square of tensor"""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
        
    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        """Approximate squared gradient using factored moments"""
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
        c_factor = (exp_avg_sq_col / exp_avg_sq_col.mean(dim=-2, keepdim=True))
        return r_factor.unsqueeze(-1) * c_factor.unsqueeze(-2)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                # Handle complex parameters
                if grad.is_complex():
                    grad = torch.view_as_real(grad)
                    p_data_fp32 = torch.view_as_real(p.data)
                else:
                    p_data_fp32 = p.data
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['RMS'] = 0
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(p_data_fp32)
                        
                    if len(p_data_fp32.shape) >= 2:
                        state['exp_avg_sq_row'] = torch.zeros(p_data_fp32.shape[:-1]).to(p_data_fp32)
                        state['exp_avg_sq_col'] = torch.zeros(p_data_fp32.shape[:-2] + 
                                                            p_data_fp32.shape[-1:]).to(p_data_fp32)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                        
                # Update step count
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                
                # Get update parameters
                factored, use_first_moment = self._get_options(group, p_data_fp32.shape)
                lr = self._get_lr(group, state)
                beta1 = group['beta1']
                decay_rate = group['decay_rate']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # Update moments and compute update
                update = p_data_fp32.clone()
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    update = exp_avg
                    
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(decay_rate).add_(
                        grad.pow(2).mean(dim=-1), alpha=1 - decay_rate
                    )
                    exp_avg_sq_col.mul_(decay_rate).add_(
                        grad.pow(2).mean(dim=-2), alpha=1 - decay_rate
                    )
                    
                    # Approximate squared gradient
                    update.div_(
                        self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col).add_(eps[0])
                    )
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(decay_rate).add_(grad.pow(2), alpha=1 - decay_rate)
                    update.div_(exp_avg_sq.sqrt().add_(eps[0]))
                    
                update.mul_(lr)
                
                if weight_decay != 0:
                    update.add_(p_data_fp32, alpha=weight_decay)
                    
                p_data_fp32.add_(-update)
                
                if p.data.is_complex():
                    p.data = torch.view_as_complex(p_data_fp32)
                    
        return loss

def create_optimizer(params,
                    config: OptimizerConfig) -> torch.optim.Optimizer:
    """
    Create optimizer with specified configuration
    
    Args:
        params: Model parameters
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    if config.optimizer == 'adam':
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'lamb':
        return LAMB(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adafactor':
        return Adafactor(
            params,
            lr=config.learning_rate,
            eps=(config.eps, 1e-3),
            weight_decay=config.weight_decay,
            beta1=config.beta1
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

def create_scheduler(optimizer: torch.optim.Optimizer,
                    config: OptimizerConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        config: Optimizer configuration
        
    Returns:
        Learning rate scheduler
    """
    if config.lr_schedule == 'constant':
        return None
    elif config.lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.decay_steps,
            eta_min=config.learning_rate * config.min_lr_ratio
        )
    elif config.lr_schedule == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr_ratio,
            total_iters=config.decay_steps
        )
    elif config.lr_schedule == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=(config.min_lr_ratio) ** (1.0 / config.decay_steps)
        )
    else:
        raise ValueError(f"Unsupported learning rate schedule: {config.lr_schedule}")

def apply_gradient_updates(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         config: OptimizerConfig):
    """
    Apply gradient updates with clipping
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        config: Optimizer configuration
    """
    # Clip gradients
    if config.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.clip_grad_norm
        )
    if config.clip_grad_value is not None:
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            config.clip_grad_value
        )
        
    # Update weights
    optimizer.step()
    optimizer.zero_grad() 