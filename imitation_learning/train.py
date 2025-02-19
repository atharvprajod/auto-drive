import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import random
from training.trainer import TrainingConfig, ImitationLearningTrainer

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train imitation learning policy')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = ImitationLearningTrainer(
        config=config,
        data_dir=args.data_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train policy
    trainer.train()

if __name__ == '__main__':
    main() 