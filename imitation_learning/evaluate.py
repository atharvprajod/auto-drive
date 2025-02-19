import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List

from training.trainer import ImitationLearningTrainer
from dataset.driving_dataset import DrivingDataset, DataTransform

class PolicyEvaluator:
    def __init__(self,
                 policy: torch.nn.Module,
                 data_dir: str,
                 device: torch.device = None):
        """
        Initialize policy evaluator
        
        Args:
            policy: Trained policy
            data_dir: Path to dataset
            device: Torch device
        """
        self.policy = policy
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        
        # Create test dataset
        transform = DataTransform(
            image_size=(224, 224),
            num_points=2048,
            normalize=True
        )
        
        self.test_dataset = DrivingDataset(
            data_path=data_dir,
            split='test',
            transform=transform
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate policy on test set
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get policy predictions
                pred_actions = self.policy(**{k: v for k, v in batch.items() 
                                           if k != 'actions'})
                
                # Compute metrics
                metrics = self.compute_metrics(pred_actions, batch['actions'])
                all_metrics.append(metrics)
                
        # Aggregate metrics
        mean_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        return mean_metrics
    
    def compute_metrics(self,
                       pred_actions: torch.Tensor,
                       target_actions: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            pred_actions: Predicted actions
            target_actions: Ground truth actions
            
        Returns:
            Dictionary of metrics
        """
        # MSE for each action dimension
        action_mse = torch.mean((pred_actions - target_actions)**2, dim=0)
        
        # Overall metrics
        metrics = {
            'steering_mse': action_mse[0].item(),
            'acceleration_mse': action_mse[1].item(),
            'total_mse': torch.mean(action_mse).item(),
            
            # Absolute errors
            'steering_mae': torch.mean(torch.abs(pred_actions[:, 0] - 
                                               target_actions[:, 0])).item(),
            'acceleration_mae': torch.mean(torch.abs(pred_actions[:, 1] - 
                                                   target_actions[:, 1])).item(),
                                                   
            # Maximum errors
            'max_steering_error': torch.max(torch.abs(pred_actions[:, 0] - 
                                                     target_actions[:, 0])).item(),
            'max_acceleration_error': torch.max(torch.abs(pred_actions[:, 1] - 
                                                        target_actions[:, 1])).item()
        }
        
        return metrics
        
    def visualize_predictions(self,
                            num_examples: int = 5,
                            save_dir: str = None):
        """
        Visualize policy predictions
        
        Args:
            num_examples: Number of examples to visualize
            save_dir: Optional directory to save visualizations
        """
        self.policy.eval()
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        with torch.no_grad():
            # Get random batch
            batch_idx = np.random.randint(len(self.test_loader))
            batch = next(iter(self.test_loader))
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get predictions
            pred_actions = self.policy(**{k: v for k, v in batch.items() 
                                       if k != 'actions'})
            
            # Plot predictions vs ground truth
            import matplotlib.pyplot as plt
            
            for i in range(min(num_examples, len(batch['states']))):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Plot steering angle
                ax1.plot(pred_actions[i, 0].cpu().numpy(), label='Predicted')
                ax1.plot(batch['actions'][i, 0].cpu().numpy(), label='Ground Truth')
                ax1.set_title('Steering Rate')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Steering Rate (rad/s)')
                ax1.legend()
                
                # Plot acceleration
                ax2.plot(pred_actions[i, 1].cpu().numpy(), label='Predicted')
                ax2.plot(batch['actions'][i, 1].cpu().numpy(), label='Ground Truth')
                ax2.set_title('Acceleration Rate')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Acceleration Rate (m/s^3)')
                ax2.legend()
                
                if save_dir is not None:
                    plt.savefig(save_dir / f'example_{i}.png')
                else:
                    plt.show()
                    
                plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate imitation learning policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Path to save evaluation results')
    parser.add_argument('--num_vis', type=int, default=5,
                       help='Number of examples to visualize')
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    
    # Create trainer and load model
    trainer = ImitationLearningTrainer(
        config=checkpoint['config'],
        data_dir=args.data_dir,
        checkpoint_dir=''  # Not needed for evaluation
    )
    trainer.policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = PolicyEvaluator(
        policy=trainer.policy,
        data_dir=args.data_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Visualize predictions
    evaluator.visualize_predictions(
        num_examples=args.num_vis,
        save_dir=output_dir / 'visualizations'
    )
    
    # Print results
    print('\nEvaluation Results:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main() 