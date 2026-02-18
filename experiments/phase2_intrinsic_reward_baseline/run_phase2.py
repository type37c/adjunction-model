"""
Phase 2: Comparative η Experiment

This experiment tests whether Agent C can learn to decompose unknown composite objects
into known components using attention selection guided by comparative η (η_whole vs η_part).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from src.models.adjunction_model import AdjunctionModel
from src.models.agent_c_attention import AttentionAgent, DQNTrainer
from src.data.composite_dataset import CompositeShapeDataset


class Phase2Experiment:
    """
    Phase 2 Experiment: Comparative η for Decomposition
    """
    
    def __init__(
        self,
        phase1_checkpoint_path: str,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        alpha: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            phase1_checkpoint_path: Path to Phase 1 trained F/G checkpoint
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum attention selections per episode
            alpha: Intrinsic reward scaling factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            device: Device to run on
        """
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Load Phase 1 trained F/G (frozen)
        print(f"[{datetime.now()}] Loading Phase 1 checkpoint from {phase1_checkpoint_path}")
        self.fg_model = self._load_frozen_fg(phase1_checkpoint_path)
        
        # Initialize Agent C with attention selection
        print(f"[{datetime.now()}] Initializing Agent C with attention selection")
        self.agent = AttentionAgent(
            state_dim=16,  # Affordance dimension from F
            num_segments=8,
            hidden_dim=256,
            eta_history_len=10
        ).to(device)
        
        # Initialize DQN trainer
        self.trainer = DQNTrainer(
            agent=self.agent,
            learning_rate=1e-4,
            gamma=0.99,
            target_update_freq=100,
            alpha=alpha
        )
        
        # Load datasets
        print(f"[{datetime.now()}] Loading datasets")
        self.train_dataset = CompositeShapeDataset(
            num_samples=200,
            category="all",
            seed=42
        )
        self.eval_dataset = CompositeShapeDataset(
            num_samples=50,
            category="all",
            seed=123
        )
        
        # Training history
        self.history = {
            "episode_rewards": [],
            "episode_losses": [],
            "epsilon_values": [],
            "affordance_accuracy": []
        }
    
    def _load_frozen_fg(self, checkpoint_path: str) -> AdjunctionModel:
        """Load Phase 1 trained F/G and freeze parameters."""
        model = AdjunctionModel(
            input_dim=3,
            affordance_dim=16,
            hidden_dim=128
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze F/G parameters
        for param in model.F.parameters():
            param.requires_grad = False
        for param in model.G.parameters():
            param.requires_grad = False
        
        model.eval()
        return model
    
    def _compute_eta(self, points: torch.Tensor) -> float:
        """Compute reconstruction error η for given points."""
        with torch.no_grad():
            points = points.to(self.device)  # (N, 3)
            affordances = self.fg_model.F(points)  # (N, affordance_dim)
            reconstructed = self.fg_model.G(affordances)  # (N, 3)
            eta = torch.mean((points - reconstructed) ** 2).item()
        return eta
    
    def _extract_segment_points(
        self,
        points: torch.Tensor,
        segments: torch.Tensor,
        segment_id: int
    ) -> torch.Tensor:
        """Extract points belonging to a specific segment."""
        mask = segments == segment_id
        segment_points = points[mask]
        
        # Normalize to object center
        if len(segment_points) > 0:
            centroid = segment_points.mean(dim=0)
            segment_points = segment_points - centroid
        
        return segment_points
    
    def run_episode(self, sample: dict, train: bool = True) -> dict:
        """
        Run one episode of attention selection on a composite object.
        
        Args:
            sample: Composite object sample from dataset
            train: Whether to update agent (training mode)
            
        Returns:
            Episode statistics
        """
        points = sample["points"]  # (N, 3)
        segments = sample["segments"]  # (N,)
        affordances_gt = sample["affordances"]  # Ground truth affordances
        
        # Reset agent history
        self.agent.reset_history()
        
        # Compute η_whole
        eta_whole = self._compute_eta(points)
        
        # Get available segments
        unique_segments = torch.unique(segments).tolist()
        num_segments = len(unique_segments)
        
        # Initialize state (use F to encode whole object)
        with torch.no_grad():
            affordances = self.fg_model.F(points.to(self.device))  # (N, affordance_dim)
            state = affordances.mean(dim=0, keepdim=True)  # (1, affordance_dim) - global pooling
        
        # Episode loop
        episode_reward = 0.0
        episode_loss = 0.0
        visited_segments = set()
        predicted_affordances = {}
        
        for step in range(min(self.max_steps_per_episode, num_segments)):
            # Select segment to focus on
            available_segments = [s for s in unique_segments if s not in visited_segments]
            if len(available_segments) == 0:
                break
            
            action = self.agent.select_action(
                state,
                eta_whole,
                eta_whole if step == 0 else eta_part,  # Use η_whole initially
                epsilon=self.epsilon if train else 0.0,
                available_segments=available_segments
            )
            
            visited_segments.add(action)
            
            # Extract segment points
            segment_points = self._extract_segment_points(points, segments, action)
            
            if len(segment_points) == 0:
                continue
            
            # Compute η_part
            eta_part = self._compute_eta(segment_points)
            
            # Compute intrinsic reward
            reward = self.trainer.compute_intrinsic_reward(eta_whole, eta_part)
            episode_reward += reward
            
            # Get affordance prediction for this segment
            with torch.no_grad():
                affordances_part = self.fg_model.F(segment_points.to(self.device))  # (N_part, affordance_dim)
                affordance_pred = affordances_part.mean(dim=0, keepdim=True)  # (1, affordance_dim) - global pooling
                predicted_affordances[action] = affordance_pred
            
            # Prepare next state
            next_state = state  # State doesn't change in this simplified version
            done = (step == self.max_steps_per_episode - 1) or (len(visited_segments) == num_segments)
            
            # Update agent
            if train:
                loss = self.trainer.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    eta_whole,
                    eta_whole if step == 0 else eta_part,
                    eta_whole,
                    eta_part
                )
                episode_loss += loss
        
        # Compute affordance prediction accuracy
        accuracy = self._compute_affordance_accuracy(
            predicted_affordances,
            affordances_gt
        )
        
        return {
            "reward": episode_reward,
            "loss": episode_loss / max(1, step + 1),
            "accuracy": accuracy,
            "num_steps": step + 1,
            "eta_whole": eta_whole,
            "visited_segments": list(visited_segments)
        }
    
    def _compute_affordance_accuracy(
        self,
        predicted: dict,
        ground_truth: dict
    ) -> float:
        """
        Compute affordance prediction accuracy.
        
        For now, this is a placeholder. In a full implementation,
        we would compare predicted affordance vectors with ground truth labels.
        """
        # Placeholder: return 1.0 if any segment was visited, else 0.0
        return 1.0 if len(predicted) > 0 else 0.0
    
    def train(self):
        """Train Agent C on composite objects."""
        print(f"\n[{datetime.now()}] Starting Phase 2 training")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Max steps per episode: {self.max_steps_per_episode}")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Evaluation samples: {len(self.eval_dataset)}")
        sys.stdout.flush()
        
        for episode in range(self.num_episodes):
            # Sample random object from training set
            sample_idx = np.random.randint(len(self.train_dataset))
            sample = self.train_dataset[sample_idx]
            
            # Run episode
            stats = self.run_episode(sample, train=True)
            
            # Record history
            self.history["episode_rewards"].append(stats["reward"])
            self.history["episode_losses"].append(stats["loss"])
            self.history["epsilon_values"].append(self.epsilon)
            self.history["affordance_accuracy"].append(stats["accuracy"])
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.history["episode_rewards"][-10:])
                avg_loss = np.mean(self.history["episode_losses"][-10:])
                avg_accuracy = np.mean(self.history["affordance_accuracy"][-10:])
                
                print(f"[{datetime.now()}] Episode {episode+1}/{self.num_episodes}: "
                      f"Reward={avg_reward:.4f}, Loss={avg_loss:.4f}, "
                      f"Accuracy={avg_accuracy:.4f}, ε={self.epsilon:.4f}")
                sys.stdout.flush()
            
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(episode + 1)
        
        print(f"\n[{datetime.now()}] Training completed!")
        sys.stdout.flush()
    
    def evaluate(self):
        """Evaluate Agent C on held-out composite objects."""
        print(f"\n[{datetime.now()}] Evaluating Agent C on {len(self.eval_dataset)} objects")
        sys.stdout.flush()
        
        eval_stats = {
            "rewards": [],
            "accuracies": [],
            "num_steps": [],
            "eta_wholes": []
        }
        
        for idx in range(len(self.eval_dataset)):
            sample = self.eval_dataset[idx]
            stats = self.run_episode(sample, train=False)
            
            eval_stats["rewards"].append(stats["reward"])
            eval_stats["accuracies"].append(stats["accuracy"])
            eval_stats["num_steps"].append(stats["num_steps"])
            eval_stats["eta_wholes"].append(stats["eta_whole"])
        
        # Compute summary statistics
        summary = {
            "mean_reward": np.mean(eval_stats["rewards"]),
            "mean_accuracy": np.mean(eval_stats["accuracies"]),
            "mean_steps": np.mean(eval_stats["num_steps"]),
            "mean_eta_whole": np.mean(eval_stats["eta_wholes"])
        }
        
        print(f"\n[{datetime.now()}] Evaluation Results:")
        print(f"  Mean Reward: {summary['mean_reward']:.4f}")
        print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
        print(f"  Mean Steps: {summary['mean_steps']:.2f}")
        print(f"  Mean η_whole: {summary['mean_eta_whole']:.4f}")
        sys.stdout.flush()
        
        return summary
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"phase2_episode_{episode}.pt"
        torch.save({
            "episode": episode,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "history": self.history,
            "epsilon": self.epsilon
        }, checkpoint_path)
        
        print(f"[{datetime.now()}] Checkpoint saved: {checkpoint_path}")
        sys.stdout.flush()
    
    def save_results(self):
        """Save final results."""
        results_path = Path("results.json")
        with open(results_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"[{datetime.now()}] Results saved: {results_path}")
        sys.stdout.flush()


if __name__ == "__main__":
    # Phase 1 checkpoint path
    phase1_checkpoint = "../../experiments/phase1_basic_adjunction/checkpoints/phase1_final.pt"
    
    # Initialize experiment
    experiment = Phase2Experiment(
        phase1_checkpoint_path=phase1_checkpoint,
        num_episodes=1000,
        max_steps_per_episode=10,
        alpha=1.0
    )
    
    # Train
    experiment.train()
    
    # Evaluate
    experiment.evaluate()
    
    # Save results
    experiment.save_results()
    
    print(f"\n[{datetime.now()}] Phase 2 experiment completed!")
    sys.stdout.flush()
