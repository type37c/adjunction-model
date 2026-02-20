"""
Suspension Structure: The Core of Intelligence

This module implements the suspension structure from the initial experiment note.

Theory:
When η (coherence signal) exceeds a threshold, the agent enters "suspension mode":
- Actions are withheld (not executed)
- Data is buffered for F/G fine-tuning
- F/G adapts to the new situation ("riverbed erosion")
- Once η drops below threshold, suspension is released

This is the implementation of Merleau-Ponty's "maximal grip" and the detection
of "tool breakdown" (Heidegger).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque


class SuspensionStructure:
    """
    Suspension structure for handling coherence breakdown.
    
    When η exceeds threshold:
    1. Enter suspension mode (withhold actions)
    2. Buffer observations for F/G adaptation
    3. Fine-tune F/G on buffered data
    4. Exit suspension when η drops below threshold
    """
    
    def __init__(
        self,
        eta_threshold: float = 0.1,
        buffer_size: int = 100,
        min_buffer_for_update: int = 10,
        fg_update_steps: int = 10
    ):
        """
        Args:
            eta_threshold: Threshold for entering suspension
            buffer_size: Maximum size of suspension buffer
            min_buffer_for_update: Minimum buffer size before F/G update
            fg_update_steps: Number of gradient steps for F/G fine-tuning
        """
        self.eta_threshold = eta_threshold
        self.buffer_size = buffer_size
        self.min_buffer_for_update = min_buffer_for_update
        self.fg_update_steps = fg_update_steps
        
        # State
        self.suspended = False
        self.buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.suspension_count = 0
        self.total_suspensions = 0
        self.suspension_durations = []
        self.current_suspension_duration = 0
    
    def check_suspension(self, eta: torch.Tensor) -> bool:
        """
        Check if suspension should be triggered.
        
        Args:
            eta: Current coherence signal (scalar or (B,))
        
        Returns:
            should_suspend: Whether to enter/stay in suspension
        """
        # Handle both scalar and batched eta
        if isinstance(eta, torch.Tensor):
            if eta.numel() == 1:
                eta_value = eta.item()
            else:
                eta_value = eta.mean().item()
        else:
            eta_value = eta
        
        # Check threshold
        if eta_value > self.eta_threshold:
            if not self.suspended:
                # Enter suspension
                self.suspended = True
                self.total_suspensions += 1
                self.current_suspension_duration = 0
                print(f"[SUSPENSION] Entered suspension mode (η={eta_value:.4f} > {self.eta_threshold})")
            else:
                # Stay in suspension
                self.current_suspension_duration += 1
            
            self.suspension_count += 1
            return True
        
        else:
            if self.suspended:
                # Exit suspension
                self.suspended = False
                self.suspension_durations.append(self.current_suspension_duration)
                print(f"[SUSPENSION] Exited suspension mode (η={eta_value:.4f} < {self.eta_threshold}, duration={self.current_suspension_duration})")
            
            return False
    
    def add_to_buffer(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        affordance: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None
    ):
        """
        Add observation to suspension buffer.
        
        Args:
            pos: Point cloud positions
            batch: Batch indices
            affordance: Affordance representation (optional)
            eta: Coherence signal (optional)
        """
        self.buffer.append({
            'pos': pos.detach().cpu(),
            'batch': batch.detach().cpu() if batch is not None else None,
            'affordance': affordance.detach().cpu() if affordance is not None else None,
            'eta': eta.detach().cpu() if eta is not None else None
        })
    
    def get_buffer_batch(
        self,
        batch_size: int,
        device: str = 'cpu'
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch from the suspension buffer.
        
        Args:
            batch_size: Batch size
            device: Device to move tensors to
        
        Returns:
            Batch dictionary or None if buffer is too small
        """
        if len(self.buffer) < self.min_buffer_for_update:
            return None
        
        # Sample from buffer
        import random
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        # Collate
        batch_dict = {
            'pos': torch.cat([s['pos'] for s in samples], dim=0).to(device),
        }
        
        # Handle optional fields
        if samples[0]['batch'] is not None:
            batch_dict['batch'] = torch.cat([s['batch'] for s in samples], dim=0).to(device)
        
        if samples[0]['affordance'] is not None:
            batch_dict['affordance'] = torch.cat([s['affordance'] for s in samples], dim=0).to(device)
        
        if samples[0]['eta'] is not None:
            batch_dict['eta'] = torch.cat([s['eta'] for s in samples], dim=0).to(device)
        
        return batch_dict
    
    def should_update_fg(self) -> bool:
        """Check if F/G should be updated."""
        return len(self.buffer) >= self.min_buffer_for_update
    
    def clear_buffer(self):
        """Clear the suspension buffer."""
        self.buffer.clear()
    
    def get_statistics(self) -> Dict:
        """Get suspension statistics."""
        return {
            'suspended': self.suspended,
            'suspension_count': self.suspension_count,
            'total_suspensions': self.total_suspensions,
            'buffer_size': len(self.buffer),
            'avg_suspension_duration': (
                sum(self.suspension_durations) / len(self.suspension_durations)
                if self.suspension_durations else 0
            ),
            'max_suspension_duration': (
                max(self.suspension_durations) if self.suspension_durations else 0
            )
        }
    
    def reset_statistics(self):
        """Reset suspension statistics."""
        self.suspension_count = 0
        self.total_suspensions = 0
        self.suspension_durations = []
        self.current_suspension_duration = 0


def finetune_fg_during_suspension(
    fg_model: nn.Module,
    suspension_structure: SuspensionStructure,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    batch_size: int = 32
) -> Optional[float]:
    """
    Fine-tune F/G during suspension using buffered data.
    
    This implements the "riverbed erosion" process where F/G adapts
    to new situations through gradient descent on recent observations.
    
    Args:
        fg_model: Bidirectional F/G model
        suspension_structure: Suspension structure with buffer
        optimizer: Optimizer for F/G
        device: Device
        batch_size: Batch size for fine-tuning
    
    Returns:
        Average loss or None if buffer is too small
    """
    if not suspension_structure.should_update_fg():
        return None
    
    fg_model.train()
    total_loss = 0.0
    num_steps = suspension_structure.fg_update_steps
    
    for step in range(num_steps):
        # Sample batch from buffer
        batch_dict = suspension_structure.get_buffer_batch(batch_size, device)
        
        if batch_dict is None:
            return None
        
        pos = batch_dict['pos']
        batch = batch_dict.get('batch', None)
        
        # Compute η (shape reconstruction loss)
        eta = fg_model.compute_eta(pos, batch)
        loss = eta.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fg_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_steps
    print(f"[F/G UPDATE] Fine-tuned F/G for {num_steps} steps, avg loss: {avg_loss:.6f}")
    
    return avg_loss
