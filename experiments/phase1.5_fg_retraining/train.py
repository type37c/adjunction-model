"""
Phase 1.5 訓練スクリプト

FunctorF_v2とFunctorGを「形状 + 行動 → 結果」のデータで訓練する。

損失関数:
L = λ_recon · L_recon + λ_action · L_action

L_recon: 再構成損失（Phase 1と同じ）
L_action: 行動予測損失（新規）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# パスの設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from models.functor_f_v2 import FunctorF_v2
from models.functor_g import FunctorG


class Phase15Dataset(Dataset):
    """Phase 1.5のデータセット"""
    
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} episodes from {dataset_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        episode = self.data[idx]
        
        # 点群をTensorに変換
        initial_pc = torch.from_numpy(episode['initial_pointcloud']).float()
        final_pc = torch.from_numpy(episode['final_pointcloud']).float()
        action = torch.from_numpy(episode['action_onehot']).float()
        
        return {
            'initial_pointcloud': initial_pc,
            'final_pointcloud': final_pc,
            'action': action,
        }


class Phase15Trainer:
    """Phase 1.5の訓練器"""
    
    def __init__(
        self,
        functor_f,
        functor_g,
        device='cuda',
        lr=1e-4,
        lambda_recon=1.0,
        lambda_action=0.5
    ):
        self.functor_f = functor_f.to(device)
        self.functor_g = functor_g.to(device)
        self.device = device
        
        self.lambda_recon = lambda_recon
        self.lambda_action = lambda_action
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.functor_f.parameters()) + list(self.functor_g.parameters()),
            lr=lr
        )
        
        # 行動予測器（簡易版）
        # アフォーダンスの平均から最終点群を予測
        self.action_predictor = nn.Sequential(
            nn.Linear(functor_f.affordance_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512 * 3),  # 512点 × 3次元
        ).to(device)
        
        self.optimizer_action = optim.Adam(self.action_predictor.parameters(), lr=lr)
        
        # 訓練履歴
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_action_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_action_loss': [],
        }
    
    def compute_loss(self, batch):
        """損失関数の計算"""
        initial_pc = batch['initial_pointcloud'].to(self.device)
        final_pc = batch['final_pointcloud'].to(self.device)
        action = batch['action'].to(self.device)
        
        B, N, D = initial_pc.shape
        
        # Forward pass (initial)
        affordances_initial = self.functor_f(initial_pc, goal=action)  # (B, N, affordance_dim)
        
        # FunctorG expects (B*N, affordance_dim)
        B_N_aff = affordances_initial.shape
        affordances_initial_flat = affordances_initial.reshape(-1, affordances_initial.shape[-1])
        recon_initial_flat = self.functor_g(affordances_initial_flat)  # (B*N, 3)
        recon_initial = recon_initial_flat.reshape(B, N, D)  # (B, N, 3)
        
        # Forward pass (final)
        affordances_final = self.functor_f(final_pc, goal=action)  # (B, N, affordance_dim)
        affordances_final_flat = affordances_final.reshape(-1, affordances_final.shape[-1])
        recon_final_flat = self.functor_g(affordances_final_flat)  # (B*N, 3)
        recon_final = recon_final_flat.reshape(B, N, D)  # (B, N, 3)
        
        # 再構成損失
        recon_loss_initial = nn.functional.mse_loss(recon_initial, initial_pc)
        recon_loss_final = nn.functional.mse_loss(recon_final, final_pc)
        recon_loss = (recon_loss_initial + recon_loss_final) / 2
        
        # 行動予測損失
        # アフォーダンスの平均から最終点群を予測
        affordance_summary = affordances_initial.mean(dim=1)  # (B, affordance_dim)
        predicted_final_flat = self.action_predictor(affordance_summary)  # (B, 512*3)
        predicted_final = predicted_final_flat.view(B, N, D)  # (B, 512, 3)
        
        action_loss = nn.functional.mse_loss(predicted_final, final_pc)
        
        # 総損失
        total_loss = self.lambda_recon * recon_loss + self.lambda_action * action_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'action_loss': action_loss,
        }
    
    def train_epoch(self, train_loader):
        """1エポックの訓練"""
        self.functor_f.train()
        self.functor_g.train()
        self.action_predictor.train()
        
        epoch_losses = {'total': 0, 'recon': 0, 'action': 0}
        
        for batch in tqdm(train_loader, desc='Training'):
            self.optimizer.zero_grad()
            self.optimizer_action.zero_grad()
            
            losses = self.compute_loss(batch)
            
            losses['total_loss'].backward()
            self.optimizer.step()
            self.optimizer_action.step()
            
            epoch_losses['total'] += losses['total_loss'].item()
            epoch_losses['recon'] += losses['recon_loss'].item()
            epoch_losses['action'] += losses['action_loss'].item()
        
        # 平均損失
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def validate(self, val_loader):
        """検証"""
        self.functor_f.eval()
        self.functor_g.eval()
        self.action_predictor.eval()
        
        epoch_losses = {'total': 0, 'recon': 0, 'action': 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                losses = self.compute_loss(batch)
                
                epoch_losses['total'] += losses['total_loss'].item()
                epoch_losses['recon'] += losses['recon_loss'].item()
                epoch_losses['action'] += losses['action_loss'].item()
        
        # 平均損失
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)
        
        return epoch_losses
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints'):
        """訓練ループ"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 訓練
            train_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_recon_loss'].append(train_losses['recon'])
            self.history['train_action_loss'].append(train_losses['action'])
            
            print(f"Train - Total: {train_losses['total']:.6f}, "
                  f"Recon: {train_losses['recon']:.6f}, "
                  f"Action: {train_losses['action']:.6f}")
            
            # 検証
            val_losses = self.validate(val_loader)
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_recon_loss'].append(val_losses['recon'])
            self.history['val_action_loss'].append(val_losses['action'])
            
            print(f"Val   - Total: {val_losses['total']:.6f}, "
                  f"Recon: {val_losses['recon']:.6f}, "
                  f"Action: {val_losses['action']:.6f}")
            
            # チェックポイントの保存
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint_path = os.path.join(checkpoint_dir, 'phase1.5_best.pth')
                torch.save({
                    'epoch': epoch,
                    'functor_f_state_dict': self.functor_f.state_dict(),
                    'functor_g_state_dict': self.functor_g.state_dict(),
                    'action_predictor_state_dict': self.action_predictor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'optimizer_action_state_dict': self.optimizer_action.state_dict(),
                    'val_loss': val_losses['total'],
                    'history': self.history,
                }, checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
        
        # 訓練曲線の保存
        self.plot_training_curves(checkpoint_dir)
    
    def plot_training_curves(self, output_dir):
        """訓練曲線のプロット"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(self.history['train_recon_loss'], label='Train')
        axes[1].plot(self.history['val_recon_loss'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Action prediction loss
        axes[2].plot(self.history['train_action_loss'], label='Train')
        axes[2].plot(self.history['val_action_loss'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Action Prediction Loss')
        axes[2].set_title('Action Prediction Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        print(f"Saved training curves to {os.path.join(output_dir, 'training_curves.png')}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phase 1.5 model')
    parser.add_argument('--dataset', type=str, default='data/phase1.5_dataset.pkl',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda-recon', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda-action', type=float, default=0.5,
                        help='Weight for action prediction loss')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--phase1-checkpoint', type=str, default=None,
                        help='Path to Phase 1 checkpoint (for transfer learning)')
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセットの読み込み
    dataset = Phase15Dataset(args.dataset)
    
    # 訓練/検証の分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # モデルの初期化
    functor_f = FunctorF_v2(
        input_dim=3,
        affordance_dim=32,
        goal_dim=4,  # 行動の種類（4種類: push, pull, lift, topple）
        k=16,
        hidden_dim=128
    )
    
    functor_g = FunctorG(
        affordance_dim=32,
        output_dim=3
    )
    
    # Phase 1のチェックポイントからの転移学習（オプション）
    if args.phase1_checkpoint:
        print(f"Loading Phase 1 checkpoint from {args.phase1_checkpoint}")
        functor_f.load_phase1_weights(args.phase1_checkpoint)
    
    # 訓練器の初期化
    trainer = Phase15Trainer(
        functor_f=functor_f,
        functor_g=functor_g,
        device=device,
        lr=args.lr,
        lambda_recon=args.lambda_recon,
        lambda_action=args.lambda_action
    )
    
    # 訓練の実行
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
