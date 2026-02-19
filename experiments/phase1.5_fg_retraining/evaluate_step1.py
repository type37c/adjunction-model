"""
Phase 1.5評価スクリプト: Step 1を再実行

Phase 1.5で訓練されたF/Gを使ってStep 1の実験を再実行し、
ηの挙動が改善されたかを検証する。
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pybullet as p

# パスの設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.functor_f_v2 import FunctorF_v2
from models.functor_g import FunctorG
from env.pybullet_simple_env import SimplePyBulletEnv


def compute_eta(functor_f, functor_g, pointcloud, action_onehot, device='cpu'):
    """
    ηを計算する
    
    Args:
        functor_f: FunctorF_v2モデル
        functor_g: FunctorGモデル
        pointcloud: 点群 (N, 3)
        action_onehot: 行動のone-hotベクトル (4,)
        device: デバイス
    
    Returns:
        float: η（再構成誤差）
    """
    functor_f.eval()
    functor_g.eval()
    
    with torch.no_grad():
        # 点群をTensorに変換
        points = torch.from_numpy(pointcloud).float().unsqueeze(0).to(device)  # (1, N, 3)
        action = torch.from_numpy(action_onehot).float().unsqueeze(0).to(device)  # (1, 4)
        
        # Forward pass
        affordances = functor_f(points, goal=action)  # (1, N, affordance_dim)
        
        # FunctorGに通す
        B, N, aff_dim = affordances.shape
        affordances_flat = affordances.reshape(-1, aff_dim)  # (N, aff_dim)
        recon_flat = functor_g(affordances_flat)  # (N, 3)
        recon = recon_flat.reshape(1, N, 3)  # (1, N, 3)
        
        # 再構成誤差
        eta = torch.nn.functional.mse_loss(recon, points).item()
    
    return eta


def run_step1_experiment(functor_f, functor_g, device='cpu'):
    """
    Step 1の実験を実行
    
    Returns:
        pd.DataFrame: 結果のデータフレーム
    """
    object_types = ['box', 'cup', 'bowl']
    action_types = ['static', 'push', 'pull', 'lift', 'topple']
    
    # 行動のone-hotエンコーディング（staticは全て0）
    action_to_onehot = {
        'static': np.zeros(4, dtype=np.float32),
        'push': np.array([1, 0, 0, 0], dtype=np.float32),
        'pull': np.array([0, 1, 0, 0], dtype=np.float32),
        'lift': np.array([0, 0, 1, 0], dtype=np.float32),
        'topple': np.array([0, 0, 0, 1], dtype=np.float32),
    }
    
    results = []
    
    total_trials = len(object_types) * len(action_types) * 5
    with tqdm(total=total_trials, desc='Running Step 1') as pbar:
        for object_type in object_types:
            for action_type in action_types:
                for trial in range(5):
                    # 環境の初期化
                    env = SimplePyBulletEnv(object_type=object_type, gui=False)
                    
                    try:
                        # シミュレーションを安定させる
                        for _ in range(50):
                            p.stepSimulation()
                        
                        # 初期点群の取得とη計算
                        initial_pc = env.get_point_cloud(num_points=512, fill_interior=True)
                        action_onehot = action_to_onehot[action_type]
                        eta_initial = compute_eta(functor_f, functor_g, initial_pc, action_onehot, device)
                        
                        # 行動の実行（staticの場合は何もしない）
                        if action_type == 'push':
                            for _ in range(50):
                                p.applyExternalForce(env.object_id, -1, [10, 0, 0], [0, 0, 0], p.WORLD_FRAME)
                                p.stepSimulation()
                        elif action_type == 'pull':
                            for _ in range(50):
                                p.applyExternalForce(env.object_id, -1, [-10, 0, 0], [0, 0, 0], p.WORLD_FRAME)
                                p.stepSimulation()
                        elif action_type == 'lift':
                            for _ in range(50):
                                p.applyExternalForce(env.object_id, -1, [0, 0, 15], [0, 0, 0], p.WORLD_FRAME)
                                p.stepSimulation()
                        elif action_type == 'topple':
                            for _ in range(30):
                                p.applyExternalForce(env.object_id, -1, [5, 0, 0], [0, 0, 0.05], p.LINK_FRAME)
                                p.stepSimulation()
                        
                        # 最終点群の取得とη計算
                        for _ in range(20):
                            p.stepSimulation()
                        final_pc = env.get_point_cloud(num_points=512, fill_interior=True)
                        eta_final = compute_eta(functor_f, functor_g, final_pc, action_onehot, device)
                        
                        # 結果の記録
                        results.append({
                            'object_type': object_type,
                            'action_type': action_type,
                            'trial': trial,
                            'eta_initial': eta_initial,
                            'eta_final': eta_final,
                            'eta_change': eta_final - eta_initial,
                        })
                        
                        pbar.update(1)
                        pbar.set_description(f"{object_type}-{action_type} ({trial+1}/5)")
                    
                    except Exception as e:
                        print(f"\nError in {object_type}-{action_type} trial {trial}: {e}")
                        pbar.update(1)
                    
                    finally:
                        p.disconnect()
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Phase 1.5 model with Step 1')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase1.5_best.pth',
                        help='Path to Phase 1.5 checkpoint')
    parser.add_argument('--output', type=str, default='results/step1_phase1.5_results.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルの初期化
    functor_f = FunctorF_v2(
        input_dim=3,
        affordance_dim=32,
        goal_dim=4,
        k=16,
        hidden_dim=128
    )
    
    functor_g = FunctorG(
        affordance_dim=32,
        output_dim=3
    )
    
    # チェックポイントのロード
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    functor_f.load_state_dict(checkpoint['functor_f_state_dict'])
    functor_g.load_state_dict(checkpoint['functor_g_state_dict'])
    
    functor_f.to(device)
    functor_g.to(device)
    
    print("Models loaded successfully")
    
    # Step 1の実験を実行
    results_df = run_step1_experiment(functor_f, functor_g, device)
    
    # 結果の保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # 統計情報の表示
    print("\n=== Results Summary ===")
    print(f"Total trials: {len(results_df)}")
    print(f"\nη statistics:")
    print(f"  Mean: {results_df['eta_final'].mean():.6f}")
    print(f"  Std: {results_df['eta_final'].std():.6f}")
    print(f"  Min: {results_df['eta_final'].min():.6f}")
    print(f"  Max: {results_df['eta_final'].max():.6f}")
    print(f"  CV (Coefficient of Variation): {results_df['eta_final'].std() / results_df['eta_final'].mean():.6f}")
    
    print(f"\nη by action type:")
    print(results_df.groupby('action_type')['eta_final'].agg(['mean', 'std', 'count']))
    
    print(f"\nη by object type:")
    print(results_df.groupby('object_type')['eta_final'].agg(['mean', 'std', 'count']))


if __name__ == "__main__":
    main()
