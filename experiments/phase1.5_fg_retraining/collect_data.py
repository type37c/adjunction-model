"""
Phase 1.5 データ収集スクリプト

PyBulletで「形状 + 行動 → 結果」のデータを収集する。

オブジェクト: box, cylinder, sphere, cup, bowl
行動: push, pull, rotate, lift, topple
各組み合わせ: 100エピソード
総エピソード数: 5 × 5 × 100 = 2500エピソード
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import torch
from tqdm import tqdm
import pickle

# パスの設定
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from env.pybullet_simple_env import SimplePyBulletEnv


class Phase15DataCollector:
    """Phase 1.5のデータ収集器"""
    
    def __init__(self, num_points=512, use_gui=False):
        self.num_points = num_points
        self.use_gui = use_gui
        
        # オブジェクトの種類
        self.object_types = ['box', 'cup', 'bowl']  # cylinder, sphereはSimplePyBulletEnvに未実装
        
        # 行動の種類
        self.action_types = ['push', 'pull', 'lift', 'topple']  # rotateは一旦除外
        
        # 行動のパラメータ
        self.action_params = {
            'push': {'force': [10, 0, 0], 'duration': 0.5},
            'pull': {'force': [-10, 0, 0], 'duration': 0.5},
            'lift': {'force': [0, 0, 15], 'duration': 0.5},
            'topple': {'force': [5, 0, 0], 'duration': 0.3, 'offset': [0, 0, 0.05]},
        }
    
    def collect_episode(self, object_type, action_type):
        """
        1エピソードのデータを収集
        
        Args:
            object_type (str): オブジェクトの種類
            action_type (str): 行動の種類
        
        Returns:
            dict: エピソードデータ
        """
        # 環境の初期化
        env = SimplePyBulletEnv(object_type=object_type, gui=self.use_gui)
        
        try:
            # シミュレーションを安定させる
            for _ in range(50):
                p.stepSimulation()
            
            # 初期点群の取得
            initial_pointcloud = env.get_point_cloud(num_points=self.num_points, fill_interior=True)
            
            # 行動の実行
            params = self.action_params[action_type]
            
            if action_type in ['push', 'pull', 'lift']:
                force = params['force']
                duration = params['duration']
                num_steps = int(duration / 0.01)  # dt=0.01s
                
                for _ in range(num_steps):
                    p.applyExternalForce(
                        env.object_id, -1, force, [0, 0, 0],
                        p.WORLD_FRAME
                    )
                    p.stepSimulation()
            
            elif action_type == 'topple':
                force = params['force']
                duration = params['duration']
                offset = params['offset']
                num_steps = int(duration / 0.01)
                
                for _ in range(num_steps):
                    p.applyExternalForce(
                        env.object_id, -1, force, offset,
                        p.LINK_FRAME
                    )
                    p.stepSimulation()
            
            # シミュレーションを少し進めて安定させる
            for _ in range(20):
                p.stepSimulation()
            
            # 最終点群の取得
            final_pointcloud = env.get_point_cloud(num_points=self.num_points, fill_interior=True)
            
            # 行動のone-hotエンコーディング
            action_onehot = np.zeros(len(self.action_types), dtype=np.float32)
            action_onehot[self.action_types.index(action_type)] = 1.0
            
            # エピソードデータ
            episode_data = {
                'object_type': object_type,
                'action_type': action_type,
                'action_onehot': action_onehot,
                'initial_pointcloud': initial_pointcloud,
                'final_pointcloud': final_pointcloud,
            }
            
            return episode_data
        
        finally:
            p.disconnect()
    
    def collect_dataset(self, episodes_per_combination=100, output_path='data/phase1.5_dataset.pkl'):
        """
        データセット全体を収集
        
        Args:
            episodes_per_combination (int): 各組み合わせのエピソード数
            output_path (str): 出力ファイルのパス
        """
        dataset = []
        
        total_combinations = len(self.object_types) * len(self.action_types)
        total_episodes = total_combinations * episodes_per_combination
        
        print(f"Collecting {total_episodes} episodes...")
        print(f"Objects: {self.object_types}")
        print(f"Actions: {self.action_types}")
        print(f"Episodes per combination: {episodes_per_combination}")
        
        with tqdm(total=total_episodes) as pbar:
            for object_type in self.object_types:
                for action_type in self.action_types:
                    for episode_idx in range(episodes_per_combination):
                        try:
                            episode_data = self.collect_episode(object_type, action_type)
                            dataset.append(episode_data)
                            pbar.update(1)
                            pbar.set_description(
                                f"{object_type}-{action_type} ({episode_idx+1}/{episodes_per_combination})"
                            )
                        except Exception as e:
                            print(f"\nError in {object_type}-{action_type} episode {episode_idx}: {e}")
                            pbar.update(1)
                            continue
        
        # データセットの保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nDataset saved to {output_path}")
        print(f"Total episodes collected: {len(dataset)}")
        
        # 統計情報の表示
        self.print_dataset_stats(dataset)
        
        return dataset
    
    def print_dataset_stats(self, dataset):
        """データセットの統計情報を表示"""
        print("\n=== Dataset Statistics ===")
        
        # オブジェクトごとのエピソード数
        object_counts = {}
        for episode in dataset:
            obj_type = episode['object_type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        print("\nEpisodes per object:")
        for obj_type, count in sorted(object_counts.items()):
            print(f"  {obj_type}: {count}")
        
        # 行動ごとのエピソード数
        action_counts = {}
        for episode in dataset:
            act_type = episode['action_type']
            action_counts[act_type] = action_counts.get(act_type, 0) + 1
        
        print("\nEpisodes per action:")
        for act_type, count in sorted(action_counts.items()):
            print(f"  {act_type}: {count}")
        
        # 点群の統計
        initial_points = [episode['initial_pointcloud'] for episode in dataset]
        final_points = [episode['final_pointcloud'] for episode in dataset]
        
        initial_mean = np.mean([pc.shape[0] for pc in initial_points])
        final_mean = np.mean([pc.shape[0] for pc in final_points])
        
        print(f"\nAverage number of points:")
        print(f"  Initial: {initial_mean:.1f}")
        print(f"  Final: {final_mean:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Phase 1.5 dataset')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes per object-action combination')
    parser.add_argument('--num-points', type=int, default=512,
                        help='Number of points in point cloud')
    parser.add_argument('--output', type=str, default='data/phase1.5_dataset.pkl',
                        help='Output file path')
    parser.add_argument('--gui', action='store_true',
                        help='Use PyBullet GUI (slower)')
    
    args = parser.parse_args()
    
    collector = Phase15DataCollector(
        num_points=args.num_points,
        use_gui=args.gui
    )
    
    dataset = collector.collect_dataset(
        episodes_per_combination=args.episodes,
        output_path=args.output
    )
    
    print("\nData collection complete!")


if __name__ == "__main__":
    main()
