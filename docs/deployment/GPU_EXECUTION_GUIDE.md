# GPU Execution Guide for Purpose-Emergent Experiment

**Date**: 2026-02-15

---

## 前提条件

- CoreWeaveまたは同等のGPUクラウドサービスのアカウント
- SSH/API経由でGPU環境にアクセス可能
- NVIDIA GPU（A100, H100推奨）とCUDA 11.8+

---

## ステップ1: GPU環境へのアクセス

### CoreWeaveの場合

```bash
# Kubernetesクラスタへの接続（CoreWeaveから提供されるkubeconfigを使用）
export KUBECONFIG=/path/to/coreweave-kubeconfig.yaml

# または、SSH経由でGPUインスタンスに接続
ssh user@gpu-instance.coreweave.com
```

### 他のクラウドサービスの場合

```bash
# 例: AWS EC2
ssh -i your-key.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute.amazonaws.com

# 例: GCP
gcloud compute ssh gpu-instance --zone=us-central1-a
```

---

## ステップ2: リポジトリのセットアップ

GPU環境で以下を実行します。

```bash
# セットアップスクリプトをダウンロード
wget https://raw.githubusercontent.com/type37c/adjunction-model/master/scripts/setup_gpu_environment.sh

# 実行権限を付与
chmod +x setup_gpu_environment.sh

# セットアップを実行
./setup_gpu_environment.sh
```

このスクリプトは以下を自動的に実行します：

1. システム情報の確認
2. 必要なシステムパッケージのインストール
3. PyTorchおよび依存ライブラリのインストール
4. GPU動作確認
5. リポジトリのクローン/更新
6. 実験スクリプトの動作確認

---

## ステップ3: 実験の実行

### デフォルト設定での実行（小規模テスト）

```bash
cd /workspace/adjunction-model
python3 experiments/purpose_emergent_experiment.py
```

デフォルトパラメータ：
- エポック数: 3
- バッチサイズ: 2
- サンプル数: 100
- ステップ数: 8
- デバイス: CPU（自動的にGPUが検出される）

### カスタムパラメータでの実行（本格実験）

```bash
cd /workspace/adjunction-model
python3 << 'EOF'
from experiments.purpose_emergent_experiment import run_purpose_emergent_experiment

# 論文用の本格実験
run_purpose_emergent_experiment(
    num_epochs=50,        # エポック数を増やす
    num_samples=1000,     # サンプル数を増やす
    num_points=256,       # 点群の点数
    num_steps=8,          # 時間ステップ数
    batch_size=8,         # バッチサイズを増やす
    lr=1e-4,              # 学習率
    device='cuda'         # GPUを明示的に指定
)
EOF
```

### バックグラウンド実行（長時間実験）

```bash
cd /workspace/adjunction-model
nohup python3 -c "from experiments.purpose_emergent_experiment import run_purpose_emergent_experiment; run_purpose_emergent_experiment(num_epochs=50, batch_size=8, device='cuda')" > experiment.log 2>&1 &

# ログの確認
tail -f experiment.log
```

---

## ステップ4: 結果の確認

実験が完了すると、以下のディレクトリに結果が保存されます。

```
results/purpose_emergent/
├── purpose_emergent/
│   ├── metrics.json          # 訓練・検証メトリクス
│   ├── model.pt              # 学習済みモデル
│   └── displacement_head.pt  # 学習済みDisplacementHead
├── baseline/
│   ├── metrics.json
│   ├── model.pt
│   └── displacement_head.pt
└── summary.json              # 実験全体のサマリー
```

### 結果のダウンロード

```bash
# ローカルマシンから実行（GPU環境からダウンロード）
scp -r user@gpu-instance:/workspace/adjunction-model/results/purpose_emergent ./local_results/

# または、tarで圧縮してからダウンロード
ssh user@gpu-instance "cd /workspace/adjunction-model && tar czf results.tar.gz results/purpose_emergent"
scp user@gpu-instance:/workspace/adjunction-model/results.tar.gz ./
tar xzf results.tar.gz
```

---

## ステップ5: 結果の分析

結果をダウンロードした後、分析スクリプトを実行します。

```bash
cd /home/ubuntu/adjunction-model
python3 experiments/analyze_purpose_emergent.py
```

これにより、以下の可視化が生成されます：

- `eta_trajectory`: 保留の時間発展
- `purpose_switches`: 目的の切り替え頻度
- `displacement_magnitude`: 行動の大きさの変化
- `chosen_shape_distribution`: 選択された形状の分布

---

## トラブルシューティング

### GPU が認識されない

```bash
# CUDA の確認
nvidia-smi

# PyTorch から GPU が見えるか確認
python3 -c "import torch; print(torch.cuda.is_available())"

# CUDA ドライバの再インストールが必要な場合
sudo apt-get install --reinstall nvidia-driver-535
```

### メモリ不足エラー

```bash
# バッチサイズを減らす
python3 -c "from experiments.purpose_emergent_experiment import run_purpose_emergent_experiment; run_purpose_emergent_experiment(batch_size=1, device='cuda')"

# または、より小さいモデルを使用
# src/models/adjunction_model.py の hidden_dim パラメータを調整
```

### 依存関係エラー

```bash
# 依存関係を再インストール
pip3 install --force-reinstall torch torch-geometric torch-scatter torch-sparse
```

---

## コスト最適化のヒント

1. **実験の段階的実行**: まず小規模（3エポック）で動作確認し、問題がなければ本格実験（50エポック）を実行
2. **スポットインスタンスの利用**: CoreWeaveやAWSのスポットインスタンスを利用してコストを削減
3. **結果の定期バックアップ**: 長時間実験の場合、定期的に中間結果をダウンロード

---

## 推定実行時間とコスト

| 構成 | GPU | エポック | バッチサイズ | サンプル数 | 推定時間 | 推定コスト (CoreWeave) |
|:---|:---|:---|:---|:---|:---|:---|
| 小規模テスト | 1x A100 | 3 | 2 | 100 | 2-4時間 | $10-20 |
| 本格実験 | 8x A100 | 50 | 8 | 1000 | 8-12時間 | $160-240 |
| 論文用実験 | 8x H100 | 100 | 16 | 2000 | 12-16時間 | $600-800 |

---

## サポート

問題が発生した場合は、以下を確認してください：

1. `experiment.log` のエラーメッセージ
2. `results/purpose_emergent/metrics.json` の内容
3. GPU メモリ使用状況（`nvidia-smi`）

それでも解決しない場合は、GitHubのIssueで報告してください。
