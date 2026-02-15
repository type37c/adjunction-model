# GPU Service Recommendation for Purpose-Emergent Experiment

**Date**: 2026-02-15

---

## Meta関連のGPUサービス

調査の結果、**CoreWeave**がMeta関連のGPUサービスとして最適であることが判明しました。

### CoreWeaveとMetaの関係

1. **$14.2B契約（2025年9月）**: MetaはCoreWeaveと6年間で$14.2Bの大型契約を締結し、AI workload用のGPU computeを調達しています。
2. **PyTorchネイティブサポート**: CoreWeaveはMeta開発のPyTorchをネイティブサポートし、Kubernetes APIを通じた柔軟なデプロイが可能です。
3. **NVIDIA GPU最新世代へのアクセス**: Blackwell (GB200/GB300), H100, A100など、最新のNVIDIA GPUに早期アクセスできます。

---

## CoreWeaveの特徴

| 項目 | 詳細 |
|:---|:---|
| **GPU種類** | H100 (80GB), A100 (80GB), A40, RTX A6000等 |
| **料金例** | H100 8GPU: $49.24/時間, A100 8GPU: $20.00/時間 |
| **PyTorchサポート** | ネイティブサポート、Kubernetes API経由でデプロイ |
| **スケーラビリティ** | 96% cluster goodput、3倍の高速化実績 |
| **利用開始** | Contact Sales経由でアカウント作成 |

---

## Purpose-Emergent実験の推奨構成

### 小規模実験（初回検証）

- **GPU**: 1x A100 (80GB) または 1x H100 (80GB)
- **推定時間**: 3エポック、バッチサイズ2、100サンプル → 約2-4時間
- **推定コスト**: $10-20（A100）または $25-50（H100）

### 本格実験（論文用データ取得）

- **GPU**: 8x A100 (80GB)
- **推定時間**: 50エポック、バッチサイズ8、1000サンプル → 約8-12時間
- **推定コスト**: $160-240

---

## 次のステップ

1. **CoreWeaveアカウント作成**: [Contact Sales](https://www.coreweave.com/pricing)からアカウント申請
2. **SSH/API認証情報の取得**: Kubernetes APIまたはSSH経由でアクセス
3. **実験コードのデプロイ**: このサンドボックスからCoreWeave環境にコードを転送し、実行

---

## 代替案

もしCoreWeaveの契約プロセスが長期化する場合、以下の代替サービスも検討可能です。

| サービス | 特徴 | Meta関連性 |
|:---|:---|:---|
| **Lambda Labs** | 低価格、即時利用可能 | なし（独立系） |
| **RunPod** | 柔軟な課金、Jupyter対応 | なし（独立系） |
| **Google Cloud (GCP)** | Metaが$10B+契約 | 間接的（Metaが顧客） |

ただし、「企業がMetaを好む」という要件を最も満たすのは**CoreWeave**です。
