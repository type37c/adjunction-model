# Implementation Log

このファイルは、Physical-Semantic Adjunction Modelの実装過程で行った設計判断、遭遇した問題、理論との齟齬を記録します。

---

## 2026-02-12: Phase 0 実装開始

### 環境構築

**実施内容**:
- PyTorch 2.10.0 (CPU版) をインストール
- PyTorch Geometric 2.7.0 をインストール
- その他の依存関係（h5py, trimesh, scipy, tqdm, pyyaml）をインストール

**設計判断**:
- CPU版を選択した理由: 初期プロトタイプでは小規模データセットを使用するため、GPU不要
- PyTorch Geometricの拡張ライブラリ（torch-scatter, torch-sparse）はPyG 2.7.0では不要になったため、インストールをスキップ

**ディレクトリ構造**:
```
adjunction-model/
├── src/
│   ├── models/      # F, G, Agent Layer Cの実装
│   ├── data/        # データローダー
│   ├── training/    # 学習ループ
│   └── utils/       # ユーティリティ関数
├── tests/           # 単体テスト
├── configs/         # 設定ファイル
├── data/            # データセット保存先
└── logs/            # 学習ログ
```

---

## 次のステップ

1. データローダーの実装（3D AffordanceNetの簡易版）
2. F（Shape→Action）の最小実装
3. 単体テストの作成
