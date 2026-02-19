# 物理的意味的随伴モデル (Physical-Semantic Adjunction Model)

**バージョン:** 4.0  
**最終更新日:** 2026年2月19日

## プロジェクト概要

このリポジトリは、**物理的意味的随伴モデル (Physical-Semantic Adjunction Model)** の研究と実装を含みます。本プロジェクトの核心的な目標は、**「未知の形状に対してアフォーダンスを推測し、目的を創発できるエージェントを開発すること」**です。

このモデルは、知性の本質が「未確定性を保持し、新しい意味を創発する能力」にあるという問いから出発し、圏論における**随伴 (adjunction)** と、本モデルで提唱する**保留構造 (suspension structure)**、そして**目的機構 (purpose mechanism)** を統合したアーキテクチャを構築します。これにより、記号接地問題の解決と創造性の創発を目指します。

## 哲学的基盤

本プロジェクトは、以下の哲学的理論に基づいています：

- **ハイデガー (Heidegger)**: 道具連関、「〜のために存在する」構造
- **メルロ＝ポンティ (Merleau-Ponty)**: 運動志向性、最大把握への傾向性
- **ギブソン (Gibson)**: アフォーダンス理論、直接知覚
- **Rietveld & Kiverstein**: 熟練した志向性の枠組み (Skilled Intentionality Framework)
- **Bruineberg & Rietveld**: 自由エネルギー原理と最適把握の統合
- **エナクティヴィズム (Enactivism)**: 自己産出、適応性、意味生成

詳細は [哲学的理論調査ノート](./research/philosophy_of_purpose_notes.md) を参照してください。

## 主要ドキュメント

### 理論・設計
- **[セオリー](./docs/THEORY.md)**: モデルの理論的背景、保留構造、Coherence Signal、Active Inferenceとの関係
- **[アーキテクチャ](./ARCHITECTURE.md)**: 「川床」と「水」の二重学習システム、フィルター機構、軌跡ベースの記憶モデル
- **[目的機構の設計](./research/PURPOSE_MECHANISM_DESIGN.md)**: Module M（Concern State, Grip Monitor, Salience Modulator）の哲学的基盤と実装設計
- **[ロードマップ](./docs/ROADMAP.md)**: プロジェクトのゴール達成に向けた段階的な開発計画

### 実験・結果
- **[Phase 1: F/G事前訓練](./experiments/phase1_basic_adjunction/)**: 基本的な随伴の訓練と評価
- **[Phase 2.1: PyBulletシミュレーション](./experiments/phase2.1_trajectory_prediction/)**: 物理シミュレーション環境でのアフォーダンス学習
  - **[Step 1: ηの物理的意味の検証](./experiments/phase2.1_trajectory_prediction/step1_eta_validation/)**: F/Gの再構成誤差ηが物理的相互作用に対して意味のある信号を出すかの検証

## 現在の状況

### 完了したフェーズ
- **Phase 1 (F/G事前訓練)**: ✅ **完了** ([詳細](experiments/phase1_basic_adjunction/))
  - 形状の再構成タスクでF/Gを訓練
  - 問題: ηが主に視点変化に反応し、物理的相互作用への反応が鈍い

### 進行中のフェーズ
- **Phase 1.5 (F/Gの改修と再訓練)**: 🔄 **進行中**
  - FunctorF_v2: 近傍構造の導入 + 目的条件付け
  - PyBulletで「形状 + 行動 → 結果」のデータを収集して再訓練
  - 目標: ηが物理的相互作用に対して意味のある信号を出すようにする

### 今後のフェーズ
- **Phase 2.1 (Agent Cとの統合)**: 📋 **計画中**
  - Module M（目的機構）の実装
  - Agent CとF/Gのend-to-end訓練
  - タスク報酬 + 内発的報酬の二重構造

## リポジトリ構造

```
adjunction-model/
├── README.md                          # このファイル
├── TODO.md                            # 現在のタスクリスト
├── ARCHITECTURE.md                    # アーキテクチャの詳細
├── docs/                              # 理論・設計ドキュメント
│   ├── THEORY.md                      # 理論的背景
│   └── ROADMAP.md                     # 開発ロードマップ
├── research/                          # 哲学的理論調査と設計ノート
│   ├── philosophy_of_purpose_notes.md # 哲学的理論の調査ノート
│   └── PURPOSE_MECHANISM_DESIGN.md    # Module Mの設計
├── src/                               # ソースコード
│   ├── models/                        # F/G/Agent Cのモデル定義
│   ├── data/                          # データセット・データローダー
│   └── training/                      # 訓練ループ・損失関数
└── experiments/                       # 実験コード・結果
    ├── phase1_basic_adjunction/       # Phase 1の実験
    ├── phase2.1_trajectory_prediction/ # Phase 2.1の実験
    │   └── step1_eta_validation/      # Step 1: ηの検証
    └── archived/                      # アーカイブされた実験
```

## 重要な発見

### Step 1: ηの物理的意味の検証（2026年2月18日）

**発見**: Phase 1で訓練されたF/Gの再構成誤差ηは、**視点変化に最も敏感に反応し、物理的相互作用（押す、倒す、回転）への反応は鈍い**。

**原因**: F/Gが学習しているのは「アフォーダンス」ではなく「視点からの形状再構成」という低レベルな特徴。

**対応**: Phase 1.5でF/Gを改修し、物理的相互作用を含むデータで再訓練する。

詳細は [Step 1の結果](./experiments/phase2.1_trajectory_prediction/step1_eta_validation/RESULTS_FIXED.md) を参照してください。

### 目的機構の必要性（2026年2月19日）

**洞察**: アフォーダンスは「オブジェクトの性質」ではなく「オブジェクトとエージェントの関係」である。エージェントの**目的**なしにアフォーダンスは成立しない。

**設計**: Module M（目的機構）を導入し、以下の3つのサブコンポーネントで構成：
1. **Concern State**: エージェントの関心事を表現（ハイデガーの「〜のために存在する」に対応）
2. **Grip Monitor**: 最適からの逸脱を検出（メルロ＝ポンティの「最大把握」に対応）
3. **Salience Modulator**: アフォーダンスのランドスケープをフィールドに変換（SIFの「選択的開放性」に対応）

詳細は [目的機構の設計](./research/PURPOSE_MECHANISM_DESIGN.md) を参照してください。

## クイックスタート

### 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/type37c/adjunction-model.git
cd adjunction-model

# 依存パッケージのインストール
pip install torch torchvision numpy matplotlib
pip install pybullet scipy  # Phase 2.1用
```

### Phase 1の実験を再現

```bash
cd experiments/phase1_basic_adjunction
python run.py
```

### Step 1の実験を再現

```bash
cd experiments/phase2.1_trajectory_prediction/step1_eta_validation
python observe_eta.py
python analyze_results.py
```

## 貢献

このプロジェクトへの貢献に興味がある方は、まず主要ドキュメントをお読みください。バグ報告や改善提案は、GitHubのIssuesで受け付けています。

## ライセンス

このプロジェクトは研究目的で公開されています。商用利用については、事前にご連絡ください。

## 引用

このプロジェクトを研究で使用する場合は、以下のように引用してください：

```
Physical-Semantic Adjunction Model
https://github.com/type37c/adjunction-model
```

## 謝辞

本プロジェクトは、ハイデガー、メルロ＝ポンティ、ギブソン、Rietveld、Kiverstein、Bruineberg、Di Paoloらの哲学的・認知科学的研究に多大な影響を受けています。
