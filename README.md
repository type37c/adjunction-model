# 物理的意味的随伴モデル (Physical-Semantic Adjunction Model)

**バージョン:** 5.0  
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
- **[実験結果総括](./EXPERIMENT_SUMMARY.md)**: 2026年2月19日までの全実験結果のまとめ
- **[理論的議論](./THEORETICAL_DISCUSSIONS.md)**: 言語層、goal-grounding、命令解釈機構に関する議論
- **[Phase 1: F/G事前訓練](./experiments/phase1_basic_adjunction/)**: 基本的な随伴の訓練と評価
- **[Phase 1.5: F/G再訓練](./experiments/phase1.5_fg_retraining/)**: 物理的相互作用を含むデータでF/Gを再訓練
- **[Step 2 v2: Agent C再設計](./experiments/step2_v2_redesign/)**: 環境とAgent Cの根本的な再設計
- **[Dynamic F/G実験](./experiments/dynamic_fg/)**: 動的表現の学習と検証

## 現在の状況

### 完了したフェーズ

#### Phase 1: F/G事前訓練 ✅
- 形状の再構成タスクでF/Gを訓練
- **問題**: ηが主に視点変化に反応し、物理的相互作用への反応が鈍い

#### Phase 1.5: F/G再訓練 ✅
- FunctorF_v2: 近傍構造の導入 + 目的条件付け
- PyBulletで「形状 + 行動 → 結果」のデータを収集して再訓練
- **成功**: ηが物理的相互作用に対して意味のある信号を出すようになった

#### Step 2 v2: Agent C再設計 ✅
- **環境の再設計**: 位置制御、改善された報酬関数、豊かな状態表現
- **ベースライン成功**: 報酬2.69、成功率3%（Phase 2.1の-84930から大幅改善）
- **F/G統合失敗**: F/G特徴量が学習を阻害（報酬-6.22、成功率0%）
- **根本原因**: 表現空間（静的形状）とタスク空間（動的運動）のミスマッチ

#### Dynamic F/G実験 ✅
- 時系列点群でF/Gを訓練（動的な到達可能性を表現）
- **F/G訓練成功**: Loss 0.123 → 0.068
- **Agent C統合失敗**: Dynamic F/Gも学習を改善せず（報酬-4.78、成功率0%）
- **新たな洞察**: タスクミスマッチ、次元の呪い、ηの発散

### 重要な発見

#### 1. 随伴の成立条件（2026年2月19日）

随伴が成立するには、表現空間の整合性だけでは不十分。以下が必要：

1. **情報の有用性**: Affordanceがタスクに関連する情報を含む
2. **予測精度**: ηが安定し、発散しない
3. **分布の一致**: 訓練データとテストデータの分布が整合
4. **次元の適切性**: 高次元すぎると学習困難

#### 2. 知能 vs 知性（2026年2月19日）

現在のモデルは「知性寄り」：
- **知能**: 与えられたタスクを効率的に解く
- **知性**: 何が問題かを自分で見出す

ηの改善を駆動力として、自分で「何を見るか」を選ぶ。しかし、タスクが与えられないと、目的地に着かない。

**解決策**: 命令解釈機構の追加
- 命令は「何をするか」を与える
- 「どうやるか」はエージェントが自分で見つける

#### 3. タスクの複雑性（2026年2月19日）

Reachingタスクは単純すぎて、F/Gの能力を検証できない。F/Gは以下のような複雑なタスクで検証すべき：
- **把持 (Grasping)**: 形状に応じた把持点の推論
- **組み立て (Assembly)**: 部品間の幾何学的制約
- **道具使用 (Tool use)**: 道具のaffordanceの理解

### 次のステップ

詳細は [TODO.md](./TODO.md) を参照。

#### 短期（優先度: 高）
1. **言語層の導入**
   - CLIP text encoderを使用
   - 命令 → goal embedding → affordance
   - Multi-task環境（Reaching, Grasping, Pushing）

2. **Goal-conditioned F/G**
   - 言語条件付きaffordance
   - η-grounded goal vectors

#### 中期（優先度: 中）
1. **複雑なタスクでの検証**
   - 把持、組み立て、道具使用
   - F/Gの真の能力を評価

2. **Module Mの実装**
   - Concern State, Grip Monitor, Salience Modulator
   - 目的機構の完全な統合

#### 長期（優先度: 低）
1. **階層的な目的分解**
   - 抽象的命令 → 具体的なηプロファイル
   - Goal profileの学習

2. **マルチモーダル基盤モデルの統合**
   - GPT-4V, Geminiなどの活用
   - ゼロショット汎化

## リポジトリ構造

```
adjunction-model/
├── README.md                          # このファイル
├── TODO.md                            # 現在のタスクリスト
├── EXPERIMENT_SUMMARY.md              # 実験結果総括
├── THEORETICAL_DISCUSSIONS.md         # 理論的議論
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
    ├── phase1.5_fg_retraining/        # Phase 1.5の実験
    ├── step2_v2_redesign/             # Step 2 v2の実験
    ├── dynamic_fg/                    # Dynamic F/Gの実験
    └── archived/                      # アーカイブされた実験
```

## クイックスタート

### 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/type37c/adjunction-model.git
cd adjunction-model

# 依存パッケージのインストール
pip install torch torchvision numpy matplotlib
pip install pybullet scipy  # PyBullet実験用
```

### 最新の実験を再現

#### Step 2 v2（Agent C再設計）

```bash
cd experiments/step2_v2_redesign

# ベースライン訓練
python train.py --mode baseline --episodes 1500

# F/G-enhanced訓練
python train.py --mode with_fg --episodes 1500

# 結果の比較
python run_experiment.py
```

#### Dynamic F/G実験

```bash
cd experiments/dynamic_fg

# Dynamic F/Gの訓練
python scripts/train_agent_c_with_online_fg.py --episodes 1000

# Agent C v3の訓練
python scripts/train_agent_c_v3.py --mode baseline --episodes 1500
python scripts/train_agent_c_v3.py --mode with_fg --episodes 1500
```

## エージェントへの引き継ぎガイド

将来のエージェントがこのプロジェクトを引き継ぐ際は、以下の順序でドキュメントを読むことを推奨します：

1. **[EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md)**: 全実験の結果と主要な発見
2. **[THEORETICAL_DISCUSSIONS.md](./THEORETICAL_DISCUSSIONS.md)**: 理論的課題と解決策
3. **[TODO.md](./TODO.md)**: 次のステップと優先順位
4. **各実験のREADME**: 実験の詳細な設計と結果
   - [Step 2 v2](./experiments/step2_v2_redesign/README.md)
   - [Dynamic F/G](./experiments/dynamic_fg/README.md)

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
