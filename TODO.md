# TODOリスト

**最終更新:** 2026年2月19日

## 現在の状況

### 完了したフェーズ ✅
- Phase 1: F/G事前訓練
- Phase 1.5: F/G再訓練（物理的相互作用への対応）
- Step 2 v2: Agent C再設計（環境の根本的改善）
- Dynamic F/G実験（時系列点群での訓練）

### 主要な発見（2026年2月19日）

#### 1. 随伴の成立条件
表現空間の整合性だけでは不十分。以下が必要：
- 情報の有用性: Affordanceがタスクに関連する情報を含む
- 予測精度: ηが安定し、発散しない
- 分布の一致: 訓練データとテストデータの分布が整合
- 次元の適切性: 高次元すぎると学習困難

#### 2. タスクミスマッチ
Reachingタスクは単純すぎて、F/Gの能力を検証できない。F/Gは以下のような複雑なタスクで検証すべき：
- 把持 (Grasping): 形状に応じた把持点の推論
- 組み立て (Assembly): 部品間の幾何学的制約
- 道具使用 (Tool use): 道具のaffordanceの理解

#### 3. 知能 vs 知性
現在のモデルは知性寄り（自律的探索、好奇心駆動）だが、タスクが与えられないと目的地に着かない。
- 解決策の候補: 命令解釈機構の追加（「何をするか」を与える）
- 「どうやるか」はエージェントが自分で見つける

---

## 次のステップ（優先順位順）

### 優先度: 最高 🔴

#### 1. タスクの複雑化
**目的:** F/Gの真の能力を評価できる環境を構築

**候補タスク:**
- **把持 (Grasping)**: 形状に応じた把持点の推論が必要
- **組み立て (Assembly)**: 部品間の幾何学的制約の理解が必要
- **道具使用 (Tool use)**: 道具のaffordanceの理解が必要

**実装:**
```
experiments/complex_tasks/
├── grasping/          # 把持タスク
├── assembly/          # 組み立てタスク
└── tool_use/          # 道具使用タスク
```

**期待される成果:**
- F/Gが形状の構造的特徴を活用できるか検証
- Affordanceが実際にタスクに有用かを評価

---

### 優先度: 高 🟠

#### 2. 環境とベースラインの改善
**目的:** Step 2 v2のベースラインが不安定（最高25%→最終0%）だった問題を解決

**改善案:**
1. **ハイパーパラメータ調整**
   - 学習率、バッチサイズ、エントロピー係数
   - PPOのクリッピング範囲

2. **報酬関数の改善**
   - 距離改善量のスケーリング
   - 成功閾値の調整

3. **訓練の安定化**
   - 報酬の正規化
   - Value networkの改善

**期待される成果:**
- 安定したベースライン（成功率10%以上）
- F/G統合の効果を正確に評価できる基盤

---

#### 3. 命令解釈機構の設計検討
**目的:** 「何をするか」を指定する機構の必要性を検討

**検討事項:**
1. **命令の形式**
   - シンボリック（例: "reach", "grasp"）
   - 言語（例: "grasp the red cup"）
   - デモンストレーション（軌跡から推論）

2. **Goal vectorの設計**
   - η-grounded: Goal vectorがηの重み付けを指定
   - 言語埋め込み: CLIPなどの事前訓練モデルを活用
   - 学習可能: Goal embeddingを訓練データから学習

3. **実装の段階**
   - Stage 1: シンボリック命令（最もシンプル）
   - Stage 2: 言語層の導入（CLIPなど）
   - Stage 3: マルチモーダル基盤モデル（GPT-4V, Geminiなど）

**注意:** 言語層の導入は確定事項ではなく、複数の選択肢の一つ。まずはシンボリック命令から検討。

**参考:** [理論的議論](./THEORETICAL_DISCUSSIONS.md)

---

### 優先度: 中 🟡

#### 4. Multi-task F/Gの訓練
**目的:** 複数タスクで共通のaffordance表現を学習

**実装:**
```python
# 複数タスクのデータを収集
tasks = ["reach", "grasp", "push"]
for task in tasks:
    collect_data(task, episodes=500)

# Multi-taskでF/Gを訓練
train_fg_multitask(tasks, goal_vectors)
```

**期待される成果:**
- タスク間の共通構造を学習
- ゼロショット汎化の可能性

---

#### 5. Module Mの実装
**目的:** 目的機構の完全な統合

**コンポーネント:**
1. **Concern State**: エージェントの関心事を表現
2. **Grip Monitor**: 最適からの逸脱を検出
3. **Salience Modulator**: Affordanceのランドスケープをフィールドに変換

**実装:**
```
src/models/module_m/
├── concern_state.py
├── grip_monitor.py
└── salience_modulator.py
```

**参考:** [目的機構の設計](./research/PURPOSE_MECHANISM_DESIGN.md)

**詳細タスク:**
- [ ] **Concern State**の実装
  - [ ] GRUベースの関心状態更新
  - [ ] 基本的な関心の初期化（自己維持に相当）
  - [ ] 関心状態の可視化ツール

- [ ] **Grip Monitor**の実装
  - [ ] ηの変化パターンの統合
  - [ ] 行動結果のフィードバック処理
  - [ ] 把握度の履歴管理（GRU）
  - [ ] 最適からの逸脱度の計算

- [ ] **Salience Modulator**の実装
  - [ ] 注意機構（Multi-head Attention）
  - [ ] Concern Stateからクエリへの射影
  - [ ] アフォーダンスランドスケープからフィールドへの変換
  - [ ] 顕著性重みの可視化

---

### 優先度: 低 🟢

#### 6. 階層的な目的分解
**目的:** 抽象的命令を具体的なηプロファイルに変換

**実装:**
```python
class GoalInterpreter:
    def interpret(self, abstract_command):
        # 抽象的命令 → ηプロファイル
        return eta_profile
```

**期待される成果:**
- 「grasp」→「まず近づく、次に把持」のような階層的な目的分解

---

#### 7. マルチモーダル基盤モデルの統合
**目的:** GPT-4V, Geminiなどの活用

**実装:**
```python
# 基盤モデルからgoal embeddingを取得
goal_embedding = foundation_model.encode("grasp the red cup")
affordance = F(point_cloud, goal_embedding)
```

**期待される成果:**
- ゼロショット汎化
- 自然言語での命令

---

## 技術的負債

### コードの整理
- [ ] Step 2 v2とDynamic F/Gのコードを統合
- [ ] 共通のユーティリティ関数を`src/utils/`に移動
- [ ] 訓練スクリプトのテンプレート化

### ドキュメント
- [x] 実験結果の総括（EXPERIMENT_SUMMARY.md）
- [x] 理論的議論の整理（THEORETICAL_DISCUSSIONS.md）
- [x] READMEの更新
- [ ] 各実験のREADMEに「次のエージェントへの引き継ぎ」セクションを追加
- [ ] ARCHITECTURE.mdの更新（Module Mの追加）
- [ ] ROADMAP.mdの更新（Phase 1.5とPhase 2.1の詳細）

### テスト
- [ ] F/Gの単体テスト
- [ ] Agent Cの単体テスト
- [ ] 環境の単体テスト

---

## 長期的な目標

1. **未知の形状に対するアフォーダンス推論**
   - 訓練時に見たことのない形状でテスト
   - ゼロショット汎化の評価

2. **目的の創発**
   - エージェントが自分で目的を見出す
   - 好奇心駆動の探索

3. **記号接地問題の解決**
   - 言語と物理的相互作用の結びつき
   - 意味の創発

---

## 設計メモ

### Phase 1.5の設計判断
- **近傍構造のk値**: 16を初期値とする（調整可能）
- **目的ベクトルの次元**: 16を初期値とする
- **行動の種類**: 5種類（push, pull, rotate, lift, topple）
- **訓練データサイズ**: 5オブジェクト × 5行動 × 100エピソード = 2500エピソード

### Phase 2.1の設計判断
- **Concern Stateの次元**: 32を初期値とする
- **Grip Monitorの隠れ層**: 128を初期値とする
- **Salience Modulatorのヘッド数**: 4を初期値とする
- **損失関数の初期重み**: λ_recon=1.0, λ_task=1.0, λ_grip=0.5, λ_div=0.1

### 哲学的対応の確認
各コンポーネントが哲学的概念に対応していることを常に意識する：
- Concern State ↔ Worumwillen（ハイデガー）
- Grip Monitor ↔ 最大把握（メルロ＝ポンティ）
- Salience Modulator ↔ フィールド（SIF）
- η ↔ 不調和度（Bruineberg/Rietveld）

---

## 今日の議論から（2026年2月19日）

**Goal vectorの意味:**
- シンボルグラウンディング問題: Goal vectorはただの数値
- η-groundingで解決可能: Goal vectorがηの重み付けを指定
- 言語層は一つの選択肢（確定事項ではない）

**知能 vs 知性:**
- 現在のモデルは知性寄り（自律的探索、好奇心駆動）
- しかし、タスクが与えられないと目的地に着かない
- 命令解釈機構が必要（「何をするか」を与える）
- 「どうやるか」はエージェントが自分で見つける

**タスクミスマッチ:**
- Reachingは単純すぎる
- F/Gは複雑なタスク（把持、組み立て、道具使用）で検証すべき

---

**次のエージェントへ:**
- まず[EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md)と[THEORETICAL_DISCUSSIONS.md](./THEORETICAL_DISCUSSIONS.md)を読んでください
- 優先度: 最高のタスクから着手してください
- 不明点があれば、各実験のREADMEとANALYSIS.mdを参照してください
