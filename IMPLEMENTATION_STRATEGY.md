# 実装戦略: 最速で理論を完全実装する

**作成日:** 2026年2月20日  
**目標:** 初期実験ノートの理論を最速で完全実装し、Phase 0-2を検証する  
**予定期間:** 2週間以内

---

## 戦略の核心

### 1. 既存コードの最大活用

既存のコードベースには使える要素が多数存在：

- ✅ `FunctorF`, `FunctorG`: 基本的なF/G実装
- ✅ `SyntheticAffordanceDataset`: データ生成
- ✅ PyBullet環境の実装例（phase2.1, phase1.5）
- ✅ PPOの実装（agent_c.py）

**方針:** ゼロから書かずに、既存コードを統合・拡張する

### 2. 並行実装ではなく、依存関係順に高速実装

4週間の計画は保守的すぎる。依存関係を明確にし、順次高速実装：

```
Day 1-2:  双方向F/G (η + ε) の実装と訓練
Day 3-4:  保留構造と提案生成Agentの実装
Day 5-6:  脱出部屋環境の実装
Day 7-8:  訓練ループの統合とPhase 0実験
Day 9-10: Phase 1実験（未知形状への般化）
Day 11-12: Phase 2実験（保留構造の検証）
Day 13-14: 結果分析とドキュメント整備
```

### 3. シンプルな実装を優先

完璧を目指さず、動くものを最速で作る：

- 点群: 最初は簡易実装（カメラなしでも可）
- 物体: URDFではなくプリミティブ形状
- 評価: 複雑な指標ではなく、成功率とη/εの推移

---

## リポジトリ構造の再編

### 新しい構造

```
adjunction-model/
├── core/                          # コアコンポーネント（新規）
│   ├── models/
│   │   ├── bidirectional_fg.py    # η + ε
│   │   ├── suspension.py          # 保留構造
│   │   └── proposal_agent.py      # 提案生成Agent
│   ├── envs/
│   │   └── escape_room.py         # 脱出部屋環境
│   └── training/
│       ├── phase0_basic.py        # Phase 0訓練
│       ├── phase1_generalization.py  # Phase 1訓練
│       └── phase2_suspension.py   # Phase 2訓練
├── src/                           # 既存コード（保持）
│   ├── models/                    # 再利用
│   ├── data/                      # 再利用
│   └── ...
├── experiments/                   # 既存実験（アーカイブ）
│   └── ...
├── results/                       # 新しい実験結果（新規）
│   ├── phase0/
│   ├── phase1/
│   └── phase2/
├── docs/                          # ドキュメント
│   └── ...
└── scripts/                       # 実行スクリプト（新規）
    ├── run_phase0.py
    ├── run_phase1.py
    └── run_phase2.py
```

### 設計原則

1. **core/**: 新しい理論実装の中心。既存コードから独立
2. **src/**: 既存の有用なコンポーネントを保持、必要に応じて再利用
3. **scripts/**: 実験実行を簡単にするエントリーポイント
4. **results/**: 実験結果を明確に分離

---

## 実装の詳細

### Phase 0: 基礎学習（Day 1-8）

**目標:** F/Gが既知の形状-行動ペアを学習できることを示す

**実装:**

1. **双方向F/G** (Day 1-2)
   - 既存の`FunctorF`, `FunctorG`を拡張
   - `F_inv`, `G_inv`を追加
   - η + ε損失で訓練

2. **保留構造** (Day 3-4)
   - ηの閾値判定
   - 保留バッファ
   - 保留中のF/G更新

3. **提案生成Agent** (Day 3-4)
   - 複数の行動候補を生成
   - εでフィルタリング
   - ηで選択

4. **脱出部屋環境（簡易版）** (Day 5-6)
   - 立方体、球、円柱の3つの物体
   - Push, Pull, Rotateの3つの操作
   - 成功判定: 正しい操作をしたらドアが開く

5. **Phase 0訓練** (Day 7-8)
   - 既知の形状で訓練
   - η, εの収束を確認
   - 成功率を測定

**成功基準:**
- η < 0.1
- ε < 0.1
- 成功率 > 80%（既知形状）

---

### Phase 1: 未知形状への般化（Day 9-10）

**目標:** 未知の形状に対してaffordanceを推論できることを示す

**実装:**

1. **新しい形状の追加**
   - レバー（長方形）
   - ボタン（扁平な円柱）
   - ノブ（球 + 円柱）

2. **Phase 1訓練**
   - 訓練: 立方体、球、円柱
   - テスト: レバー、ボタン、ノブ
   - 提案生成メカニズムで未知形状に対応

**成功基準:**
- 未知形状に対する成功率 > ランダムベースライン
- ηが未知形状で一時的に上昇するが、保留構造で対応
- 適切な操作を発見できる

---

### Phase 2: 保留構造の検証（Day 11-12）

**目標:** Coherence breakdownが保留構造を起動し、創造的解決を生むことを示す

**実装:**

1. **制約の追加**
   - 重力の変化（0.5x, 2x）
   - 摩擦係数の変化
   - 物体の質量変化

2. **Phase 2訓練**
   - 通常環境で訓練
   - 制約環境でテスト
   - 保留構造の起動を観察

**成功基準:**
- Coherence breakdownの検出（ηの急増）
- 保留構造の起動（行動が止まる）
- F/Gの適応（ηが低下）
- 制約を克服してタスクを達成

---

## 実装の優先順位

### 最優先（Day 1-2）

1. **双方向F/Gの実装**
   - `core/models/bidirectional_fg.py`
   - 既存の`FunctorF`, `FunctorG`を統合
   - `F_inv`, `G_inv`を追加
   - 訓練スクリプト: `scripts/train_bidirectional_fg.py`

### 高優先（Day 3-6）

2. **保留構造と提案Agent**
   - `core/models/suspension.py`
   - `core/models/proposal_agent.py`

3. **脱出部屋環境**
   - `core/envs/escape_room.py`
   - 既存のPyBullet実装を参考に

### 中優先（Day 7-12）

4. **訓練ループとPhase 0-2実験**
   - `core/training/phase0_basic.py`
   - `core/training/phase1_generalization.py`
   - `core/training/phase2_suspension.py`

### 低優先（Day 13-14）

5. **結果分析とドキュメント**
   - 実験結果のプロット
   - `RESULTS.md`の作成
   - 論文用の図表

---

## 技術的な決定事項

### 1. 点群の取得

**Phase 0-1:** 物体の頂点から直接サンプリング（カメラなし）

```python
def get_point_cloud_from_object(object_id, num_points=512):
    # 物体のAABBから点をサンプリング
    aabb_min, aabb_max = p.getAABB(object_id)
    points = np.random.uniform(aabb_min, aabb_max, (num_points, 3))
    return points
```

**Phase 2:** カメラからの深度画像（必要に応じて）

### 2. 行動空間

**離散行動:** Push, Pull, Rotate（3つ）

```python
action_space = gym.spaces.Discrete(3)
```

**連続行動への拡張:** Phase 2で必要に応じて

### 3. F/Gの更新

**保留中の更新:**

```python
# 保留バッファから最新N個のサンプルで勾配降下
for _ in range(10):  # 10ステップのfine-tuning
    batch = sample_from_buffer(suspension_buffer, batch_size=32)
    loss = compute_eta(batch)
    loss.backward()
    optimizer.step()
```

### 4. 提案生成

**複数候補の生成:**

```python
# Actor networkから10個の行動候補をサンプリング
proposals = [actor.sample() for _ in range(10)]

# εでフィルタリング
valid_proposals = [p for p in proposals if compute_epsilon(p) < threshold]

# ηで選択
best_proposal = min(valid_proposals, key=lambda p: compute_eta(state, p))
```

---

## リスク管理

### リスク1: 環境の複雑性

**対策:** 最初は超シンプルな環境から始める
- 物体1つ
- 操作1つ
- 成功判定: 物体が動いたらOK

### リスク2: 訓練の不安定性

**対策:** 
- 学習率を小さく（1e-4）
- バッチサイズを大きく（32-64）
- 勾配クリッピング

### リスク3: η/εが収束しない

**対策:**
- Affordance次元を調整（16 → 32 or 8）
- F/Gのアーキテクチャを調整
- 事前訓練を長くする

---

## 成功の定義

### Phase 0の成功

- [x] η < 0.1
- [x] ε < 0.1
- [x] 既知形状での成功率 > 80%

### Phase 1の成功

- [x] 未知形状での成功率 > ランダム（33%）
- [x] 提案生成メカニズムが機能
- [x] 適切な操作を発見

### Phase 2の成功

- [x] Coherence breakdownを検出
- [x] 保留構造が起動
- [x] F/Gが適応してηが低下
- [x] 制約下でタスクを達成

---

## 次のアクション

### 即座に実行（今日中）

1. **リポジトリ構造の作成**
   ```bash
   mkdir -p core/{models,envs,training}
   mkdir -p scripts
   mkdir -p results/{phase0,phase1,phase2}
   ```

2. **双方向F/Gの実装**
   - `core/models/bidirectional_fg.py`を作成
   - 既存の`src/models/functor_f.py`, `functor_g.py`を統合

3. **訓練スクリプトの作成**
   - `scripts/train_bidirectional_fg.py`
   - Phase 1のデータで訓練開始

### 明日実行

4. **保留構造と提案Agentの実装**
5. **脱出部屋環境の骨格**

---

## まとめ

この戦略により、4週間ではなく**2週間以内**に理論の完全実装を達成できる。

**鍵となる決定:**
- 既存コードの最大活用
- シンプルな実装を優先
- 依存関係順に高速実装
- 完璧を目指さず、動くものを最速で

**次のステップ:** リポジトリ構造を作成し、双方向F/Gの実装を開始する。
