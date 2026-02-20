# Revised Phase 2: Bodily Constraints (Not Physical Laws)
# 修正版Phase 2：身体的制約（物理法則ではない）

**Date**: February 20, 2026  
**Status**: Back to original plan

---

## 重要な指摘への対応

### ユーザーの指摘

> 形や制約に対して行動がどうなるか、とかは適切にシミュレーション環境内で学習すれば必然的に経験則的に学習はできるようになる。ただ、それは物理法則を理解しているというわけではない。
> 
> それはシミュレータで大規模実験をする時に勝手に分かることだから今やらなくてもいい。
> 
> それよりももともとエージェントが右手を怪我してる時にどういう行動を取るかとかがもとの計画だったはず。

### 正しい理解

- ❌ **物理制約**（重力、摩擦）→ 経験則で学習できる、今やる必要なし
- ✅ **身体的制約**（右手の怪我）→ 保留構造の本質的テスト、これが本来の計画

---

## 初期実験ノートの該当箇所

### Phase 2の本来の設定（初期ノートより）

> **Phase 2: 制約下での創造的解決とCoherence Breakdownの検証（設定B）**
> 
> **目的**: 既存の随伴構造が破綻するような制約が与えられた際、保留構造が起動し、新しい随伴構造（動作）を創発できることを示す。
> 
> **設定**: Phase 1の環境に、エージェントの状態Cに影響を与える制約（例：重力の変化、接触面の摩擦係数の変化、**エージェントの身体の一部が使用不能になる**）を追加。これによりCoherence signalが急増する状況を作り出す。

### 初期ノートの具体例

> たとえば、**右腕を怪我した状態で、静かにしなければならない環境でスーツケースを運ぶ**場面を考える。「引きずる」という通常の代替手段は騒音を生むため使えない。身体状態フィルターと文脈フィルターが干渉し、可換性が崩れる。このとき、エージェントは既存の動作プリミティブを新しい方法で合成し（たとえば「両膝で抱えて歩く」）、非可換性を解消しなければならない。

---

## Phase 2の正しい設計：身体的制約

### Concept / 概念

**物理法則の変化ではなく、エージェント自身の身体状態の変化**

- ❌ 環境の物理パラメータが変わる（重力、摩擦）
- ✅ エージェントの身体の一部が使用不能になる（右手の怪我、左足の骨折）

### Why This Matters / なぜ重要か

#### 1. 保留構造の本質的テスト

**物理制約**:
- 環境が変わる → 経験則で学習できる
- 「このくらいの力で押せば動く」という統計的関係を学習

**身体的制約**:
- エージェント自身が変わる → **自己モデルの再構成が必要**
- 「自分は何ができるか」という根本的な問い直し
- これこそが保留構造の本質

#### 2. 創造性の発生条件

**物理制約**:
- 既存の行動を微調整すればよい
- 創造性は不要

**身体的制約**:
- 既存の行動プリミティブが使えない
- **新しい組み合わせを創発する必要**がある
- 「両膝で抱えて歩く」のような非自明な解

#### 3. Coherence Breakdownの明確性

**物理制約**:
- ηがじわじわ上昇（物理が少し違う）
- Breakdownが曖昧

**身体的制約**:
- ηが急激に上昇（右手が使えない！）
- **明確なBreakdown**
- 保留構造の起動が観測しやすい

---

## Revised Phase 2 Experimental Design / 修正版Phase 2実験設計

### Phase 2.1: Right Hand Injury (右手の怪我)

#### Scenario / シナリオ

エージェントは脱出部屋にいる。物体を操作して脱出する必要がある。

**Phase 0-1**: 両手が使える状態で学習
- Push: 右手で押す
- Pull: 右手で引く
- Rotate: 両手で回転

**Phase 2.1**: **右手が使用不能**
- 右手を使う行動が実行できない
- 左手のみ、または両足、または体全体を使う必要がある

#### Implementation / 実装

```python
class EscapeRoomEnv:
    def __init__(self, ..., disabled_limbs=None):
        """
        Args:
            disabled_limbs: List of disabled limbs, e.g., ['right_hand']
        """
        self.disabled_limbs = disabled_limbs or []
    
    def step(self, action):
        # Check if action requires disabled limb
        if self._requires_disabled_limb(action):
            # Action fails or has reduced effect
            reward = -1.0
            info = {'failure_reason': 'disabled_limb'}
            return obs, reward, done, info
        
        # Normal execution
        ...
```

#### Expected Behavior / 期待される動作

1. **Initial attempt**: エージェントは右手を使う行動を試みる
2. **Failure**: 行動が失敗する
3. **η spikes**: Coherence signalが急上昇（「右手で押す」→「押せない」）
4. **Suspension**: 保留構造が起動
5. **Exploration**: 左手、両足、体全体を使う行動を探索
6. **Adaptation**: F/Gが新しい身体状態に適応
7. **Recovery**: 新しい行動で成功

#### Metrics / 指標

| Metric | Expected Value |
|--------|----------------|
| Initial η | > 0.2（大幅な上昇）|
| Suspension count | > 10（頻繁に発動）|
| F/G updates | > 5（適応が必要）|
| Final η | < 0.1（適応後に回復）|
| Success rate | > 40%（Phase 0-1より低いが0ではない）|

### Phase 2.2: Left Leg Injury (左足の怪我)

#### Scenario / シナリオ

**Phase 2.2**: **左足が使用不能**
- 移動が制限される
- 物体を運ぶ行動が困難
- バランスを取る必要がある

#### Expected Behavior / 期待される動作

- 片足でバランスを取りながら物体を操作
- または、座って操作
- または、物体を転がす

### Phase 2.3: Multiple Constraints (複数制約)

#### Scenario / シナリオ

**Phase 2.3**: **右手と左足が同時に使用不能**
- さらに困難な状況
- より創造的な解決が必要

#### Expected Behavior / 期待される動作

- 左手と右足のみを使う
- または、体全体を使う（頭で押す、背中で押す）
- または、物体を口で咥える（もし可能なら）

---

## Escape Room Environment Improvements / 脱出部屋環境の改善

### Current Issues / 現在の問題点

1. **行動が単純すぎる**
   - Push, Pull, Rotate の3つのみ
   - 身体部位が明示されていない

2. **物体が単純すぎる**
   - 1つの物体のみ
   - 操作が単純（正しい行動を選ぶだけ）

3. **脱出条件が不明確**
   - 「正しい行動をすれば脱出」では創造性が測れない

### Proposed Improvements / 改善案

#### 1. Explicit Body Parts (身体部位の明示化)

**Current**:
```python
action = 0  # Push
action = 1  # Pull
action = 2  # Rotate
```

**Proposed**:
```python
action = {
    'limb': 'right_hand',  # or 'left_hand', 'right_foot', 'left_foot', 'torso'
    'primitive': 'push',   # or 'pull', 'rotate', 'lift'
    'direction': [1, 0, 0],
    'force': 10.0
}
```

**Benefits**:
- 身体部位が明示される
- 怪我の制約が自然に表現できる
- より細かい行動の組み合わせが可能

#### 2. Multi-Step Tasks (複数ステップのタスク)

**Current**:
- 1つの物体に1つの行動 → 脱出

**Proposed**:
- 複数の物体を順番に操作 → 脱出

**Example**:
1. レバーを引く → ドアのロックが外れる
2. ボタンを押す → ドアが開く
3. ドアを通過 → 脱出成功

**Benefits**:
- より複雑なタスク
- 創造性が測りやすい
- 保留と適応のプロセスが観測しやすい

#### 3. Noisy Environment (騒音制約) - オプション

**Scenario** (初期ノートより):
> 静かにしなければならない環境でスーツケースを運ぶ

**Implementation**:
```python
class EscapeRoomEnv:
    def __init__(self, ..., noise_constraint=False):
        self.noise_constraint = noise_constraint
    
    def step(self, action):
        if self.noise_constraint:
            noise_level = self._compute_noise(action)
            if noise_level > threshold:
                reward -= noise_penalty
```

**Benefits**:
- 文脈フィルターの実装
- 身体状態フィルターとの干渉をテスト
- より現実的なシナリオ

**Decision**: Phase 2.1-2.3の後で実装（脱線しないため）

---

## Implementation Priority / 実装優先度

### Phase 2.1: Right Hand Injury (最優先)

**Focus**: 身体的制約の基本的なテスト

**Why**: 
- 保留構造の本質的テスト
- 実装が比較的簡単
- 明確な成功基準

**Timeline**: 2-3日

### Phase 2.2: Left Leg Injury (次の優先度)

**Focus**: 異なる身体部位の制約

**Why**:
- Phase 2.1の結果を確認
- 汎化能力のテスト

**Timeline**: 1-2日（Phase 2.1の後）

### Phase 2.3: Multiple Constraints (オプション)

**Focus**: 複数制約の同時適用

**Why**:
- より困難な状況
- しかし論文には必須ではない

**Timeline**: 必要に応じて

### Environment Improvements (並行)

**Focus**: 身体部位の明示化、複数ステップタスク

**Why**:
- Phase 2.1-2.3の実装に必要
- より現実的な環境

**Timeline**: Phase 2.1と並行して実装

---

## Revised Publication Roadmap / 修正版論文化ロードマップ

### Week 1: Experiments

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Baseline implementations | 12 |
| 3-4 | **Phase 2.1: Right hand injury** | 12 |
| 5 | **Phase 2.2: Left leg injury (optional)** | 6 |
| 6-7 | Multiple seeds + data analysis | 8 |
| **Total** | | **38** |

### Week 2: Writing

(変更なし)

---

## Expected Results / 期待される結果

### Quantitative / 定量的

| Method | Phase 0 | Phase 1 | Phase 2.1 (right hand) |
|--------|---------|---------|------------------------|
| **Ours** | 58% | 62% | **45%** ✅ (低下するが0ではない) |
| Static F/G | 58% | 42% | **10%** ❌ (大幅に低下) |
| PPO | 54% | 35% | **5%** ❌ (ほぼ失敗) |

### Qualitative / 定性的

- ✅ ηが急激に上昇（右手が使えない！）
- ✅ 保留が頻繁に発動（> 10回）
- ✅ F/G適応が実行される（> 5回）
- ✅ 新しい行動の創発（左手、両足を使う）
- ✅ 最終的に成功率が回復（45%程度）

### Why This is Better / なぜこれが優れているか

1. **理論の本質をテスト**: 保留構造、創造性、自己モデルの再構成
2. **明確なBreakdown**: ηの急激な上昇が観測しやすい
3. **論文のストーリー**: 「エージェントは自分の身体が変わったときに適応できるか？」
4. **独自性**: 既存研究にない視点

---

## Summary / まとめ

### 重要な修正

- ❌ **物理制約**（重力、摩擦）→ 削除
- ✅ **身体的制約**（右手の怪我、左足の怪我）→ Phase 2の核心

### 初期計画への回帰

初期実験ノートの本来の意図に立ち返る：

> エージェントの身体の一部が使用不能になる

> 右腕を怪我した状態で、静かにしなければならない環境でスーツケースを運ぶ

### 次のステップ

1. Phase 2.1の実装（右手の怪我）
2. 環境の改善（身体部位の明示化）
3. ベースライン比較
4. 複数シード実行
5. 論文執筆

---

**Ready to implement the correct Phase 2?** 🚀
