# Phase 2 Design and Escape Experiment Improvements
# Phase 2の設計と脱出実験の改善案

**Date**: February 20, 2026  
**Status**: Design phase

---

## Phase 2: Constraints on Known Shapes

### Concept / 概念

Phase 2では、**既知の形状に物理的制約を加える**ことで、保留構造の適応能力をテストします。

**重要な点**:
- 形状は既知（cube, cylinder, sphere）
- しかし物理パラメータが変化（重力、摩擦、質量）
- F/Gは形状を「知っている」が、物理的振る舞いが異なる
- 期待: 保留が発動し、F/Gが新しい物理に適応

### Why This Matters / なぜ重要か

1. **形状 vs 物理の分離**: F/Gが「形状」だけでなく「物理的性質」も学習できるか
2. **適応の範囲**: 保留構造は未知形状だけでなく、未知物理にも対応できるか
3. **汎化の深さ**: 単なる形状マッチングではなく、物理的理解があるか

---

## Phase 2 Experimental Design / Phase 2実験設計

### Constraints to Test / テストする制約

#### 1. Gravity Change (重力変化)

**Scenario**: 
- Phase 0: gravity = -9.8 m/s²（地球）
- Phase 2: gravity = -1.6 m/s²（月面）または -3.7 m/s²（火星）

**Expected Behavior**:
- 物体の落下速度が変わる
- 「押す」行動の効果が変わる
- η上昇 → 保留 → F/G適応

**Implementation**:
```python
env = EscapeRoomEnv(
    object_type=EscapeRoomEnv.CUBE,
    gravity=-1.6,  # Moon gravity
    render=False
)
```

#### 2. Friction Change (摩擦変化)

**Scenario**:
- Phase 0: friction = 0.5（通常）
- Phase 2: friction = 0.1（氷上）または friction = 1.5（粗い表面）

**Expected Behavior**:
- 物体の滑りやすさが変わる
- 「回転」行動の効果が変わる
- η上昇 → 保留 → F/G適応

**Implementation**:
```python
env = EscapeRoomEnv(
    object_type=EscapeRoomEnv.CYLINDER,
    friction=0.1,  # Icy surface
    render=False
)
```

#### 3. Mass Change (質量変化)

**Scenario**:
- Phase 0: mass = 1.0 kg（通常）
- Phase 2: mass = 0.1 kg（軽い）または mass = 10.0 kg（重い）

**Expected Behavior**:
- 物体の動かしやすさが変わる
- 「引く」行動の効果が変わる
- η上昇 → 保留 → F/G適応

**Implementation**:
```python
env = EscapeRoomEnv(
    object_type=EscapeRoomEnv.SPHERE,
    mass=10.0,  # Heavy object
    render=False
)
```

### Experimental Protocol / 実験プロトコル

1. **Pre-training**: Phase 0で訓練済みのモデルをロード
2. **Constraint introduction**: 1つの制約を変更
3. **Observation**: ηの変化を監視
4. **Suspension**: η > 0.1で保留発動
5. **Adaptation**: F/Gをファインチューニング
6. **Recovery**: ηが低下し、成功率が回復するか確認

### Success Criteria / 成功基準

| Metric | Target | Rationale |
|--------|--------|-----------|
| Initial η | > 0.1 | 制約変化を検出 |
| Suspension count | > 5 | 保留が発動 |
| F/G updates | > 2 | 適応が実行される |
| Final η | < 0.1 | 適応が成功 |
| Success rate | > 50% | 性能が回復 |

---

## Escape Experiment Improvements / 脱出実験の改善案

### Current Issues / 現在の問題点

1. **点群のサンプリングが単純すぎる**
   - 現在: AABBから一様サンプリング
   - 問題: 実際の形状を正確に反映していない

2. **行動の効果が観測できない**
   - 現在: 行動を選択 → 即座に報酬
   - 問題: 物理的な結果を見ていない

3. **単一物体のみ**
   - 現在: 1つの物体のみ
   - 問題: 複雑な状況をテストできない

4. **エピソードが短すぎる**
   - 現在: max_steps = 10
   - 問題: 保留と適応に十分な時間がない

### Proposed Improvements / 改善案

#### 1. Better Point Cloud Sampling (より良い点群サンプリング)

**Current**:
```python
# AABBから一様サンプリング
points = np.random.uniform(aabb_min, aabb_max, (num_points, 3))
```

**Proposed**:
```python
# 表面サンプリング（より正確）
def sample_surface_points(object_id, num_points):
    """Sample points from object surface using ray casting"""
    # PyBulletのrayTestを使って表面点を取得
    # または、メッシュから直接サンプリング
```

**Benefits**:
- 形状の幾何学的特徴をより正確に捉える
- F/Gの学習が改善される

#### 2. Action Execution with Physics (物理シミュレーション付き行動実行)

**Current**:
```python
# 行動を選択 → 即座に報酬
reward = 1.0 if action == correct_action else -0.1
```

**Proposed**:
```python
# 行動を実行 → 物理シミュレーション → 結果を観測
def execute_action(action):
    if action == PUSH:
        apply_force(object_id, direction=[1, 0, 0], force=10.0)
    elif action == PULL:
        apply_force(object_id, direction=[-1, 0, 0], force=10.0)
    elif action == ROTATE:
        apply_torque(object_id, torque=[0, 0, 1], magnitude=5.0)
    
    # Simulate for N steps
    for _ in range(simulation_steps):
        p.stepSimulation()
    
    # Observe result
    new_pos, new_orn = p.getBasePositionAndOrientation(object_id)
    return new_pos, new_orn
```

**Benefits**:
- 行動の物理的効果を観測できる
- 制約の変化（重力、摩擦）が行動結果に反映される
- より現実的なタスク

#### 3. Multi-Object Scenarios (複数物体シナリオ)

**Proposed**:
```python
class MultiObjectEscapeRoom(EscapeRoomEnv):
    """
    複数の物体がある脱出部屋
    - 正しい物体を選択
    - 正しい行動を実行
    """
    def __init__(self, num_objects=3, ...):
        self.objects = []
        for i in range(num_objects):
            obj_type = random.choice([CUBE, CYLINDER, SPHERE])
            self.objects.append(self._create_object(obj_type))
        
        # 1つだけが「正解」
        self.target_object = random.choice(self.objects)
```

**Benefits**:
- より複雑な意思決定
- 「どの物体に作用すべきか」の判断が必要
- **Phase 3以降で実装**（Phase 2では単純に保つ）

#### 4. Longer Episodes (より長いエピソード)

**Current**: max_steps = 10

**Proposed**: 
- Phase 0-1: max_steps = 10（現状維持）
- Phase 2: max_steps = 50（保留と適応に十分な時間）

**Rationale**:
- 保留中にF/Gを更新するには複数ステップが必要
- 適応後の性能回復を観測するには時間が必要

---

## Implementation Priority / 実装優先度

### Phase 2.1: Constraints (最優先)

**Focus**: 重力・摩擦・質量の変化

**Why**: 
- 保留構造の核心をテスト
- 実装が比較的簡単
- 明確な成功基準

**Timeline**: 今すぐ実装可能

### Phase 2.2: Better Physics (次の優先度)

**Focus**: 行動の物理シミュレーション

**Why**:
- より現実的
- 制約の効果が観測可能
- Phase 2.1の結果を改善

**Timeline**: Phase 2.1の後

### Phase 2.3: Surface Sampling (オプション)

**Focus**: より正確な点群サンプリング

**Why**:
- F/Gの学習を改善
- しかし現状でも動作している

**Timeline**: 必要に応じて

### Phase 3+: Multi-Object (将来)

**Focus**: 複数物体、複雑なシナリオ

**Why**:
- より複雑なタスク
- しかし脱線のリスク

**Timeline**: Phase 2が成功した後

---

## Recommended Next Steps / 推奨される次のステップ

### 1. Phase 2.1を実装（今すぐ）

```python
# scripts/run_phase2.py
def run_phase_2(agent, fg_model, constraint_type='gravity', num_episodes=200):
    """
    Phase 2: Test on known shapes with constraints
    
    Args:
        constraint_type: 'gravity', 'friction', or 'mass'
    """
    # Load Phase 0 models
    # Apply constraint
    # Run experiments
    # Monitor η, suspension, F/G updates
    # Report results
```

### 2. 3つの制約をテスト

- Gravity: -1.6 (moon)
- Friction: 0.1 (ice)
- Mass: 10.0 (heavy)

### 3. 結果を分析

- ηは上昇するか？
- 保留は発動するか？
- F/G適応は機能するか？
- 成功率は回復するか？

### 4. 必要に応じてPhase 2.2へ

物理シミュレーション付き行動実行を実装

---

## Questions for Discussion / 議論すべき質問

### 1. Phase 2.1の制約の選択

**Question**: 重力、摩擦、質量のうち、どれから始めるべきか？

**Recommendation**: 
- **重力**から始める
- 理由: 最も直感的で、効果が明確

### 2. Phase 2.2の必要性

**Question**: 物理シミュレーション付き行動実行は必要か？

**Recommendation**:
- Phase 2.1の結果を見てから判断
- もし保留が発動しない場合、物理シミュレーションを追加

### 3. エピソード長

**Question**: max_steps = 50は長すぎるか？

**Recommendation**:
- まず20から始める
- 必要に応じて調整

### 4. 複数制約の同時適用

**Question**: 重力と摩擦を同時に変更すべきか？

**Recommendation**:
- **1つずつテスト**
- 理由: 何が効いているか明確にするため

---

## Summary / まとめ

### Phase 2の核心

**既知の形状 + 未知の物理 = 保留構造のテスト**

### 実装計画

1. ✅ Phase 0-1: 完了（既知・未知形状）
2. 🔜 Phase 2.1: 制約付き実験（重力・摩擦・質量）
3. 🔜 Phase 2.2: 物理シミュレーション（オプション）
4. 🔜 Phase 2.3: 表面サンプリング（オプション）
5. ⏸️ Phase 3+: 複数物体（将来）

### 脱線しないために

- **Phase 2.1に集中**
- 複数物体は後回し
- 表面サンプリングは必要に応じて
- 1つずつ検証

---

**Next Action**: Phase 2.1の実装を開始
