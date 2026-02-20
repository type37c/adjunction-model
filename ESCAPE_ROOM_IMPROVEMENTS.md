# Escape Room Environment: Improvements and Redesign
# 脱出部屋環境：改善と再設計

**Date**: February 20, 2026  
**Status**: Addressing simplicity issues

---

## Current Problems / 現在の問題点

### 1. Actions are Too Simple / 行動が単純すぎる

**Current**:
```python
action = 0  # Push
action = 1  # Pull  
action = 2  # Rotate
```

**Problems**:
- 身体部位が明示されていない
- 離散的すぎる（3択）
- 創造的な組み合わせができない
- 怪我の制約を表現できない

### 2. Tasks are Too Simple / タスクが単純すぎる

**Current**:
- 1つの物体に1つの正しい行動を選ぶ
- 成功 or 失敗の2値

**Problems**:
- 創造性が測れない
- 保留と適応のプロセスが観測しにくい
- 「たまたま正解」と「理解して正解」の区別がつかない

### 3. No Multi-Step Reasoning / 複数ステップの推論がない

**Current**:
- 1ステップで終了

**Problems**:
- 複雑な問題解決能力が測れない
- 保留構造の持続的な動作が観測できない

---

## Proposed Improvements / 改善案

### Improvement 1: Explicit Body Parts (身体部位の明示化)

#### New Action Space

```python
@dataclass
class Action:
    limb: str           # 'right_hand', 'left_hand', 'right_foot', 'left_foot', 'torso', 'head'
    primitive: str      # 'push', 'pull', 'rotate', 'lift', 'press', 'twist'
    target_point: int   # Index of point cloud point to interact with
    force: float        # Magnitude of force (0.0 to 1.0)
```

#### Benefits

1. **身体部位が明示される**
   - 「右手で押す」vs「左手で押す」が区別できる
   - 怪我の制約が自然に表現できる

2. **より豊かな行動空間**
   - 6 limbs × 6 primitives × 100 points = 3600通り
   - 創造的な組み合わせが可能

3. **連続的な制御**
   - 力の大きさを調整できる
   - より細かい操作が可能

#### Implementation

```python
class EscapeRoomEnv:
    def __init__(self, ..., disabled_limbs=None):
        self.disabled_limbs = disabled_limbs or []
        self.limbs = ['right_hand', 'left_hand', 'right_foot', 'left_foot', 'torso', 'head']
        self.primitives = ['push', 'pull', 'rotate', 'lift', 'press', 'twist']
    
    def step(self, action: Action):
        # Check if limb is disabled
        if action.limb in self.disabled_limbs:
            return self._handle_disabled_limb(action)
        
        # Execute action
        success = self._execute_action(action)
        
        # Compute reward
        reward = self._compute_reward(success)
        
        return obs, reward, done, info
    
    def _handle_disabled_limb(self, action):
        """Handle action using disabled limb"""
        # Action fails or has reduced effect
        obs = self._get_observation()
        reward = -1.0
        done = False
        info = {
            'failure_reason': 'disabled_limb',
            'attempted_limb': action.limb,
            'eta': self._compute_eta()  # Should spike
        }
        return obs, reward, done, info
```

### Improvement 2: Multi-Step Tasks (複数ステップのタスク)

#### Scenario: Escape Room with Multiple Objects

**Setup**:
- 部屋には3つの物体がある
  1. **Lever** (レバー): 引くとドアのロックが外れる
  2. **Button** (ボタン): 押すとドアが開く
  3. **Door** (ドア): 通過すると脱出成功

**Sequence**:
1. Lever を引く → ロック解除
2. Button を押す → ドア開く
3. Door を通過 → 脱出成功

**Constraints**:
- 順番を間違えると失敗（Button → Leverは無効）
- 各ステップで適切な身体部位と行動を選ぶ必要がある

#### Benefits

1. **複雑な問題解決**
   - 単なる1ステップの選択ではない
   - 計画と実行の両方が必要

2. **保留と適応の観測**
   - 各ステップでηを測定
   - 保留が複数回発動する可能性

3. **「理解」の測定**
   - 正しい順番を理解しているか
   - 未知の物体でも適切な順番を推論できるか

#### Implementation

```python
class MultiStepEscapeRoom(EscapeRoomEnv):
    def __init__(self, ..., sequence=None):
        super().__init__(...)
        self.sequence = sequence or ['lever', 'button', 'door']
        self.current_step = 0
        self.completed_steps = []
    
    def step(self, action: Action):
        # Check if action is correct for current step
        target_object = self.sequence[self.current_step]
        
        if self._is_correct_action(action, target_object):
            self.completed_steps.append(target_object)
            self.current_step += 1
            reward = 1.0
            
            if self.current_step >= len(self.sequence):
                done = True
                reward = 10.0  # Escape success
        else:
            reward = -0.1
            done = False
        
        obs = self._get_observation()
        info = {
            'current_step': self.current_step,
            'completed_steps': self.completed_steps,
            'eta': self._compute_eta()
        }
        
        return obs, reward, done, info
```

### Improvement 3: "Accidental Success" Prevention (偶然の成功を防ぐ)

#### Problem

> 機械学習でやっちゃうとたまたま正解になったら出れちゃうけどどうするの？

#### Solution 1: Multiple Trials (複数回の試行)

**Concept**: 同じ物体に対して複数回成功する必要がある

```python
class EscapeRoomEnv:
    def __init__(self, ..., required_successes=3):
        self.required_successes = required_successes
        self.success_count = defaultdict(int)
    
    def step(self, action):
        if self._is_correct_action(action):
            object_id = self._get_target_object(action)
            self.success_count[object_id] += 1
            
            if self.success_count[object_id] >= self.required_successes:
                # Object is "solved"
                self.solved_objects.add(object_id)
        
        # Escape only if all objects are solved
        done = len(self.solved_objects) >= len(self.objects)
        ...
```

**Benefits**:
- 偶然の成功を排除
- 安定した理解を要求

**Concerns**:
- 訓練時間が長くなる

#### Solution 2: Multiple Unknown Objects (複数の未知物体)

**Concept**: 部屋の中に複数の未知物体があり、すべてに対して適切な操作が必要

```python
class EscapeRoomEnv:
    def __init__(self, ..., num_objects=3):
        self.objects = self._generate_random_objects(num_objects)
        self.solved_objects = set()
    
    def step(self, action):
        # Each object requires appropriate action
        for obj in self.objects:
            if self._is_correct_action_for_object(action, obj):
                self.solved_objects.add(obj.id)
        
        # Escape only if all objects are solved
        done = len(self.solved_objects) >= len(self.objects)
        ...
```

**Benefits**:
- 1つだけ偶然成功しても脱出できない
- 複数の異なる未知形状すべてに対して適切な動作を見つける必要
- これは偶然ではなく理解に基づいている

**Recommendation**: Solution 2（複数の未知物体）

### Improvement 4: Noisy Environment (騒音制約) - Optional

#### Scenario (from initial note)

> 静かにしなければならない環境でスーツケースを運ぶ

#### Implementation

```python
class EscapeRoomEnv:
    def __init__(self, ..., noise_constraint=False, noise_threshold=0.5):
        self.noise_constraint = noise_constraint
        self.noise_threshold = noise_threshold
    
    def step(self, action):
        # Compute noise level
        noise = self._compute_noise(action)
        
        if self.noise_constraint and noise > self.noise_threshold:
            # Penalty for being too noisy
            reward -= noise_penalty
            info['noise_violation'] = True
        
        ...
```

#### Noise Computation

```python
def _compute_noise(self, action):
    """Compute noise level based on action"""
    noise = 0.0
    
    # Different primitives have different noise levels
    noise_levels = {
        'push': 0.3,
        'pull': 0.3,
        'drag': 0.8,  # Very noisy!
        'lift': 0.1,  # Quiet
        'carry': 0.1,
    }
    
    noise += noise_levels.get(action.primitive, 0.5)
    
    # Force affects noise
    noise += action.force * 0.5
    
    return noise
```

#### Benefits

- 文脈フィルターの実装
- 身体状態フィルターとの干渉をテスト
- より現実的なシナリオ

**Decision**: Phase 2.1-2.2の後で実装（脱線しないため）

---

## Revised Environment Architecture / 修正版環境アーキテクチャ

### Class Hierarchy

```
EscapeRoomEnv (base)
├── SimpleEscapeRoom (current, for Phase 0-1)
│   └── 1 object, discrete actions
├── BodyPartEscapeRoom (Phase 2.1-2.2)
│   └── Explicit body parts, disabled limbs
└── MultiStepEscapeRoom (Phase 2.3+)
    └── Multiple objects, sequential tasks
```

### Implementation Plan

#### Phase 0-1: Keep Current (SimpleEscapeRoom)

- 既存の実装を維持
- 基礎的な学習と汎化のテスト

#### Phase 2.1: Add BodyPartEscapeRoom

**New Features**:
- Explicit body parts (limb field in Action)
- Disabled limbs (right_hand, left_foot)
- η spike when using disabled limb

**Timeline**: 2-3 days

#### Phase 2.2: Extend to MultiStepEscapeRoom

**New Features**:
- Multiple objects (3+)
- Sequential tasks (lever → button → door)
- Prevention of accidental success

**Timeline**: 2-3 days

#### Phase 2.3+: Add Noisy Environment (Optional)

**New Features**:
- Noise constraint
- Context filter

**Timeline**: 1-2 days (if needed)

---

## Revised Experimental Protocol / 修正版実験プロトコル

### Phase 0: Known Shapes, Full Body (既知形状、健常な身体)

**Setup**:
- Objects: cube, cylinder, sphere (既知)
- Body: すべての身体部位が使える
- Task: 1つの物体に適切な行動

**Goal**: ベースライン性能の確立

**Expected**: 58% success rate (already achieved)

### Phase 1: Unknown Shapes, Full Body (未知形状、健常な身体)

**Setup**:
- Objects: lever, button, knob (未知)
- Body: すべての身体部位が使える
- Task: 1つの物体に適切な行動

**Goal**: 汎化能力のテスト

**Expected**: 62% success rate (already achieved)

### Phase 2.1: Known Shapes, Right Hand Disabled (既知形状、右手負傷)

**Setup**:
- Objects: cube, cylinder, sphere (既知)
- Body: **右手が使用不能**
- Task: 1つの物体に適切な行動（左手、両足、体全体を使う）

**Goal**: 身体的制約への適応

**Expected**:
- Initial η > 0.2（右手が使えない！）
- Suspension > 10回
- F/G updates > 5回
- Final success rate: 45%（低下するが0ではない）

### Phase 2.2: Unknown Shapes, Right Hand Disabled (未知形状、右手負傷)

**Setup**:
- Objects: lever, button, knob (未知)
- Body: **右手が使用不能**
- Task: 1つの物体に適切な行動

**Goal**: 汎化 + 身体的制約への適応

**Expected**:
- さらに困難
- Success rate: 35-40%

### Phase 2.3: Multi-Step, Right Hand Disabled (複数ステップ、右手負傷) - Optional

**Setup**:
- Objects: lever → button → door (sequential)
- Body: **右手が使用不能**
- Task: 正しい順番で操作

**Goal**: 複雑な問題解決 + 身体的制約

**Expected**:
- Success rate: 20-30%（非常に困難だが不可能ではない）

---

## Comparison with Baselines / ベースラインとの比較

### Expected Performance Table

| Method | Phase 0 | Phase 1 | Phase 2.1 (right hand) | Phase 2.2 (unknown + right hand) |
|--------|---------|---------|------------------------|----------------------------------|
| **Ours** | 58% | 62% | **45%** ✅ | **38%** ✅ |
| Static F/G | 58% | 42% | **10%** ❌ | **5%** ❌ |
| PPO | 54% | 35% | **5%** ❌ | **2%** ❌ |

### Why Ours is Better

1. **Phase 0-1**: 汎化能力（未知形状）
2. **Phase 2.1**: 身体的制約への適応（保留構造）
3. **Phase 2.2**: 汎化 + 適応の両方

### Why Baselines Fail

**Static F/G**:
- F/Gが凍結されている
- 右手が使えないことに適応できない
- ηが上昇したまま

**PPO**:
- 身体部位の概念がない
- 右手が使えない状況を学習していない
- ほぼランダムな行動

---

## Implementation Checklist / 実装チェックリスト

### Week 1: Core Improvements

- [ ] Day 1-2: Implement BodyPartEscapeRoom
  - [ ] Action dataclass with limb field
  - [ ] Disabled limbs mechanism
  - [ ] η spike detection
  
- [ ] Day 3: Implement Phase 2.1 (right hand disabled)
  - [ ] Train Phase 0 model
  - [ ] Test with right_hand disabled
  - [ ] Record η, suspension, F/G updates
  
- [ ] Day 4: Implement Phase 2.2 (unknown + right hand)
  - [ ] Test with unknown shapes + right_hand disabled
  - [ ] Compare with Phase 2.1
  
- [ ] Day 5: Baselines
  - [ ] Static F/G baseline
  - [ ] PPO baseline (if time permits)
  
- [ ] Day 6-7: Multiple seeds and analysis
  - [ ] 5 seeds × all phases
  - [ ] Statistical significance tests
  - [ ] Visualization

### Week 2: Optional Extensions

- [ ] Day 1-2: MultiStepEscapeRoom (if needed for paper)
- [ ] Day 3: Noisy environment (if time permits)
- [ ] Day 4-7: Paper writing

---

## Success Criteria / 成功基準

### For Phase 2.1 (Right Hand Disabled)

✅ **Must Have**:
1. η spikes when using disabled limb (> 0.2)
2. Suspension triggers (> 10 times)
3. F/G adaptation occurs (> 5 times)
4. Success rate recovers (> 40%)
5. Ours > Static F/G (p < 0.05)

⭐ **Nice to Have**:
1. Observation of creative solutions (using left hand, feet, torso)
2. η decreases after adaptation (< 0.1)
3. Ours > PPO (p < 0.05)

### For Paper Acceptance

✅ **Must Have**:
1. Phase 0-1 results (already achieved)
2. Phase 2.1 results (right hand disabled)
3. Baseline comparisons (Static F/G minimum)
4. Statistical significance (5 seeds)

⭐ **Nice to Have**:
1. Phase 2.2 results (unknown + right hand)
2. Multiple baselines (PPO, Active Inference)
3. Multi-step tasks

---

## Summary / まとめ

### Key Improvements

1. **Explicit Body Parts**: 身体部位の明示化
2. **Multi-Step Tasks**: 複数ステップのタスク
3. **Accidental Success Prevention**: 偶然の成功を防ぐ
4. **Bodily Constraints**: 身体的制約（右手の怪我）

### Priority

1. 🔴 **Phase 2.1**: Right hand disabled（最優先）
2. 🟠 **Baselines**: Static F/G, PPO
3. 🟡 **Phase 2.2**: Unknown + right hand（論文に有用）
4. ⚪ **Multi-step**: Optional（時間があれば）

### Timeline

- **Week 1**: Phase 2.1-2.2 + baselines + analysis
- **Week 2**: Paper writing

---

**Ready to implement these improvements?** 🚀
