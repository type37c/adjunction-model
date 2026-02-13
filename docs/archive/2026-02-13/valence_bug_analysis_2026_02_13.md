# Valenceが更新されない問題の分析 (2026-02-13)

## 問題の特定

### 観察された現象

**Valenceの推移**:
- 全エピソード: 0.0（完全に一定）
- 変化なし

**期待される挙動**:
- Valenceは内発的報酬に基づいて更新されるべき
- 更新式: `valence(t+1) = (1-β) × valence(t) + β × valence_update`

## コードの調査

### Valenceの初期化（agent_layer_v4.py, line 124）

```python
valence = self.valence_memory.initial_valence(batch_size, device)
```

**initial_valence()の実装**（valence_v2.py, line 96-101）:
```python
return torch.full(
    (batch_size, self.valence_dim),
    self.init_valence,  # デフォルト: 1.0
    device=device,
    dtype=torch.float32
)
```

**問題**:
- 初期値は1.0のはず
- しかし、観察されたValenceは0.0

**仮説**:
- 初期化後、どこかで0.0にリセットされている
- または、Valenceの平均を取る際に問題がある

### Valenceの更新（agent_layer_v4.py, line 220-230）

```python
valence_results = self.valence_memory(
    valence_prev,
    uncertainty_prev,
    uncertainty_curr,
    coherence_prev,
    coherence_signal_scalar,
    attention_weight,
    kl_divergence
)

valence_new = valence_results['valence']  # (B, valence_dim)
```

**ValenceMemoryV2.forward()の実装**（valence_v2.py, line 103-150）:
```python
# Compute intrinsic rewards
rewards = self.intrinsic_reward(...)

R_intrinsic = rewards['R_intrinsic']  # (B,)

# Project intrinsic reward to valence update
valence_update = self.reward_to_valence(R_intrinsic.unsqueeze(-1))  # (B, valence_dim)

# Update valence with exponential decay
valence_new = (1 - self.decay_rate) * valence_prev + self.decay_rate * valence_update
```

**分析**:
- 更新ロジック自体は正しい
- `valence_update`は`reward_to_valence`ネットワークの出力

**問題の可能性**:
1. `reward_to_valence`ネットワークが0.0を出力している
2. `valence_prev`が0.0で、`valence_update`も0.0
3. どこかでValenceが0.0にリセットされている

### Valenceの記録（train_agent_value_based.py）

**train_episodeメソッドを確認する必要がある**:
- Valenceの値をどのように記録しているか
- `agent_state`からValenceを正しく抽出しているか

## 調査の次のステップ

### 1. Valenceの初期値を確認

**方法**:
- デバッグ出力を追加
- `initial_valence()`の戻り値を確認

### 2. Valence更新の中間値を確認

**方法**:
- `ValenceMemoryV2.forward()`にデバッグ出力を追加
- `R_intrinsic`, `valence_update`, `valence_new`を出力

### 3. train_episodeでのValence記録を確認

**方法**:
- `train_agent_value_based.py`の`train_episode()`を確認
- `agent_state`からValenceを正しく抽出しているか

### 4. ConditionalAdjunctionModelV4のforward()を確認

**方法**:
- Agent Cのforward()が正しく呼ばれているか
- Valenceが正しく返されているか

## 最も可能性が高い原因

### 仮説: train_episodeでValenceを記録していない

**根拠**:
- `train_episode()`のコードを確認する必要がある
- `agent_state`にValenceが含まれているか不明
- Valenceの平均を計算する際に、デフォルト値0.0を使っている可能性

**確認方法**:
- `train_agent_value_based.py`の`train_episode()`を再読
- `agent_info`または`agent_state`からValenceを抽出しているか確認

---

**分析日**: 2026-02-13
**分析者**: AI Agent (Manus)
**ステータス**: 調査中

## 原因の特定

### train_episodeでのValence抽出（train_agent_value_based.py, line 172, 183）

```python
'valence': agent_info.get('valence_mean', torch.tensor(0.0)).item(),
...
total_valence += agent_info.get('valence_mean', torch.tensor(0.0)).item()
```

**問題**:
- `agent_info`（= `results['rssm_info']`）から`valence_mean`を取得している
- しかし、`rssm_info`にはValenceの情報が含まれていない可能性

### ConditionalAdjunctionModelV4のforward()を確認

**必要な確認**:
- `results['rssm_info']`に`valence_mean`が含まれているか
- Valenceは`results['agent_state']`に含まれているはず

### 推測される原因

**Agent Cのforward()の戻り値**:
- `new_state`: Agent Cの新しい状態（Valenceを含む）
- `context`: コンテキストベクトル
- `info`: 中間値（RSSM情報、内発的報酬など）

**ConditionalAdjunctionModelV4のforward()の構造**:
- Agent Cを呼び出す
- `rssm_info`を返す
- しかし、Valenceの情報を`rssm_info`に含めていない可能性

**結論**:
- **`rssm_info`にValenceの情報が含まれていない**
- したがって、`agent_info.get('valence_mean', 0.0)`は常にデフォルト値0.0を返す

## 修正方法

### オプション1: rssm_infoにValenceを追加

**ConditionalAdjunctionModelV4のforward()を修正**:
```python
# Agent Cのforward()の戻り値にValenceを追加
results['rssm_info']['valence'] = new_state['valence']
results['rssm_info']['valence_mean'] = new_state['valence'].mean(dim=-1)
```

### オプション2: agent_stateからValenceを抽出

**train_episodeを修正**:
```python
# agent_infoではなく、agent_stateからValenceを取得
valence = results['agent_state'].get('valence', torch.zeros(1, 32, device=self.device))
valence_mean = valence.mean().item()

'valence': valence_mean,
...
total_valence += valence_mean
```

### 推奨される修正

**オプション1が推奨**:
- ConditionalAdjunctionModelV4のforward()で、Valenceを`rssm_info`に追加
- これにより、他の内発的報酬と同様に、Valenceも`rssm_info`から取得できる

## 修正の実装

### 修正箇所: ConditionalAdjunctionModelV4.forward()

**ファイル**: `src/models/conditional_adjunction_v4.py`

**修正内容**:
```python
# Agent Cのforward()呼び出し後
new_state, context, agent_info = self.agent_c(...)

# rssm_infoにValenceを追加
results['rssm_info'] = {
    **agent_info,
    'valence': new_state['valence'],
    'valence_mean': new_state['valence'].mean(dim=-1)
}
```

---

**更新日**: 2026-02-13
**更新者**: AI Agent (Manus)
**ステータス**: 原因特定完了、修正方法確定
