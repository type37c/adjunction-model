# Valence学習の設計再考察

**Date**: 2026-02-16  
**Context**: Phase 2.5 Valence Role Experiment実装レビュー

## 問題の提起

ValenceMemoryV3では、`delta_eta_to_valence`という学習可能なLinear層を使って、Δη（Slackの変化）をvalence空間に射影している。

```python
self.delta_eta_to_valence = nn.Linear(1, valence_dim)
```

この重みは`Phase2SlackTrainer`の`optimizer.step()`でモデル全体のパラメータとして一緒に更新される。つまり、**valenceの「意味」はAffordance Loss (L_aff)の勾配によって決まる**ことになる。

## 核心的な問い

> valenceの使い方は「創発」しているように見えて、実際にはL_affが「こう使え」と教えているのではないか？

これは条件3（Designed）との違いが思ったより小さいかもしれないことを示唆する。

## 分析

### 条件2（Emergent）の実際の学習ダイナミクス

1. **Forward Pass**:
   - Δη → `delta_eta_to_valence` → valence
   - valence → RSSM → context
   - context → F, G → affordances, reconstructed_pos

2. **Backward Pass**:
   - L_aff = ||affordances - target_affordances||²
   - ∂L_aff/∂affordances → ∂L_aff/∂context → ∂L_aff/∂valence → ∂L_aff/∂delta_eta_to_valence

3. **結果**:
   - `delta_eta_to_valence`の重みは、**L_affを最小化するように**更新される
   - つまり、valenceは「L_affを下げるのに役立つΔηの特徴」を学習する

### 条件3（Designed）の学習ダイナミクス

1. **Forward Pass**:
   - intrinsic_reward = α×R_curiosity + β×R_competence + γ×R_novelty
   - valence += learning_rate × intrinsic_reward × attention_weight
   - valence → Priority = coherence × uncertainty × valence
   - Priority → attention → context
   - context → F, G → affordances, reconstructed_pos

2. **Backward Pass**:
   - L_aff = ||affordances - target_affordances||²
   - ∂L_aff/∂affordances → ∂L_aff/∂context → ∂L_aff/∂Priority → ∂L_aff/∂valence

3. **結果**:
   - valence自体は勾配で更新されないが、valenceを使う**Priority計算の下流**（context生成など）がL_affの勾配を受ける
   - intrinsic_rewardの重み（α, β, γ）は固定されている

### 本質的な違い

| 側面 | 条件2（Emergent） | 条件3（Designed） |
|:---|:---|:---|
| **valenceの更新** | Δηから学習可能な射影で生成 | intrinsic_rewardから固定ルールで更新 |
| **valenceの意味** | L_affの勾配が決定 | 人間が設計（curiosity, competence, novelty） |
| **使い方の学習** | RSSMが学習 | Priority計算が固定 |

## 問題点と代替案

### 問題点

条件2は「valenceの使い方を創発に委ねる」という原則を実装しているが、**valenceの意味自体はL_affが決定している**。これは以下の問題を引き起こす可能性がある：

1. **条件3との差が小さい**: どちらもL_affの勾配がvalenceの役割を決める
2. **創発の余地が限定的**: valenceの意味がL_affに縛られるため、L_affを超えた目的の創発が難しい

### 代替案1: Valenceを勾配から切り離す

```python
# Δηをvalence空間に射影するが、勾配を止める
delta_eta_projected = self.delta_eta_to_valence(delta_eta.detach())
```

**効果**:
- valenceの更新がL_affの勾配から独立する
- valenceは純粋に「Δηの記憶」として機能する
- RSSMがvalenceの使い方を学習する際、L_affの勾配のみが教師信号になる

**問題**:
- `delta_eta_to_valence`の重みが更新されない → 初期化に依存

### 代替案2: Valenceを固定ルールで更新（条件3に近づく）

```python
# 学習可能な射影を削除し、Δηを直接使う
valence_new = valence_prev * (1 - decay) + learning_rate * delta_eta
```

**効果**:
- valenceの意味が明確（Δηの指数移動平均）
- 学習可能なパラメータがないため、L_affの勾配の影響を受けない

**問題**:
- valence_dimが1に固定される（多次元表現ができない）

### 代替案3: Valenceを別の損失で学習

```python
# Valenceの予測誤差を別の損失として追加
L_valence = ||predicted_delta_eta - actual_delta_eta||²
```

**効果**:
- valenceが「Δηの予測」という明確な意味を持つ
- L_affとは独立した学習目標

**問題**:
- 新しい損失項の追加 → 実験設計が複雑化

## 推奨事項

### 短期的（Phase 2.5実験）

**現状の実装を維持する**理由：

1. **実験の目的**: 条件2 vs 条件3の比較であり、「valenceがL_affの勾配を受ける」こと自体は両条件に共通
2. **差異の焦点**: 条件2は「valenceの使い方」を学習、条件3は「valenceの使い方」を固定
3. **解釈の明確性**: 実験結果から「使い方の学習」の効果を測定できる

### 中期的（実験結果を踏まえて）

実験結果に基づき、以下を検討：

- **条件2が条件3より優れている場合**: 現在の設計が有効 → 次のフェーズへ
- **条件2と条件3が同等の場合**: 代替案1（勾配の切り離し）を試す
- **条件2が条件3より劣る場合**: Priority計算の設計を再考

### 長期的（Purpose-Emergent実験）

valenceの意味をL_affから解放するために：

1. **複数の目的関数**: L_affだけでなく、他の目的（例: 探索、多様性）を導入
2. **メタ学習**: valenceの更新ルール自体を学習可能にする
3. **階層的目的**: 短期目的（L_aff）と長期目的（purpose）を分離

## 結論

ユーザーの指摘は正しい。条件2のvalenceは「創発」しているように見えて、実際にはL_affが意味を決定している。

しかし、これは**Phase 2.5実験の無効化を意味しない**。条件2と条件3の違いは「valenceの使い方の学習」にあり、これは実験で測定可能である。

実験結果が「使い方の学習」の効果を示さない場合、valenceの設計を根本的に見直す必要がある。その際、本ドキュメントの代替案が出発点となる。

---

**著者**: Manus AI（ユーザーの指摘に基づく）  
**レビュー**: Pending  
**関連文書**: `priority_and_valence_reconsidered.md`, `NEW_PLAN.md`
