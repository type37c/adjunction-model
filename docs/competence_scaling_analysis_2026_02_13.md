# Competence報酬のスケーリング効果の分析 (2026-02-13)

## 修正内容

**変更**:
```python
# Before
R_competence = delta_coherence * attention_weight

# After
R_competence = delta_coherence * attention_weight * 100.0
```

**目的**:
- Competence報酬のスケールを増やして、内発的報酬への寄与を高める

## 検証実験の結果（20エピソード）

### Competence報酬の値

**観察**:
- 範囲: -0.0095 ~ -0.0075
- 平均: -0.0086

**比較（v2: スケーリングなし）**:
- 範囲: -9.5e-5 ~ -7.5e-5
- 平均: -8.6e-5

**改善**:
- スケールが約**100倍**増加（期待通り）

### 内発的報酬への寄与

**計算**（エピソード20）:
```
R_intrinsic = α × R_curiosity + β × R_competence + γ × R_novelty
0.0150 = 0.0 × (-0.099) + 0.6 × (-0.0092) + 0.4 × 0.051
0.0150 = 0 + (-0.0055) + 0.0204
0.0150 ≈ 0.0149  ✓
```

**寄与率**:
- Novelty: 0.0204 / 0.0150 = **136%**
- Competence: -0.0055 / 0.0150 = **-37%**

**問題**:
- Competence報酬が**負**
- 内発的報酬を**減少**させている

## 問題の診断

### Competence報酬が負である理由

**定義**（intrinsic_reward.py, line 97-98）:
```python
delta_coherence = coherence_prev - coherence_curr  # (B, 1)
R_competence = delta_coherence * attention_weight * self.competence_scale
```

**期待される挙動**:
- Coherenceが減少（破綻が解消）→ delta_coherence > 0 → R_competence > 0
- Coherenceが増加（破綻が悪化）→ delta_coherence < 0 → R_competence < 0

**観察された挙動**:
- R_competence < 0（常に負）
- これは、Coherenceが常に増加していることを意味

### Coherenceの推移

**観察**（metrics.json）:
- エピソード0: 0.426
- エピソード20: 0.442
- トレンド: **増加**（+0.016）

**解釈**:
- Coherence（破綻）が増加している
- Agent Cは破綻を**悪化**させている

**理論的矛盾**:
- Agent Cは内発的報酬を最大化するはず
- しかし、Competence報酬は負（破綻の悪化）
- なぜAgent Cは破綻を悪化させる行動を取るのか？

## 原因の分析

### 仮説1: Novelty報酬の支配

**観察**:
- Novelty報酬: +0.051（正、大きい）
- Competence報酬: -0.0092（負、小さい）
- 合計: +0.015（正）

**解釈**:
- Agent Cは、Novelty報酬を最大化している
- Competence報酬の負の寄与は、Novelty報酬の正の寄与で相殺される
- したがって、Agent Cは破綻を悪化させてでも、Noveltyを追求する

**結論**:
- Novelty報酬が支配的すぎる
- Competence報酬の重みを増やしても、Novelty報酬の影響を打ち消せない

### 仮説2: Coherenceの定義の問題

**現在の定義**（conditional_adjunction_v4.py）:
- Coherence = Chamfer distance（再構成誤差）
- 値が大きい = 破綻が大きい

**問題**:
- Coherenceが増加 = 破綻が悪化
- しかし、これは本当に「悪化」なのか？

**代替解釈**:
- Coherenceが増加 = F/Gの固定された重みでは対応できない新しい形状
- これは、Agent Cが「探索」していることを示す

**結論**:
- Coherenceの増加は、必ずしも「悪化」ではない
- 「探索」の結果かもしれない

### 仮説3: Attentionの問題

**観察**:
- Attention weightが小さい可能性
- Competence報酬 = delta_coherence × attention × 100
- Attentionが小さいと、Competence報酬も小さい

**確認が必要**:
- Attention weightの値を確認
- Priority-based Attentionが正しく機能しているか

## 理論的再検討

### Competence報酬の本来の意図

**AGENT_GUIDELINES.mdの洞察**:
> 「破綻に向き合い、それを理解することで有能感を得る」

**現在の実装**:
- R_competence = (coherence_prev - coherence_curr) × attention
- これは、「破綻の減少」を報酬とする

**問題**:
- F/Gの重みが固定されている
- Agent Cは破綻を減少させる手段を持たない
- したがって、Competence報酬は常に負または0

**代替案**:
1. **破綻への注目を報酬とする**
   ```python
   R_competence = coherence_curr × attention
   ```
   - 破綻が大きく、かつ注目している → 報酬

2. **破綻の理解を報酬とする**
   ```python
   R_competence = -uncertainty_change × coherence_curr × attention
   ```
   - 破綻があり、不確実性が減少（理解） → 報酬

3. **破綻の予測精度を報酬とする**
   ```python
   R_competence = -prediction_error × attention
   ```
   - 破綻を正確に予測できる → 報酬

## 推奨される対処

### 短期（Competence報酬の再定義）

**オプションA: 破綻への注目を報酬とする**
```python
R_competence = coherence_curr × attention_weight × competence_scale
```

**理由**:
- F/Gが固定されている現在の設定では、破綻を減少させることは不可能
- 代わりに、「破綻に注目する」ことを報酬とする
- これは、「破綻から逃げない」という設計意図に合致

**オプションB: Competence報酬を削除**
```python
α = 0.0, β = 0.0, γ = 1.0
```

**理由**:
- 現在の実装では、Competence報酬が機能していない
- Novelty報酬のみに集中する

### 中期（F/Gの学習を再開）

**提案**:
- F/Gの重みを固定せず、学習させる
- これにより、Agent Cは破綻を減少させる手段を持つ
- Competence報酬が正しく機能する可能性

### 長期（理論的再設計）

**検討事項**:
- Competence報酬の理論的定義を再検討
- 「suspension structure」の哲学と整合する形で再設計
- 「破綻への向き合い」を定量化する方法を考案

## 次のステップ

### 即時実験

1. **オプションA（破綻への注目を報酬）を実装**
   ```python
   R_competence = coherence_curr × attention_weight × competence_scale
   ```

2. **短時間実験（20エピソード）**
   - Competence報酬が正になるか確認
   - 内発的報酬への寄与を確認

3. **結果に基づいて判断**
   - 成功 → 本格的実験（100エピソード）
   - 失敗 → オプションB（Competence報酬を削除）

---

**分析日**: 2026-02-13
**分析者**: AI Agent (Manus)
**ステータス**: Competence報酬の再定義が必要
