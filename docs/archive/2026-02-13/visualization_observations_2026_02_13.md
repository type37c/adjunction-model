# 可視化結果の観察 (2026-02-13)

## 内発的報酬の可視化分析

### Total Intrinsic Reward（左上）
**観察**:
- 明確な上昇トレンド: -0.021 → +0.025
- 滑らかな改善曲線（ノイズは少ない）
- エピソード80以降で加速的に改善

**解釈**:
- Agent Cは確実に学習している
- 価値関数の導きが効果的に機能
- 後半の加速は、Agent Cが「良い戦略」を発見した可能性

### Curiosity Reward（右上）
**観察**:
- 値の範囲: -0.094 ~ -0.102
- **常に負の値**
- 高頻度の振動（ノイズが多い）
- 明確なトレンドなし

**解釈**:
- 好奇心報酬 = uncertainty_prev - uncertainty_curr
- 常に負 → uncertaintyが常に増加している？
- これは予想外：通常、学習が進むとuncertaintyは減少するはず

**問題の可能性**:
- Curiosity報酬の計算式に問題がある
- または、Agent Cが意図的にuncertaintyを増やしている（探索的行動）

### Competence Reward（左下）
**観察**:
- 値の範囲: -9.5e-5 ~ +7.5e-5
- **極めて小さい値**（10^-5オーダー）
- 高頻度の振動
- 明確なトレンドなし

**解釈**:
- 有能感報酬 = (coherence_prev - coherence_curr) × attention
- 値が極小 → coherenceの変化が極めて小さい
- これは、coherenceが安定していることと整合（avg_coherence: 0.4328 → 0.4320）

**問題の可能性**:
- Competence報酬のスケールが小さすぎる
- β=0.5の重みでも、総合的な内発的報酬への寄与は無視できるレベル

### Novelty Reward（右下）
**観察**:
- 明確な上昇トレンド: 0.04 → 0.27
- 滑らかな改善曲線
- エピソード80以降で加速的に改善

**解釈**:
- 新奇性報酬 = KL(posterior || prior)
- 上昇トレンド → Agent Cの信念分布が事前分布から乖離している
- これは、Agent Cが「新しいパターン」を発見し続けていることを示す

**重要な発見**:
- **Total Intrinsic RewardとNovelty Rewardの曲線が酷似**
- これは、内発的報酬の改善が主にNovelty報酬によるものであることを示唆
- Curiosity（負）とCompetence（極小）は、ほとんど寄与していない

## 内発的報酬の成分分析

### 各成分の寄与度（推定）

エピソード100の値から逆算：
- R_intrinsic ≈ 0.025
- R_curiosity ≈ -0.098
- R_competence ≈ -8e-5 ≈ 0
- R_novelty ≈ 0.27

重み付き合計：
```
R_intrinsic = α × R_curiosity + β × R_competence + γ × R_novelty
0.025 ≈ 0.3 × (-0.098) + 0.5 × 0 + 0.2 × 0.27
0.025 ≈ -0.0294 + 0 + 0.054
0.025 ≈ 0.0246  ✓ (ほぼ一致)
```

**結論**:
- Novelty報酬が内発的報酬の主要な駆動力
- Curiosity報酬は負の寄与（内発的報酬を下げている）
- Competence報酬は無視できるレベル

## 問題点と改善案

### 問題1: Curiosity報酬が常に負

**原因仮説**:
1. Uncertaintyの計算が逆になっている（増加を減少と誤認）
2. Agent Cが意図的にuncertaintyを増やしている（探索行動）
3. 計算式の符号が逆

**確認が必要**:
- `src/models/intrinsic_reward.py`のCuriosity計算式を確認
- Uncertaintyの時系列を確認（agent_state.pngで確認可能）

### 問題2: Competence報酬が極小

**原因仮説**:
1. Coherenceの変化が極めて小さい（これは観察と整合）
2. Attentionの値が小さい
3. スケーリングが不適切

**改善案**:
- Competence報酬のスケーリングを調整（例: 100倍）
- または、β（competence weight）を大幅に増やす

### 問題3: Novelty報酬への過度な依存

**観察**:
- 内発的報酬の改善はほぼNovelty報酬のみによる
- これは、Agent Cが「新奇性の追求」のみを学習していることを意味

**理論的懸念**:
- 保留構造の創発には、Competence（破綻への向き合い）が重要なはず
- しかし、現在の実装ではCompetenceがほぼ機能していない

**改善案**:
- Competence報酬のスケーリングを大幅に調整
- または、β（competence weight）を増やす（例: β=0.8, γ=0.1）

## 次の実験提案

### 実験A: ハイパーパラメータの調整

**設定**:
- α (curiosity) = 0.1
- β (competence) = 0.8  ← 大幅増加
- γ (novelty) = 0.1     ← 減少

**目的**:
- Competence報酬の寄与を増やす
- 破綻への「向き合い」を促進

### 実験B: Competence報酬のスケーリング

**設定**:
- Competence報酬の計算式を変更
- `R_competence = (coherence_prev - coherence_curr) × attention × 100`

**目的**:
- Competence報酬の絶対値を増やす
- 内発的報酬への寄与を可視化

### 実験C: Curiosity報酬の符号確認

**設定**:
- Curiosity報酬の計算式を確認
- 必要に応じて符号を反転

**目的**:
- Curiosity報酬が負になる原因を特定
- 正しい計算式に修正

## 結論

可視化から得られた主要な洞察：

1. **内発的報酬は改善している**（Total Intrinsic Reward）
   - Agent Cは確実に学習している

2. **Novelty報酬が支配的**
   - 内発的報酬の改善はほぼNovelty報酬のみによる
   - Curiosity（負）とCompetence（極小）は寄与していない

3. **Competence報酬の問題**
   - 値が極小（10^-5オーダー）
   - これは、保留構造の創発を阻害している可能性

4. **Curiosity報酬の問題**
   - 常に負の値
   - 計算式の確認が必要

**次のステップ**:
1. Curiosity報酬の計算式を確認・修正
2. Competence報酬のスケーリングを調整
3. ハイパーパラメータ（α, β, γ）を再調整
4. 再実験を実施

---

**観察日**: 2026-02-13
**観察者**: AI Agent (Manus)

## Agent Cの内部状態の分析

### Average Coherence Signal（左）
**観察**:
- 値の範囲: 0.41 ~ 0.47
- 高頻度の振動（ノイズが多い）
- 明確なトレンドなし（平均は一定）

**解釈**:
- Coherence signalは安定している（平均 ≈ 0.43）
- Agent Cは破綻を「避けていない」
- これは設計意図通り：coherenceを最小化していない

**理論的意義**:
- AGENT_GUIDELINES.mdの原則「Coherence Signalは最小化すべき損失ではない」が実装されている
- 保留構造の創発条件の1つ（破綻が死を意味しない）が満たされている

### Average Uncertainty（中央）
**観察**:
- 値の範囲: 66.6 ~ 67.15
- **明確な下降トレンド**: 67.05 → 66.65
- エピソード50以降で加速的に減少

**解釈**:
- Uncertaintyは減少している（学習が進んでいる証拠）
- これは、Agent Cの信念分布が「確信」を持つようになっていることを示す

**Curiosity報酬との矛盾**:
- Curiosity報酬 = uncertainty_prev - uncertainty_curr
- Uncertaintyが減少している → uncertainty_prev > uncertainty_curr
- したがって、Curiosity報酬は**正**であるべき
- しかし、実際のCuriosity報酬は**負**

**結論**:
- **Curiosity報酬の計算式に符号の誤りがある可能性が高い**

### Average Valence（右）
**観察**:
- 値: 0.0（完全に一定）
- 全エピソードで変化なし

**問題確認**:
- Valenceが全く更新されていない
- これは重大なバグ

**次のステップ**:
- ValenceMemoryV2の実装を確認
- Valenceの更新ロジックをデバッグ

## 価値関数の分析

### Value Function (Start vs End)（左）
**観察**:
- Start（青）: -0.4 → +0.4（大幅な上昇）
- End（赤）: 初期は低いが、徐々に上昇
- エピソード90以降で急上昇

**解釈**:
- 価値関数は正しく学習している
- エピソード開始時の状態の価値が上昇 → Agent Cが「良い初期状態」を生成できるようになった
- エピソード終了時の価値も上昇 → 終了状態も価値が高い

**興味深い観察**:
- Start > End（ほとんどのエピソードで）
- これは、エピソード開始時の方が価値が高いことを意味
- 理由: エピソードが進むにつれて、内発的報酬が減少する？

### TD Loss（右）
**観察**:
- 初期: 0.01 ~ 0.03（高い）
- 中期: 0.002 ~ 0.015（減少）
- 後期: 0.001 ~ 0.01（低い）

**解釈**:
- TD lossは減少している → 価値関数の推定精度が向上
- ただし、完全には収束していない（まだ変動がある）
- より長期の訓練で、さらに収束する可能性

## 総合的な洞察

### 1. Uncertaintyの減少とCuriosity報酬の矛盾

**観察**:
- Uncertainty: 67.05 → 66.65（減少）
- Curiosity報酬: -0.098（常に負）

**結論**:
- **Curiosity報酬の計算式に符号の誤りがある**
- 修正が必要

### 2. Valenceの完全な停滞

**観察**:
- Valence: 0.0（全エピソードで一定）

**結論**:
- **ValenceMemoryV2が機能していない**
- 重大なバグ

### 3. 価値関数の成功

**観察**:
- Value (Start): -0.16 → +0.01（上昇）
- TD loss: 減少傾向

**結論**:
- 価値関数は正しく機能している
- Agent Cの学習を導いている

### 4. Coherenceの安定性

**観察**:
- Coherence: 0.43（一定）

**結論**:
- Coherence最小化崩壊は発生していない
- 設計意図通り

## 優先度付き問題リスト

### 優先度1（即時修正が必要）

1. **Curiosity報酬の符号エラー**
   - 症状: Uncertaintyが減少しているのに、Curiosity報酬が負
   - 修正: `src/models/intrinsic_reward.py`の計算式を確認

2. **Valenceの更新停止**
   - 症状: Valenceが0.0で固定
   - 修正: `src/models/valence_v2.py`の更新ロジックを確認

### 優先度2（調整が必要）

3. **Competence報酬のスケーリング**
   - 症状: 値が極小（10^-5オーダー）
   - 調整: スケーリング係数を追加、またはβを増やす

4. **ハイパーパラメータの再調整**
   - 現状: α=0.3, β=0.5, γ=0.2
   - 提案: α=0.1, β=0.8, γ=0.1（Competence重視）

---

**追記日**: 2026-02-13
**追記者**: AI Agent (Manus)
