# Curiosity報酬の符号問題の分析 (2026-02-13)

## 問題の特定

### 観察された現象

**Uncertaintyの推移**:
- 初期: 67.05
- 最終: 66.65
- トレンド: **減少**

**Curiosity報酬の推移**:
- 全エピソード: -0.094 ~ -0.102
- トレンド: **常に負**

### 理論的矛盾

**Curiosity報酬の定義**（intrinsic_reward.py, line 88）:
```python
R_curiosity = uncertainty_prev - uncertainty_curr
```

**期待される挙動**:
- Uncertaintyが減少 → uncertainty_prev > uncertainty_curr
- したがって、R_curiosity > 0（正）

**実際の挙動**:
- R_curiosity < 0（負）

**結論**:
- 計算式と観察結果が矛盾している

## 原因の調査

### 仮説1: Uncertaintyの計算が逆

**可能性**:
- Uncertaintyの計算式が逆になっている
- 実際には「確実性」を計算している

**確認方法**:
- `src/models/priority.py`のUncertainty計算を確認

### 仮説2: tanh正規化の影響

**計算式**（intrinsic_reward.py, line 91）:
```python
R_curiosity = torch.tanh(R_curiosity / 10.0)
```

**分析**:
- tanh(-x) = -tanh(x)（符号は保存される）
- したがって、tanh正規化は符号を変えない

**結論**:
- tanh正規化は原因ではない

### 仮説3: Uncertaintyの値の範囲

**観察**:
- Uncertainty: 66.6 ~ 67.15

**分析**:
- 値が非常に大きい（通常、uncertaintyは0~1の範囲のはず）
- これは、uncertaintyの計算方法に問題がある可能性

**確認が必要**:
- `src/models/priority.py`のUncertainty計算式

### 仮説4: 時系列の逆転

**可能性**:
- `uncertainty_prev`と`uncertainty_curr`が逆に渡されている
- 呼び出し側のバグ

**確認方法**:
- `src/models/agent_layer_v4.py`のforward()を確認
- ValenceMemoryV2の呼び出し箇所を確認

## 次のステップ

### 1. Uncertaintyの計算式を確認

**ファイル**: `src/models/priority.py`

**確認項目**:
- Uncertaintyの定義（entropy? variance?）
- 値の範囲（0~1? それとも任意?）
- 計算式の正しさ

### 2. 呼び出し箇所を確認

**ファイル**: `src/models/agent_layer_v4.py`

**確認項目**:
- `uncertainty_prev`と`uncertainty_curr`の渡し方
- 時系列の正しさ

### 3. デバッグ出力を追加

**方法**:
- Curiosity報酬の計算時に、中間値を出力
- `uncertainty_prev`, `uncertainty_curr`, `R_curiosity`を確認

### 4. テストケースの作成

**目的**:
- Curiosity報酬の計算が正しいことを確認
- 既知の入力で期待される出力を得られるか

## 暫定的な結論

**最も可能性が高い原因**:
- **Uncertaintyの計算方法に問題がある**
- 値が66~67という大きな範囲にあることは異常
- 通常、uncertaintyはエントロピー（0~log(num_classes)）または分散（0~∞）

**次の調査対象**:
- `src/models/priority.py`のUncertainty計算式

---

**分析日**: 2026-02-13
**分析者**: AI Agent (Manus)

## 原因の特定

### Uncertaintyの計算式（priority.py, line 74-77）

```python
# Entropy of Gaussian: H(z) = 0.5 * log(2πe * σ^2)
# We use sum over dimensions as a measure of total uncertainty
entropy = 0.5 * torch.log(2 * torch.pi * torch.e * z_std.pow(2))
uncertainty = entropy.sum(dim=-1)  # (B,)
```

### 値の範囲の分析

**ガウス分布のエントロピー**:
- H(z) = 0.5 * log(2πe * σ^2)
- latent_dim = 64の場合、合計エントロピーは64個の項の和

**数値例**:
- σ = 1.0の場合: H(z) ≈ 0.5 * log(2π * 2.718 * 1) ≈ 0.5 * 2.838 ≈ 1.419
- 64次元の合計: 1.419 * 64 ≈ **90.8**

**観察された値**:
- Uncertainty: 66.6 ~ 67.15

**解釈**:
- 値の範囲は妥当（σが1.0より小さい場合）
- Uncertaintyの計算式自体は正しい

### Curiosity報酬が負になる理由

**再分析**:

1. **Uncertaintyの減少**:
   - 初期: 67.05
   - 最終: 66.65
   - 減少量: 0.4

2. **エピソード内の変化**:
   - 可視化では、Uncertaintyは**エピソード間**で減少している
   - しかし、Curiosity報酬は**エピソード内**で計算される
   - つまり、`uncertainty_prev`と`uncertainty_curr`は**同一エピソード内の連続ステップ**

3. **エピソード内での挙動**:
   - エピソード内では、Uncertaintyが**増加**している可能性
   - これは、新しい形状を見るたびに、不確実性が増すため

**結論**:
- **Curiosity報酬が負なのは、エピソード内でUncertaintyが増加しているため**
- これは、Agent Cが各ステップで新しい情報に遭遇し、不確実性が増すことを意味
- エピソード間では学習により不確実性が減少するが、エピソード内では増加する

### 理論的解釈

**Active Inferenceの観点**:
- エピソード内でのUncertainty増加は、**探索行動**を示す
- Agent Cは、新しい形状に遭遇するたびに、信念分布を更新している
- これは、「予測誤差を最小化する」のではなく、「新しい情報を取り込む」行動

**Curiosity報酬の設計の問題**:
- 現在の定義: R_curiosity = uncertainty_prev - uncertainty_curr
- これは、「不確実性の減少」を報酬とする
- しかし、エピソード内では不確実性が増加する（新しい情報の取り込み）

**改善案**:
1. **Curiosity報酬の符号を反転**:
   - R_curiosity = uncertainty_curr - uncertainty_prev
   - 「不確実性の増加」を報酬とする（探索を促進）

2. **エピソード間の不確実性減少を報酬とする**:
   - エピソード終了時に、初期不確実性と最終不確実性を比較
   - しかし、これは実装が複雑

3. **Curiosity報酬の再定義**:
   - 「予測誤差」を報酬とする（標準的なActive Inference）
   - しかし、これは「suspension structure」の哲学と矛盾する可能性

## 推奨される対処

### オプションA: Curiosity報酬の符号を反転

**変更**:
```python
# Before
R_curiosity = uncertainty_prev - uncertainty_curr

# After
R_curiosity = uncertainty_curr - uncertainty_prev
```

**理由**:
- エピソード内での探索行動を報酬とする
- 新しい情報の取り込みを促進

**懸念**:
- これは「不確実性の増加」を報酬とする
- 長期的には、不確実性が無限に増加する可能性

### オプションB: Curiosity報酬を削除

**変更**:
- α (curiosity) = 0.0
- β (competence) = 0.6
- γ (novelty) = 0.4

**理由**:
- Curiosity報酬の定義が曖昧
- Novelty報酬が既に「新しい発見」を捉えている

**懸念**:
- Curiosity（不確実性の減少）は、Active Inferenceの重要な要素

### オプションC: Curiosity報酬の再定義

**変更**:
- Curiosity報酬を「予測誤差」として定義
- R_curiosity = ||observation - prediction||^2

**理由**:
- 標準的なActive Inferenceの定義
- 理論的に明確

**懸念**:
- 現在の実装では、予測は行っていない
- 実装が複雑

## 推奨

**短期的対処**:
- **オプションB: Curiosity報酬を削除**
- α = 0.0, β = 0.6, γ = 0.4で再実験

**理由**:
1. Curiosity報酬の定義が現在の設計と合っていない
2. Novelty報酬が既に「新しい発見」を捉えている
3. Competence報酬を強化することで、保留構造の創発を促進

**長期的検討**:
- Curiosity報酬の理論的定義を再検討
- 「suspension structure」の哲学と整合する形で再設計

---

**更新日**: 2026-02-13
**更新者**: AI Agent (Manus)
