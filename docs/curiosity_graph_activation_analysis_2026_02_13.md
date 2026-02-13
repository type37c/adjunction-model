# Curiosity報酬v4: グラフ活性化版の分析

**日付**: 2026-02-13  
**実験**: Agent C v4 with Curiosity as Graph Activation

---

## 背景

Curiosity報酬を「Uncertaintyの減少」として定義したv3では、Curiosity報酬がほぼゼロ（0.03%以下）になる問題が発生した。原因は、F/Gが固定されているため、Uncertaintyが減少せず（むしろ増加: 67.0 → 67.4）、`uncertainty_start - uncertainty_curr` が常に負またはゼロになることだった。

ユーザーとの議論を通じて、**確信度（confidence）はグラフの活性化度から来る**という洞察を得た。これに基づき、Curiosity報酬を再定義した。

---

## 新しい定義: Curiosity = グラフの活性化度

### 理論的根拠

- **確信度**: F層（affordance graph）がどれだけ活性化しているか
- **Curiosity**: グラフ構造にどれだけ関与しているか（attention_weight）
- **解釈**: 「グラフ構造を使って理解しようとしている」こと自体が報酬

### 実装

```python
# 旧: Curiosity = Uncertaintyの減少（エピソード開始時から）
R_curiosity = (uncertainty_start - uncertainty_curr) / uncertainty_start

# 新: Curiosity = グラフの活性化度（Attention weight）
R_curiosity = attention_weight
```

---

## 実験結果

### 数値比較

| バージョン | 最終内発的報酬 | Curiosity寄与 | Competence寄与 | Novelty寄与 |
|:---|---:|---:|---:|---:|
| **v3** (Competence-only) | 0.421 | 0.0% | 59.5% | 40.5% |
| **v4** (Graph Activation) | 0.254 | 12.3% | 9.2% | 78.5% |

### 観察

1. **Curiosity報酬が機能している**: 0.1018（一定値）→ 12.3%の寄与
2. **内発的報酬の総量が減少**: 0.421 → 0.254（-40%）
3. **Novelty報酬が支配的**: 78.5%（v3の40.5%から大幅増加）
4. **Competence報酬が激減**: 59.5% → 9.2%

### Curiosity報酬の推移

- **エピソード0**: 0.1018（47.6%）
- **エピソード50**: 0.1018（33.4%）
- **エピソード99**: 0.1018（12.3%）

**重要な発見**: Curiosity報酬の絶対値は一定（0.1018）だが、Novelty報酬の急増により相対的な寄与率が減少している。

---

## 問題点

### 1. Curiosity報酬が一定値

`attention_weight`がほぼ一定（~0.1018）であるため、Curiosity報酬が変化しない。これは、Agent Cがattention mechanismを学習していないことを示唆する。

**原因の仮説**:
- Attention weightが固定されている可能性
- Attention mechanismが訓練されていない可能性
- `attention_weight`の定義が不適切な可能性

### 2. Novelty報酬の異常な増加

Novelty報酬が0.037 → 0.652（+1661%）と異常に増加している。これは、KL divergenceが爆発的に増加していることを意味する。

**原因の仮説**:
- Agent Cが「驚き」を最大化するように学習している
- Novelty報酬のスケーリング（×0.1）が不適切
- KL divergenceの計算に問題がある可能性

### 3. Competence報酬の激減

Competence報酬の寄与が59.5% → 9.2%に激減した。これは、α/β/γのバランスが変わったためだが、Competenceの重要性（保留構造の創発）を考えると問題である。

---

## 理論的洞察

### Curiosity = Attention Weightの妥当性

**肯定的な側面**:
- 「グラフに触れた」こと自体が報酬という解釈は直感的
- 確信度の定義（グラフ活性化）と整合的
- F/G固定の設定でも機能する

**否定的な側面**:
- Attention weightが一定では、学習の進展を表せない
- Attention weightは「どれだけ注目するか」であり、「どれだけ理解したか」ではない
- Curiosityの本質（「分かりたい」）を捉えていない可能性

### 確信度とCuriosityの関係

ユーザーの洞察「確信度はグラフの活性化から来る」は正しいと思われるが、**Curiosity = Attention weight**という単純な等式は不十分かもしれない。

**より適切な定義**:
- **確信度**: グラフの活性化パターンの安定性（時間的変化）
- **Curiosity**: 確信度の変化（不確実→確実への移行）

つまり、Curiosityは「グラフ活性化の変化」として定義すべきかもしれない。

---

## 次のステップ

### 短期（デバッグ）

1. **Attention weightの値を確認する**
   - なぜ一定値（0.1018）なのか？
   - Attention mechanismは訓練されているか？

2. **Novelty報酬の爆発を調査する**
   - KL divergenceの値を確認
   - スケーリングを調整（0.1 → 0.01？）

3. **ハイパーパラメータの再調整**
   - α=0.2, β=0.5, γ=0.3 → α=0.3, β=0.5, γ=0.2？
   - Competence報酬の寄与を回復させる

### 中期（理論的改善）

4. **Curiosity報酬を「グラフ活性化の変化」として再定義**
   ```python
   # Curiosity = グラフ活性化の時間的変化
   R_curiosity = |attention_weight_curr - attention_weight_prev|
   ```

5. **確信度を明示的に導入する**
   - グラフ活性化パターンの安定性を測定
   - 確信度の変化をCuriosity報酬とする

### 長期（構造的改善）

6. **F/Gの学習を再開する**
   - Agent CとF/Gの共進化を実現
   - 「分からない→分かる」の往復を可能にする

7. **確信度の創発を観察する**
   - 1000エピソードの長期訓練
   - 確信度の時間的パターンを分析

---

## 結論

Curiosity報酬を「グラフの活性化度」として再定義したv4は、**理論的には妥当だが、実装に問題がある**。

**主要な問題**:
1. Attention weightが一定値（学習していない）
2. Novelty報酬が爆発的に増加（バランスが崩れた）
3. Competence報酬の寄与が激減（保留構造の創発に悪影響）

**重要な洞察**:
- 確信度はグラフの活性化から来る（ユーザーの指摘は正しい）
- Curiosity = Attention weightは単純すぎる
- Curiosityは「グラフ活性化の変化」として定義すべきかもしれない

**次の方向性**:
1. Attention weightの一定値問題をデバッグ
2. Novelty報酬のスケーリングを調整
3. Curiosityを「グラフ活性化の変化」として再定義（v5）
