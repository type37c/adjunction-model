# Curiosity報酬の根本的問題：なぜ作動しないのか

**Date**: 2026-02-13  
**Context**: v3, v4, v5と3回の再実装を試みたが、いずれもCuriosity報酬がほぼゼロのまま

---

## 1. 試みた定義と失敗の記録

### v3: Uncertaintyの減少（エピソード開始時から）

**定義**:
```python
R_curiosity = (uncertainty_start - uncertainty_curr) / uncertainty_start
```

**期待**: エピソード内でUncertaintyが減少すれば報酬

**結果**: **失敗**（Curiosity = 0.03%）

**原因**: Uncertaintyはエピソード内で**増加する**（67.0 → 67.9）
- 新しい形状を見るたびにUncertaintyが増加
- `uncertainty_start < uncertainty_curr` → `R_curiosity < 0` → `clamp(min=0)` → `R_curiosity = 0`

---

### v4: グラフの活性化度（Attention weight）

**定義**:
```python
R_curiosity = attention_weight  # グラフにどれだけ関与しているか
```

**期待**: グラフ構造を使うこと自体が報酬

**結果**: **部分的成功**（Curiosity = 12.3%）

**問題**:
1. **Attention weightが一定値**（0.1018）→ 学習していない
2. **Novelty報酬が爆発**（78.5%）→ バランスが崩れた
3. **Competence報酬が激減**（9.2%）→ 保留構造の創発に悪影響

**原因**: Attention weightそのものは「活性化の度合い」であり、「理解の進展」ではない

---

### v5: 確信度の変化（Attentionエントロピーの減少）

**定義**:
```python
confidence = 1 - normalized_entropy(priority_distribution)
R_curiosity = confidence_curr - confidence_prev
```

**期待**: 選択肢が絞られること（確信度の増加）が報酬

**結果**: **未確認**（実装完了、実験実行中だがCuriosity = 0の可能性が高い）

**予想される問題**: Priority分布が変化していない可能性
- Priority = coherence × uncertainty × valence
- これらが安定していれば、priority分布も安定
- 確信度も変化しない → Curiosity = 0

---

## 2. 根本的な問題の特定

### 2.1 構造的制約：F/Gが固定されている

**現在の設定**:
- F（Shape → Affordance）とG（Affordance → Shape）は**事前訓練済みで固定**
- Agent Cは**FiLMパラメータで微調整**するのみ

**帰結**:
- **予測精度は改善しない**（F/Gの重みが変わらない）
- **Uncertaintyは減少しない**（RSSMの予測能力が向上しない）
- **Priority分布は大きく変化しない**（coherence, uncertaintyが安定）

**結論**: **「理解の進展」が構造的に起きない**

---

### 2.2 Curiosityの本質：「分からない → 分かる」の移行

すべてのCuriosity定義に共通する前提：

| 定義 | 前提となる変化 |
|:---|:---|
| v3: Uncertainty減少 | Uncertaintyが減少する |
| v4: グラフ活性化 | Attention weightが変化する |
| v5: 確信度増加 | Priority分布が変化する |

**しかし、現在の設定ではこれらの変化が起きない。**

---

### 2.3 「分かる」とは何か？

**Active Inferenceの視点**:
- 「分かる」= 予測誤差の減少
- 「分かる」= 自由エネルギーの減少
- 「分かる」= モデルの改善

**現在の実装**:
- F/Gが固定 → モデルが改善しない
- Agent CのRSSMは学習する → しかし、F/Gを通した観測が変わらない
- **Agent Cは「自分の内部モデル」を改善できるが、「世界のモデル（F/G）」を改善できない**

**結論**: Agent Cにとって「分かる」ことの範囲が限定的

---

### 2.4 η/ε余白保持の洞察との関係

`adjunction_slack_analysis_2026_02_13.md`での洞察：
> Agent Cは余白を縮小したい（Curiosity）  
> しかし世界（F/G）は余白を維持する  
> **この緊張こそが保留構造である**

**しかし**:
- 現在、F/Gは余白を「維持」しているのではなく、「固定」している
- Agent Cは余白を「縮小」できない（F/Gを変えられない）
- **緊張ではなく、膠着状態**

**結論**: Curiosityが機能するには、**Agent CがF/Gに影響を与える経路**が必要

---

## 3. なぜCompetenceとNoveltyは機能するのか？

### Competence報酬（v3）

**定義**:
```python
R_competence = coherence_signal  # 破綻への注目
```

**なぜ機能するか**:
- Coherence signalは**F/Gの出力の結果**（再構成誤差）
- F/Gが固定でも、**入力（形状）が変われば出力（coherence）も変わる**
- Agent Cは「どの形状を見るか」を選べないが、**coherence自体は変化する**

**重要**: Competenceは「理解の進展」ではなく、「破綻の存在」を報酬とする

---

### Novelty報酬

**定義**:
```python
R_novelty = KL(posterior || prior)  # RSSMのKLダイバージェンス
```

**なぜ機能するか**:
- RSSMの内部状態の変化を測定
- F/Gが固定でも、**Agent Cの内部状態は変化する**
- 新しい観測 → posteriorが変化 → Novelty報酬

**重要**: Noveltyは「世界の理解」ではなく、「内部状態の変化」を報酬とする

---

### Curiosityが機能しない理由

**Curiosityの本質**:
- 「理解の進展」= **モデルの改善**
- これには**学習可能なパラメータの更新**が必要

**現在の制約**:
- F/Gが固定 → 世界モデルが改善しない
- Agent CのRSSMは学習する → しかし、F/Gを通した観測が変わらないため、改善の余地が限定的

**結論**: **Curiosityは「学習」を前提とする報酬であり、F/G固定の設定では機能しにくい**

---

## 4. 根本的な解決策の方向性

### Option A: F/Gの学習を再開する

**アイデア**: F/GをAgent Cと共に学習させる

**効果**:
- 予測精度が改善する → Uncertaintyが減少する
- Priority分布が変化する → 確信度が変化する
- **Curiosity報酬が機能する**

**課題**:
- F/Gの訓練目的関数をどう設計するか？
- η/ε余白保持の洞察とどう両立するか？

---

### Option B: Curiosityを「学習」ではなく「探索」として再定義する

**アイデア**: Curiosityを「新しいものを見つけた」ことの報酬とする

**例**:
```python
# Curiosity = Noveltyの別バージョン（異なるスケール）
R_curiosity = ||latent_curr - latent_prev||  # 潜在状態の変化の大きさ
```

**効果**:
- F/G固定でも機能する
- Noveltyと重複するが、異なる側面を捉える可能性

**問題**: これは本当に「Curiosity」なのか？

---

### Option C: Curiosityを諦め、CompetenceとNoveltyの2本柱で進む

**アイデア**: 現在機能している2つの報酬に集中する

**効果**:
- 実装がシンプル
- 既に良い結果が出ている（v3: 内発的報酬+1584%）

**問題**: 
- 3本柱の理論的完全性が失われる
- 「理解の進展」を報酬化できない

---

### Option D: η/ε余白保持の実験を優先する

**アイデア**: Curiosityの問題を一旦置いて、η/ε余白保持の実験を先に行う

**理由**:
- η/ε余白保持は**より根本的な設計変更**
- これが成功すれば、Curiosityの定義も自然に見えてくる可能性
- F/Gの訓練方法を変えることで、「学習」の余地が生まれる

**効果**:
- 保留構造の創発に直接アプローチ
- Curiosityは後から追加できる

---

## 5. 推奨する方向性

**私の推奨**: **Option D（η/ε余白保持の実験を優先）**

**理由**:
1. **Curiosityは「余白の縮小」として定義されている**
   - 余白が保持されていなければ、縮小もできない
   - 現在は余白が「固定」されているため、Curiosityが機能しない

2. **η/ε余白保持は、F/Gの訓練方法を変える**
   - 再構成損失を削除 or 余白保持正則化を追加
   - これにより、F/Gの振る舞いが変わる
   - Agent CとF/Gの相互作用が生まれる可能性

3. **保留構造の創発が最優先目標**
   - Curiosityは手段であり、目的ではない
   - 保留構造が創発すれば、Curiosityの定義も自然に見えてくる

4. **理論的に一貫している**
   - `suspension_and_confidence.md`の洞察に基づく
   - `adjunction_slack_analysis_2026_02_13.md`で詳細に分析済み

---

## 6. まとめ

| 問い | 答え |
|:---|:---|
| **Curiosity報酬v5は作動する見込みがあるか？** | **低い**。Priority分布が変化しない可能性が高い。 |
| **根本的な問題は何か？** | **F/Gが固定されているため、「理解の進展」が構造的に起きない**。 |
| **なぜCompetenceとNoveltyは機能するのか？** | 「理解の進展」ではなく、「破綻の存在」と「内部状態の変化」を報酬とするため。 |
| **解決策は？** | **η/ε余白保持の実験を優先**し、F/Gの訓練方法を変える。これにより、Agent CとF/Gの相互作用が生まれ、Curiosityが機能する余地が生まれる。 |

---

## 7. 次のステップ（提案）

1. **Curiosity報酬v5のデバッグは一旦中断**
   - 確信度が変化しているか確認する価値はあるが、作動する見込みは低い

2. **η/ε余白保持の実験を実装**
   - `adjunction_slack_analysis_2026_02_13.md`のOption Cを実装
   - F/Gの訓練から再構成損失を削除
   - Counit ε を計算
   - η と ε を Agent C に提供

3. **実験結果を観察**
   - η が保持されるか？
   - 確信度が動的に変化するか？
   - 保留構造が創発するか？

4. **Curiosityを再定義**
   - 実験結果を見て、Curiosityの適切な定義を考える
   - 余白が動的に変化するなら、Curiosity = 余白の縮小が機能する可能性

---

**結論**: Curiosity報酬v5の作動見込みは低い。根本的な問題（F/G固定）を解決するため、η/ε余白保持の実験を優先すべき。
