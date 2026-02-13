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
# Curiosity報酬の理論的整理と再設計 (2026-02-13)

## 概要

本ドキュメントは、現在無効化されているCuriosity報酬の理論的位置づけを整理し、再設計する。特に、「Curiosityは保留の話か、目的の話か」という根本的な問いに答える。

**関連文書**:
- [suspension_and_confidence.md](suspension_and_confidence.md): 保留構造の要件
- [purpose_space_P_design.md](purpose_space_P_design.md): 目的空間Pの設計
- [development_report_2026_02_13_final.md](development_report_2026_02_13_final.md): v3実験結果

---

## 1. 現状の整理

### 1.1 3つの内発的報酬の現状

| 報酬 | 現在の定義 | 状態 | 内発的報酬への寄与 |
|:---|:---|:---|---:|
| Curiosity | `-(uncertainty_curr - uncertainty_prev)` | **無効化**（α=0.0） | 0% |
| Competence | `coherence_curr × attention × 100` | 機能 | 59.5% |
| Novelty | `KL(posterior \|\| prior)` | 機能 | 40.5% |

### 1.2 Curiosity報酬が無効化された理由

**問題**: エピソード内でUncertaintyが増加するため、Curiosity報酬が常に負になる。

**原因**:
- 現在の定義: `R_curiosity = -(uncertainty_curr - uncertainty_prev)`
- これは「Uncertaintyの減少 = 理解が進んだ」を報酬とする
- しかし、エピソード内では新しい形状を見るため、Uncertaintyは増加する
- したがって、`uncertainty_curr > uncertainty_prev` → `R_curiosity < 0`

**対処**: Curiosity報酬を無効化（α=0.0）

---

## 2. 根本的な問い：Curiosityは保留か、目的か

### 2.1 保留構造の要件との対応

**suspension_and_confidence.mdの保留の3つの条件**:

1. **「確定」と「未確定」の両方の状態が可能であること**
2. **「確定を壊す」トリガーがあること**
3. **「未確定のまま保持する」能力があること**

**Curiosityはどれに対応するか？**

- **条件1（確定/未確定）**: Uncertaintyが高い = 未確定、低い = 確定
  - → Curiosityは「未確定を検出する」機能
  - → **保留の条件1に対応**

- **条件2（確定を壊す）**: Coherence breakdownがトリガー
  - → Curiosityは関与しない
  - → **保留の条件2には対応しない**

- **条件3（未確定のまま保持）**: 確信度の導入が必要
  - → Curiosityは関与しない
  - → **保留の条件3には対応しない**

**結論**: Curiosityは保留構造の**条件1（未確定の検出）**に対応する。

### 2.2 目的空間Pの3つの軸との対応

**purpose_space_P_design.mdのPの3つの軸**:

1. **Coherence軸**: 「壊れているものは注意すべき」
2. **Uncertainty軸**: 「分からないものは探索すべき」
3. **Valence軸**: 「経験から学んだ良し悪し」

**Curiosityはどれに対応するか？**

- **Coherence軸**: Competence報酬が対応
- **Uncertainty軸**: **Curiosity報酬が対応**
- **Valence軸**: Valence Memoryが対応（既に実装済み）

**結論**: Curiosityは目的空間Pの**Uncertainty軸**に対応する。

### 2.3 総合的な結論

**Curiosityは保留の話か、目的の話か？**

**答え**: **両方である。**

- **保留の観点**: Curiosityは「未確定の検出」（保留の条件1）を担う
- **目的の観点**: CuriosityはPのUncertainty軸（「分からないものは探索すべき」）を担う

**より正確には**: Curiosityは**目的空間Pの一部**であり、Pを通じて**保留構造の創発に寄与**する。

---

## 3. 3つの報酬の理論的役割の整理

### 3.1 保留構造の要件との対応表

| 保留の要件 | 対応する報酬 | 役割 |
|:---|:---|:---|
| 条件1: 確定/未確定の両立 | **Curiosity** | 未確定を検出し、探索を促す |
| 条件2: 確定を壊すトリガー | Coherence Signal（報酬ではない） | 破綻を検出 |
| 条件3: 未確定のまま保持 | （未実装: 確信度） | 保留状態の維持 |

### 3.2 目的空間Pの軸との対応表

| Pの軸 | 対応する報酬 | 問い | 役割 |
|:---|:---|:---|:---|
| Coherence軸 | **Competence** | 「困難に向き合いたい」 | 破綻への注目を促す |
| Uncertainty軸 | **Curiosity** | 「何が起きているか知りたい」 | 探索を促す |
| Valence軸 | （Valence Memory） | 「過去に良い結果が出た」 | 経験的価値判断 |

### 3.3 3つの報酬の独立性

**重要**: 3つの報酬は独立した軸であるべき。

| 報酬 | 問い | 高いとき | 低いとき |
|:---|:---|:---|:---|
| Curiosity | 「分かるか？」 | 分からない（探索） | 分かる（確定） |
| Competence | 「向き合うか？」 | 破綻に注目（保留） | 破綻を無視 |
| Novelty | 「新しいか？」 | 新しい（驚き） | 見慣れた |

**現在の問題**: CuriosityとCompetenceが重複している可能性。

- Competence = `coherence × attention`（破綻への注目）
- もしCuriosity = `coherence`（破綻の大きさ）なら、重複

**解決策**: Curiosityを「Uncertaintyの変化」として定義し、Coherenceとは独立させる。

---

## 4. Curiosity報酬の再設計

### 4.1 設計原則

1. **Pの Uncertainty軸に対応**: 「分からないものは探索すべき」
2. **CompetenceやNoveltyと独立**: 重複を避ける
3. **エピソード内で正の値を取る**: 常に負にならない
4. **理論的に正当化可能**: Active InferenceやRNDと整合

### 4.2 候補案の検討

#### 案A: Uncertaintyの絶対値（現在の逆符号）

```python
R_curiosity = uncertainty_curr
```

**理由**: Uncertaintyが高い = 分からない = 探索すべき

**問題**: 
- Uncertaintyは常に高い（67程度）
- 報酬の変化が小さい
- 「探索の結果として理解が進む」という報酬がない

#### 案B: Uncertaintyの減少（エピソード間）

```python
R_curiosity = uncertainty_episode_start - uncertainty_episode_end
```

**理由**: エピソードを通じて理解が進んだことを報酬とする

**問題**:
- エピソード終了時にしか計算できない
- ステップごとの報酬が必要な現在の実装と不整合

#### 案C: 予測誤差（Coherence Signalの再利用）

```python
R_curiosity = coherence_curr
```

**理由**: 予測誤差が大きい = 驚き = 探索の価値

**問題**:
- Competence報酬と重複（Competence = coherence × attention）
- 独立性の原則に反する

#### 案D: 信念の更新量（KLダイバージェンスの変化）

```python
R_curiosity = KL_divergence_curr - KL_divergence_prev
```

**理由**: 信念が大きく変化した = 新しいことを学んだ

**問題**:
- Novelty報酬と重複（Novelty = KL_divergence）
- 独立性の原則に反する

#### 案E: Uncertaintyの減少（ステップ間、符号反転）

```python
R_curiosity = uncertainty_prev - uncertainty_curr
```

**理由**: Uncertaintyが減少した = 理解が進んだ = 報酬

**問題**（これが現在の実装）:
- エピソード内でUncertaintyが増加するため、常に負

#### 案F: Uncertaintyの変化の絶対値

```python
R_curiosity = |uncertainty_curr - uncertainty_prev|
```

**理由**: Uncertaintyが大きく変化した = 新しい情報を得た

**利点**:
- 増加でも減少でも報酬
- 「探索」（増加）と「理解」（減少）の両方を報酬
- 常に正

**問題**:
- 理論的正当化が弱い
- 「変化しないこと」（安定）が報酬にならない

#### 案G: Uncertaintyの減少（エピソード内、正規化）

```python
# エピソード開始時のUncertaintyを基準とする
R_curiosity = max(0, uncertainty_start - uncertainty_curr) / uncertainty_start
```

**理由**: エピソード開始時から理解が進んだ量を報酬とする

**利点**:
- 常に正または0
- 「理解の進展」を明確に報酬
- 正規化により、スケールが安定

**問題**:
- エピソード開始時のUncertaintyを記録する必要がある
- 実装がやや複雑

### 4.3 推奨案：案G（エピソード内での理解の進展）

**最終的な定義**:

```python
# エピソード開始時
uncertainty_start = uncertainty_curr

# 各ステップ
R_curiosity = max(0, uncertainty_start - uncertainty_curr) / (uncertainty_start + 1e-8)
```

**理論的正当化**:

1. **Pの Uncertainty軸に対応**: 「分からない → 分かる」の移行を報酬
2. **保留の条件1に対応**: 「未確定 → 確定」の移行を検出
3. **独立性**: Coherence（破綻）やKL（新規性）とは独立
4. **Active Inferenceとの整合**: 「自由エネルギーの減少」に相当

**期待される効果**:

- Agent Cは、同じ形状を繰り返し見ることで理解を深める動機を持つ
- Novelty報酬（新しいものを見る）とバランスを取る
- Competence報酬（破綻に注目）と協調する

---

## 5. 実装計画

### 5.1 修正するファイル

**src/models/intrinsic_reward.py**:
- `uncertainty_start`を記録する機構を追加
- Curiosity報酬の計算式を案Gに変更
- α（Curiosity報酬の重み）を0.0から0.2に変更

### 5.2 実装の詳細

```python
class IntrinsicRewardComputation(nn.Module):
    def __init__(self, ...):
        ...
        self.alpha_curiosity = 0.2  # Re-enable curiosity
        self.uncertainty_start = None  # Will be set at episode start
    
    def reset_episode(self):
        """Call at the start of each episode"""
        self.uncertainty_start = None
    
    def forward(self, ...):
        # (1) Curiosity reward: understanding progress within episode
        if self.uncertainty_start is None:
            self.uncertainty_start = uncertainty_curr.detach().clone()
        
        # Normalized reduction in uncertainty from episode start
        uncertainty_reduction = torch.clamp(
            self.uncertainty_start - uncertainty_curr, min=0.0
        )
        R_curiosity = uncertainty_reduction / (self.uncertainty_start + 1e-8)
        
        # (2) Competence reward: attending to breakdowns
        ...
        
        # (3) Novelty reward: unexpected discoveries
        ...
        
        # Combine
        R_intrinsic = (
            self.alpha_curiosity * R_curiosity +
            self.beta_competence * R_competence +
            self.gamma_novelty * R_novelty
        )
```

### 5.3 訓練ループの修正

**src/training/train_agent_value_based.py**:
- エピソード開始時に`intrinsic_reward_module.reset_episode()`を呼び出す

```python
def train_episode(self, ...):
    # Reset episode-level state
    self.model.agent_c.intrinsic_reward.reset_episode()
    
    for step in range(episode_length):
        ...
```

### 5.4 ハイパーパラメータの調整

| パラメータ | v3（現在） | v4（提案） | 理由 |
|:---|---:|---:|:---|
| α（Curiosity） | 0.0 | 0.2 | 再有効化 |
| β（Competence） | 0.6 | 0.5 | バランス調整 |
| γ（Novelty） | 0.4 | 0.3 | バランス調整 |

**合計**: 1.0（変わらず）

**期待される寄与率**:
- Curiosity: ~20%
- Competence: ~50%
- Novelty: ~30%

---

## 6. 検証計画

### 6.1 短時間検証（30エピソード）

**目的**: Curiosity報酬が正しく機能するか確認

**観察項目**:
1. Curiosity報酬が正の値を取るか
2. Curiosity報酬がエピソード内で増加するか（理解の進展）
3. 内発的報酬への寄与率が期待通りか（~20%）

### 6.2 本格的実験（100エピソード）

**目的**: 3つの報酬が全て機能する状態での保留構造の萌芽を観察

**観察項目**:
1. 内発的報酬の成長パターン（v3との比較）
2. Uncertaintyの推移（v3では増加、v4では？）
3. 価値関数の学習速度（v3との比較）
4. 保留構造の萌芽の兆候（指数関数的成長の時期）

### 6.3 成功の基準

**最低限の成功**:
- Curiosity報酬が正の値を取る
- 内発的報酬がv3と同等以上（0.421以上）
- Coherence最小化崩壊が発生しない

**理想的な成功**:
- 内発的報酬がv3を上回る（0.5以上）
- Uncertaintyが「増加→減少」の往復を示す（保留の萌芽）
- 価値関数の学習が加速

---

## 7. 理論的考察

### 7.1 Curiosityと保留構造の関係

**仮説**: Curiosityは保留構造の「未確定→確定」の移行を促進する。

**根拠**:
- suspension_and_confidence.mdの条件1: 「確定と未確定の両立」
- Curiosity報酬は「未確定（高Uncertainty）→ 確定（低Uncertainty）」を報酬とする
- これは、保留の「未確定のまま保持」ではなく、「未確定から確定への移行」を促す

**予測**:
- Curiosity報酬の追加により、Uncertaintyは減少する方向に動く
- v3では Uncertainty が増加（67.0 → 67.9）したが、v4では減少する可能性
- これは、「確定→未確定→確定」の往復の後半部分に相当

### 7.2 3つの報酬のダイナミクス

**Novelty報酬**: 「新しいものを見る」→ Uncertainty増加

**Curiosity報酬**: 「理解を深める」→ Uncertainty減少

**Competence報酬**: 「破綻に注目する」→ Coherence変化（増加も減少も）

**予測されるダイナミクス**:
1. Novelty報酬により、Agent Cは新しい形状を探索（Uncertainty増加）
2. Curiosity報酬により、Agent Cは同じ形状を繰り返し見て理解（Uncertainty減少）
3. Competence報酬により、Agent Cは破綻に注目し続ける（Coherence安定）

**これは、保留構造の「確定→未確定→確定」の往復に対応する可能性がある。**

### 7.3 F/Gの学習再開との関係

現在、F/Gは固定されている。F/Gの学習を再開すると：

- Competence報酬の定義を変更する必要がある（「破綻への注目」→「破綻の減少」？）
- Curiosity報酬は変更不要（Uncertaintyの減少は常に報酬）
- Novelty報酬は変更不要（KLダイバージェンスは常に報酬）

**結論**: Curiosity報酬の再設計（案G）は、F/Gの学習再開後も有効。

---

## 8. 次のステップ

### 即時

1. **intrinsic_reward.pyの修正** — Curiosity報酬を案Gで実装
2. **train_agent_value_based.pyの修正** — エピソードリセット機構を追加
3. **短時間検証（30エピソード）** — Curiosity報酬が機能するか確認

### 短期

4. **本格的実験（100エピソード）** — 3つの報酬が全て機能する状態で
5. **結果の分析** — v3との比較、保留構造の萌芽の観察
6. **議論ログの更新** — Curiosity報酬の再設計に関する考察

### 中期

7. **F/Gの学習再開** — 「確定→未確定→確定」の往復を実現
8. **確信度の導入** — 保留の条件3（未確定のまま保持）を実装

---

## 9. 結論

**Curiosityは保留の話か、目的の話か？**

**答え**: **両方である。Curiosityは目的空間PのUncertainty軸であり、保留構造の「未確定→確定」の移行を促進する。**

**Curiosity報酬の再設計（案G）**:
```python
R_curiosity = max(0, uncertainty_start - uncertainty_curr) / uncertainty_start
```

この定義により：
- Pの Uncertainty軸（「分からないものは探索すべき」）に対応
- 保留の条件1（確定/未確定の両立）に対応
- CompetenceやNoveltyと独立
- 常に正の値を取る
- 理論的に正当化可能（Active Inferenceの自由エネルギー減少）

**期待される効果**:
- 3つの報酬が全て機能する状態で、保留構造の萌芽を観察
- Uncertaintyの「増加→減少」の往復（保留の萌芽）
- 内発的報酬のさらなる改善

---

**作成日**: 2026-02-13（夕方）
**ステータス**: 理論的整理完了、実装準備完了
