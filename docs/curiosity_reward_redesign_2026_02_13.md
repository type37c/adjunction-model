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
