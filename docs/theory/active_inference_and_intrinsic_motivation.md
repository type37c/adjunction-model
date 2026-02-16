# Active Inferenceと内発的動機：このプロジェクトとの関係

**作成日**: 2026-02-13

---

## 1. Active Inferenceと自由エネルギー原理の基本

### 1.1 自由エネルギー原理（Free Energy Principle, FEP）

Karl Fristonが提唱した脳の統一理論。核心的な主張は：

> **生物は「驚き（surprise）」を長期的に最小化するように行動する。**

ここで「驚き」とは、**感覚入力が自分の予測から外れる度合い**を意味します。

数学的には、**自由エネルギー（F）**という量を定義し、これを最小化することが驚きの最小化に対応します：

```
F = 予測誤差 + モデルの複雑さ
  = -log p(感覚|隠れ状態) + KL(q(隠れ状態)||p(隠れ状態))
```

### 1.2 Active Inference（能動的推論）

自由エネルギー原理を行動に拡張したもの。

**受動的推論（Passive Inference）**: 感覚入力を受け取り、世界の状態を推論する（知覚）

**能動的推論（Active Inference）**: 行動によって感覚入力を変化させ、予測誤差を減らす（行動）

つまり、**「世界を理解する」だけでなく、「世界を変えて理解しやすくする」**という発想です。

---

## 2. 内発的動機との関係

### 2.1 好奇心（Curiosity）の定式化

Active Inferenceでは、好奇心を以下のように定式化します：

> **好奇心 = 情報獲得の期待値（Expected Information Gain, EIG）**

エージェントは、「行動した結果、自分の信念がどれだけ変化するか」を予測し、変化が大きい行動を選びます。

これは、以下の2つの要素に分解されます：

1. **Epistemic Value（認識的価値）**: 不確実性を減らす行動の価値
   - 「よく分からないものを調べる」
   - Uncertaintyが高い状態を探索する

2. **Pragmatic Value（実用的価値）**: 目的を達成する行動の価値
   - 「報酬を得る」
   - 外部から与えられた目標を達成する

### 2.2 内発的動機の種類

Active Inferenceの枠組みでは、以下のような内発的動機が導出されます：

| 動機 | 定式化 | 意味 |
|:---|:---|:---|
| **好奇心** | EIG（情報獲得） | 不確実性を減らす |
| **有能感** | 予測誤差の減少 | 自分の行動で世界を変えられた |
| **新奇性追求** | モデルの複雑さの増加 | 新しいパターンを発見する |

---

## 3. このプロジェクトとの関係

### 3.1 すでに使っている要素

このプロジェクトは、すでにActive Inferenceの要素を「リミックス」しています：

| Active Inferenceの要素 | このプロジェクトでの対応 |
|:---|:---|
| 予測誤差の最小化 | Coherence Signal（`distance(s, G(F(s)))`）の計算 |
| 隠れ状態の推論 | RSSM（h, z）の更新 |
| 不確実性 | RSSMのエントロピーまたはKL divergence |
| 情報獲得 | Priority = coherence × uncertainty |

しかし、**決定的に欠けているもの**があります：

### 3.2 欠けているもの：「なぜ予測誤差を減らすのか」の動機

Active Inferenceでは、**「驚きを最小化する」こと自体が生物の存在理由**として公理的に仮定されます。

> 「驚きが大きい状態 = 生存に不利な状態」という前提

しかし、このプロジェクトでは：

- Coherence Signalは「モード切替のスイッチ」であり、最小化すべき損失ではない
- 保留構造の原則：「破綻は創造的問題解決の契機」

つまり、**Active Inferenceの「驚き最小化」という前提は、保留構造の思想と矛盾する可能性があります。**

---

## 4. 提案：Active Inferenceの「リミックス」

Active Inferenceを**そのまま**使うのではなく、保留構造の原則に合うように**リミックス**します。

### 4.1 従来のActive Inference

```
行動選択 = argmin_a E[F | a]
         = argmin_a E[予測誤差 + 不確実性 | a]
```

→ 予測誤差と不確実性を**最小化**する行動を選ぶ

### 4.2 保留構造版のActive Inference（2026-02-16改訂）

2026年2月16日の議論に基づき、Priority計算の考え方を修正する。

**旧来の考え方（問題点）**:
> 行動選択 = argmax_a (coherence × uncertainty × valence)

この掛け算は、3つの軸の質的な違いを無視した過剰設計であった。

**新しい考え方：軸の提供と使用法の創発**

行動選択は、Agent Cが内的に学習する。

```
観測 = {coherence, uncertainty, ...}
記憶 = {valence, ...}

行動選択 = AgentC(観測, 記憶)
```

Agent Cは、観測（coherence, uncertainty）と記憶（valence）を統合し、どの情報に注意を払うべきかを**自分で学習する**。これにより、注意の配分方式そのものが創発の対象となる。

### 4.3 具体的な内発的動機の設計

以下の3つの内発的報酬を組み合わせます：

#### (1) 好奇心報酬（Epistemic Value）

```
R_curiosity = Uncertainty の減少量
            = H(z_t-1) - H(z_t)
```

「分からなかったことが分かるようになった」という感覚

#### (2) 有能感報酬（Competence）

```
R_competence = Coherence の改善量（注意を向けた結果）
             = coherence_prev - coherence_curr (if attended)
```

「破綻に向き合った結果、理解が深まった」という感覚

#### (3) 新奇性報酬（Novelty）

```
R_novelty = KL(posterior || prior)
```

「予想外のことが起きた」という感覚（ただし、大きすぎると不安）

### 4.4 総合的な内発的報酬

```
R_intrinsic = α × R_curiosity + β × R_competence + γ × R_novelty
```

この内発的報酬を使って、Valenceを更新します：

```
valence(t+1) = valence(t) + learning_rate × R_intrinsic
```

---

## 5. 実装への落とし込み

### 5.1 Valence更新則の改訂

現在の実装：

```python
delta_coherence = coherence_prev - coherence_curr
valence(t+1) = (1-β) × valence(t) + β × delta_coherence × attention_weight
```

改訂版（内発的報酬を使用）：

```python
# (1) 好奇心報酬
R_curiosity = uncertainty_prev - uncertainty_curr

# (2) 有能感報酬
R_competence = (coherence_prev - coherence_curr) × attention_weight

# (3) 新奇性報酬
R_novelty = KL(posterior || prior)

# 総合的な内発的報酬
R_intrinsic = alpha × R_curiosity + beta × R_competence + gamma × R_novelty

# Valence更新
valence(t+1) = (1-decay) × valence(t) + decay × R_intrinsic
```

### 5.2 重みパラメータの設定

| パラメータ | 推奨値 | 意味 |
|:---|:---|:---|
| α (curiosity) | 0.3 | 知らないことを知りたい |
| β (competence) | 0.5 | 自分の行動で世界を変えたい |
| γ (novelty) | 0.2 | 新しいことに出会いたい（ほどほどに） |

### 5.3 訓練ループへの統合

逐次的訓練ループ（train_sequential.py）で、内発的報酬を計算し、Valence更新に使用します。

これにより、Agent Cは：

1. 不確実性を減らす経験を積むと、Valenceが上昇
2. 破綻に向き合って理解が深まると、Valenceが上昇
3. 予想外の発見があると、Valenceが上昇（ほどほどに）

---

## 6. 理論的整合性の確認

### 6.1 保留構造の原則との整合性

| 保留構造の原則 | 内発的動機による実現 |
|:---|:---|
| 破綻は創造的問題解決の契機 | R_competence: 破綻に向き合った結果の改善を報酬化 |
| Coherenceは最小化すべき損失ではない | Coherenceそのものではなく、「改善量」を報酬化 |
| 余白が先にある | 好奇心・新奇性報酬が、未知への探索を促す |
| 志向性 | 内発的報酬が「何に向かうべきか」を与える |

### 6.2 Active Inferenceとの違い

| 項目 | Active Inference | このプロジェクト（リミックス版） |
|:---|:---|:---|
| 目的 | 驚きの最小化 | 創造的余地の最大化 |
| Coherence | 最小化すべき | 向き合うべき契機 |
| Uncertainty | 最小化すべき | 探索すべき領域 |
| 行動選択 | 予測誤差を減らす | 破綻×不確実性×valenceを最大化 |

---

## 7. まとめ

**Active Inferenceは「使える」が、「そのまま」ではない**

- Active Inferenceの「内発的動機」の定式化（好奇心、有能感、新奇性）は有用
- しかし、「驚き最小化」という前提は、保留構造の「破綻は契機」という思想と矛盾
- 解決策：内発的報酬を「改善量」として定式化し、Valence更新に使用

**次のステップ**

1. Valence更新則を内発的報酬ベースに改訂
2. 逐次的訓練ループで内発的報酬を計算
3. Agent Cが「目的を持って」経験を蓄積することを検証

これにより、Agent Cは「何のために状態を保持するのか」を理解できるようになります。
