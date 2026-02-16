# 目的空間Pの形式化 / Formalization of Purpose Space P

**Design Document — v1.0 (2026-02-13)**

---

## 1. 動機と背景 / Motivation and Background

### 1.1 なぜPが必要か / Why P is Necessary

2月13日の議論と実験結果から、以下の問題が明らかになった。

1. **Agent Cに「自分を保つ」根拠がない**: 現在のAgent Cは、coherenceとuncertaintyから計算されるPriorityに基づいて注意を配分するが、「なぜ注意を配分するのか」「何のために状態を保持するのか」という目的を持たない。逐次的訓練ループを実装しても、目的なしに状態を更新すれば、単にcoherenceを最小化する方向に崩壊する危険がある。

2. **言語接地にPが必要**: 言語は随伴の要件を満たさない（多対多の写像、身体からの切り離し）。言語を接地するには、身体的随伴（F ⊣ G）と言語の間に、目的空間Pという中間層が必要である。

3. **Priority逆転（0.17x）がPの萌芽を示している**: 「前の経験が次の構えを決める」という現象は、Agent Cの内部状態に蓄積された経験が、原始的な「目的」として機能していることを示唆する。

### 1.2 Pの原則 / Principles of P

2月13日の議論で確立された原則を厳守する。

> **Pは「完全に創発」でもなく、「完全に設計」でもない。**

| 設計するもの (Designed) | 創発するもの (Emergent) |
|:---|:---|
| 価値判断の**軸** | その軸の上で**何が良いか**の具体的判断 |
| 「破綻＝注意すべき」という枠組み | 「どの程度の破綻なら許容できるか」 |
| 「不確実＝探索すべき」という枠組み | 「どの種類の不確実性が重要か」 |
| Pの**表現空間**（次元、構造） | Pの**内容**（具体的な値の分布） |

---

## 2. 理論的定式化 / Theoretical Formulation

### 2.1 Pの定義 / Definition of P

目的空間Pは、Agent Cの内部状態の中に存在する**価値判断の潜在構造**である。

> **P(t) ⊂ C(t) = (h_t, z_t)**

ここで、C(t)はAgent Cの内部状態全体であり、P(t)はその中で「何に向かうべきか」を表現する部分構造である。Pは独立したモジュールではなく、C(t)の潜在空間の中で**構造として創発する**。

ただし、「創発する」とは「何もしなくても勝手に現れる」という意味ではない。我々は、Pが創発するための**条件**を設計する。

### 2.2 Pの3つの軸 / Three Axes of P

Pの価値判断の枠組みとして、以下の3つの軸を設計する。

**軸1: Coherence（整合性）** — 「壊れているものは注意すべき」

これは既に実装済みの空間的coherence signalに対応する。coherence_i が高い点は、現在の随伴構造が破綻している箇所であり、修復の候補となる。

**軸2: Uncertainty（不確実性）** — 「分からないものは探索すべき」

これも既に実装済みのRSSMのエントロピーに対応する。uncertainty が高い状態は、Agent Cの信念が不確かであり、新しい情報を獲得する価値がある。

**軸3: Valence（価値性）** — 「経験から学んだ良し悪し」

これが**新たに追加する軸**である。CoherenceとUncertaintyは「今この瞬間」の状態から計算される即時的な指標だが、Valenceは**過去の経験の蓄積**から形成される持続的な指標である。

> **Valence(t) = f(history of coherence changes)**

具体的には、「ある状態に注意を向けた結果、coherenceが改善したか悪化したか」の履歴を蓄積し、「この種の破綻に向かうと良い結果になりやすい」という経験的な価値判断を形成する。

### 2.3 3つの軸の役割の再定義 / Redefining the Roles of the Three Axes

2026年2月16日の議論に基づき、Priority計算の原理を根本的に見直す。

**以前の考え方（問題点）**:
> priority_i = coherence_i × uncertainty_i × valence_i

この掛け算によるPriority計算は、3つの軸の質的な違い（特にvalenceの時間性）を算術的に平坦化し、「どう使うか」まで人間が設計してしまう**過剰設計**であった。

**新しい原理：軸の提供と使用法の創発**

> **設計するのは軸（coherence, uncertainty, valence）であり、それをどう使うかは創発に委ねる。**

この原理に基づき、3つの軸はAgent Cに対して以下のように提供されるべきである。

| 軸 (Axis) | 時間性 (Temporality) | 提供方法 (Provision Method) | 役割 (Role) |
|:---|:---|:---|:---|
| **coherence** | 現在 (Present) | 観測入力 (Observation Input) | 「今、ここが壊れている」という現状報告 |
| **uncertainty** | 現在 (Present) | 観測入力 (Observation Input) | 「今、ここがわからない」という現状報告 |
| **valence** | 過去→未来 (Past→Future) | 内部状態/記憶 (Internal State/Memory) | 「過去の経験に基づき、どう構えるか」という判断基盤 |

この新しい原理では、**Priority計算モジュールは不要になる**。Agent Cは、coherenceとuncertaintyを観測として受け取り、内部状態であるvalence（経験の蓄積）と統合して、どの情報に注意を払うべきかを**自分で学習する**。これにより、注意の配分方式そのものが創発の対象となる。

### 2.4 Pの更新則 / Update Rule for P

Pの更新は、Agent Cの状態更新の中に組み込まれる。

```
C(t+1) = update(C(t), action_result, coherence_signal)
```

この更新の中で、Pに相当する部分は以下のように更新される：

```
valence(t+1) = valence(t) + α × Δcoherence × attention_weight(t)
```

ここで：
- `Δcoherence = coherence(t) - coherence(t+1)` : coherenceの変化量（改善なら正）
- `attention_weight(t)` : 時刻tでその点に向けた注意の量
- `α` : 学習率（設計パラメータ）

この更新則は、「注意を向けた結果、coherenceが改善した」場合にvalenceが上昇し、「注意を向けたが改善しなかった」場合にvalenceが低下することを意味する。

---

## 3. アーキテクチャへの統合 / Integration into Architecture

### 3.1 Agent C v3の設計 / Agent C v3 Design

新しい原理に基づき、Agent C v3を再設計する。この設計では、Priority計算モジュールは廃止される。

```
Agent C v3 = RSSM + Valence Memory
```

#### 構成要素

| コンポーネント | 役割 | 入力 | 出力 |
|:---|:---|:---|:---|
| RSSM | 内部状態の維持・統合 | 前の状態、行動、**観測(coherence, uncertainty)**, **valence** | 新しい状態 (h, z) |
| **Valence Memory** | 行動結果の記憶 | 行動、Slack変化 (Δη) | valence (記憶) |
| Context Generator | F/Gへの文脈提供 | h, z | context vector |

**変更の要点**:
- **Priority Moduleの廃止**: `coherence`と`uncertainty`は、もはやPriority計算に使われるのではなく、RSSMへの直接の**観測入力**となる。
- **Valence Memoryの役割変更**: `valence`は「経験的価値」ではなく、「行動と結果の記憶」として機能し、RSSMの状態更新に直接影響を与える。
- **Agent Cの自己学習**: Agent Cは、観測（coherence, uncertainty）と記憶（valence）を統合し、どの情報に注意を払うべきかを**内的に学習する**。

### 3.2 Valence Memoryの実装方針 / Valence Memory Implementation

Valence Memoryは、Agent Cの潜在状態の中で経験的な価値判断を蓄積する機構である。

**設計選択肢の検討:**

**案A: 明示的なValenceベクトル**
- Agent Cの状態に `v_t ∈ R^d_v` を追加
- 各ステップで `v(t+1) = (1-β)v(t) + β × valence_update` として更新
- 利点: 解釈可能、デバッグしやすい
- 欠点: Pの「創発」ではなく「設計」に近い

**案B: RSSMの潜在状態への統合（推奨）**
- Valenceの情報をRSSMへの入力に含め、h_tとz_tの中にValenceが自然に符号化されるようにする
- Δcoherenceとattention historyをRSSMの観測として提供
- 利点: Pが潜在状態の中で創発する、プロジェクトの思想に忠実
- 欠点: 解釈が難しい、Valenceがどこに符号化されたか不明確

**案C: ハイブリッド（採用）**
- 明示的なValenceベクトル `v_t` を持つが、これはRSSMの入力にも使われる
- `v_t` は「設計された枠組み」（軸）であり、その中身（具体的な値）は経験から更新される
- RSSMの潜在状態の中で、`v_t` を超えた暗黙的な価値判断も創発しうる
- 利点: 設計と創発のバランス、解釈可能性と柔軟性の両立

### 3.3 Context Generatorの拡張 / Extended Context Generator

現在のContext Generatorは `[h, z, obs_attended]` からcontextを生成している。これを拡張する。

```
context = ContextNet([h, z, obs_attended, v])
```

Valenceベクトル `v` がcontextに含まれることで、FiLMを通じてF/Gの振る舞いがValenceに依存するようになる。これにより、「過去に良い結果が出た種類の破綻」に対しては、F/Gが異なる応答を返すようになる。

### 3.4 3層構造との対応 / Correspondence to 3-Layer Structure

| 層 | 内容 | 実装 |
|:---|:---|:---|
| Layer 1: 身体的随伴 (F ⊣ G) | 形状⇄アフォーダンス | FunctorF, FunctorG (既存) |
| Layer 2: 目的空間 (P) | 経験から創発する価値判断 | **Valence Memory + RSSM潜在状態** |
| Layer 3: 言語 (L) | Pの構造に名前をつけたもの | 将来実装 |

---

## 4. 実装計画 / Implementation Plan

### 4.1 ファイル構成 / File Structure

```
src/models/
├── agent_layer.py          # RSSM (既存、変更なし)
├── agent_layer_v2.py       # Agent C v2 (既存、変更なし)
├── agent_layer_v3.py       # Agent C v3 (新規: Valence Memory追加)
├── priority.py             # Priority計算 (既存、v3では廃止予定)
├── valence.py              # Valence Memory (新規: 行動とSlack変化の記憶)
├── conditional_adjunction_v2.py  # (既存、変更なし)
├── conditional_adjunction_v3.py  # (新規: Agent C v3統合)
```

### 4.2 実装ステップ / Implementation Steps

1. **valence.py**: ValenceMemoryモジュールの再実装
   - 行動とSlack変化(Δη)の関連を記憶する構造
   - valenceはRSSMの状態更新に使われる内部記憶として管理

2. **agent_layer_v3.py**: Agent C v3の実装
   - **Priority計算を削除**
   - `coherence`と`uncertainty`をRSSMへの観測入力として追加
   - `valence`をRSSMの状態更新に統合

3. **conditional_adjunction_v3.py**: 統合モデル
   - Agent C v3をF/Gと統合

4. **比較実験の実施**: 以下の2つの条件を比較する実験を設計・実施
   - **条件A（旧設計）**: Priority計算を用いたAgent C v2
   - **条件B（新設計）**: Priority計算を廃止したAgent C v3

### 4.3 検証方法 / Verification

1. **Valenceの蓄積テスト**: 同じ形状を繰り返し提示した際、valenceが変化するか
2. **注意配分の創発テスト**: Priority計算なしで、Agent Cが自己組織的に注意の配分パターンを学習するか
3. **RSSM状態の分析**: valenceが入力されたRSSMの潜在状態が、意味のある構造を獲得するか
4. **逐次的訓練テスト**: 複数形状の連続提示で、Agent Cの状態が意味のある蓄積を示すか
5. **比較実験**: 条件A（Priority計算あり）vs 条件B（Priority計算なし）でSlack管理の効率とタスク性能を比較

---

## 5. 理論的考察 / Theoretical Considerations

### 5.1 Pと保留構造の関係 / P and Suspension Structure

Pは保留構造の一部であり、保留構造の4要件との関係は以下の通り。

| 保留構造の要件 | Pとの関係 |
|:---|:---|
| 差異への感受性 | Coherence軸がこれを担う |
| 志向性 | **Pの中核機能**: Valenceが「何に向かうべきか」を決定する |
| 時間的持続 | Valence Memoryが経験を蓄積し、持続させる |
| 自己と非自己の区別 | Valenceは「自分の行動の結果」を記録するため、自他の区別を強化する |

### 5.2 Coherence Signalとの関係 / Relationship with Coherence Signal

**重要**: Valenceは「coherenceを最小化する」ための道具ではない。

Valenceが蓄積するのは「coherenceの変化の方向」であり、「coherenceが低い状態が良い」という判断ではない。Agent Cは、coherenceが高い状態（破綻）を経験し、それに向かった結果としてcoherenceが変化する（改善も悪化もある）という経験を蓄積する。

この設計により、「破綻を避ける」のではなく、「破綻に対してどう向き合うか」の経験的知恵が蓄積される。これは保留構造の原則——「破綻は創造的問題解決の契機」——と整合する。

### 5.3 飽和（Saturation）との統合 / Integration with Saturation

Valenceは飽和問題にも自然に対処する。同じ形状を繰り返し見ると：
- Coherenceは低下する（慣れ）
- Uncertaintyも低下する（予測精度の向上）
- Valenceは飽和する（同じ結果の繰り返しで更新量が減少）

3つの軸すべてが低下するため、Agent Cが学習した注意配分は自然にその形状から離れる。これが「退屈」の実装であり、新しい刺激への内発的動機づけとなる。

---

## 6. 未解決の問題 / Open Questions

1. **Valenceの次元**: `v_t` の次元をいくつにすべきか。高次元にすると表現力は上がるが、創発の余地が減る。低次元にすると制約が強すぎる可能性がある。→ まずは小さく始め（d_v = 16 or 32）、実験で調整する。

2. **Valenceの減衰**: 古い経験の影響をどう減衰させるか。指数減衰 `(1-β)` を使うが、βの値は実験で決定する。

3. **Valenceの空間的構造**: 現在のvalenceはバッチ単位（サンプル単位）だが、点単位のvalenceが必要か。→ まずはバッチ単位で始め、必要に応じて拡張する。

4. **負のValence**: 「向かって悪化した」経験をどう扱うか。単純に負のvalenceとするか、別の機構（回避）として扱うか。→ 連続値として扱い、負の値も許容する。Agent Cが負のvalenceをどう使うかは、Agent C自身が学習する。

---

## Appendix: 地質・川床・水流の比喩との対応 / Correspondence to Geology Metaphor

| 比喩 | 既存の対応 | Pの導入後 |
|:---|:---|:---|
| 地質（不変の原理） | 保留構造の条件 | 保留構造の条件 + **Pの枠組み（3つの軸）** |
| 川床（可変の構造） | 随伴構造 F ⊣ G | 随伴構造 F ⊣ G + **Pの内容（valenceの具体的な値）** |
| 水流（行動・経験） | coherence signal, 訓練 | coherence signal, 訓練, **valenceの更新** |

Pの枠組み（3つの軸）は地質に属する——これは設計するものであり、変わらない。
Pの内容（valenceの具体的な値）は川床に属する——これは経験によって変化し続ける。
