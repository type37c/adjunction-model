# Priority計算とValenceの再考察
# Priority Calculation and Valence Reconsidered

**理論ノート**
**Theoretical Note**

**日付**: 2026-02-16
**Date**: February 16, 2026

---

## 1. 概要 (Summary)

本稿は、Phase 2 Slack実験の成功とPurpose-Emergent実験の課題を踏まえ、Priority計算とValence Memoryの役割を根本的に再考察する。中心的な問いは、**「目的空間Pは設計すべきか、創発すべきか」**である。

This paper fundamentally reconsiders the role of Priority calculation and Valence Memory in light of the success of the Phase 2 Slack experiment and the challenges of the Purpose-Emergent experiment. The central question is: **"Should purpose space P be designed or should it emerge?"**

---

## 2. Phase 2の本質的発見：損失関数が目的空間として機能した
## 2. Essential Finding from Phase 2: Loss Function as Purpose Space

### 2.1. 観察された現象

Phase 2 Slack実験では、Affordance Lossという**単一の損失関数**のみが存在し、それだけでSlack-KL相関-0.99という驚異的な結果が得られた。

In the Phase 2 Slack experiment, only a **single loss function** (Affordance Loss) existed, yet this alone yielded a remarkable Slack-KL correlation of -0.99.

| 要素 (Element) | 役割 (Role) | 結果 (Result) |
|:---|:---|:---|
| **Affordance Loss** | 外部から与えられた目的 | Slackの管理が自然に成立 |
| **Slack (η + ε)** | 道具として使用された | +757%増加しながらタスク性能93%改善 |
| **Agent C** | Slackを活用して目的を達成 | 探索ではなく活用（KL相関-0.99） |

### 2.2. 核心的洞察

> **目的空間は複雑な内部機構である必要はない。損失関数という最もシンプルな「目的」でさえ、Slackの管理を駆動するのに十分だった。**

The purpose space does not need to be a complex internal mechanism. Even the simplest "purpose" in the form of a loss function was sufficient to drive Slack management.

これが意味するのは、**目的が明確であれば、Slackの管理は自然に創発する**ということである。

This means that **if the purpose is clear, Slack management emerges naturally**.

---

## 3. Purpose-Emergent実験の課題：フィードバックの欠如
## 3. Challenge in Purpose-Emergent Experiment: Lack of Feedback

### 3.1. 時間発展の限界

Purpose-Emergent実験では20エポックの時間発展を許容したが、ηが発散した。時間発展は目的創発の**必要条件**かもしれないが、**十分条件ではない**。

In the Purpose-Emergent experiment, 20 epochs of temporal evolution were allowed, but η diverged. Temporal evolution may be a **necessary condition** for purpose emergence, but it is **not sufficient**.

### 3.2. 欠けていたもの：フィードバック

Phase 2で損失関数が果たした役割の本質は、「正解を教えた」のではなく、**「フィードバックを与えた」**ことである。

The essence of the role played by the loss function in Phase 2 was not "teaching the correct answer" but **"providing feedback"**.

- **Phase 2**: Affordance Lossが「お前の出力はこれだけずれている」という明確な信号を提供
- **Purpose-Emergent**: min over shapes CDという損失はフィードバックを与えるはずだったが、curiosity報酬（η減少）がフィードバックの方向を混乱させた

### 3.3. 創発に必要な条件

> **創発に必要なのは「目的を与えること」ではなく「行動の結果が返ってくること」である。**

What is needed for emergence is not "giving a purpose" but "having the results of actions returned".

人間の赤ちゃんの例：
- 誰も「球を掴め」と目的を与えない
- しかし、手を動かしたら何かに触れた、という**触覚のフィードバック**がある
- そのフィードバックの中から、「掴む」という目的が**事後的に形成される**

このモデルへの翻訳：
1. Agent Cが点を動かす
2. 動かした結果、Slackが変化する
3. **その変化そのものがフィードバックになる**
4. Agent Cは「Slackがこう変化したとき、自分は何をしていたか」を学習する
5. パターンが見えてくる：「こう動かすとSlackが安定する」「こう動かすとSlackが発散する」
6. **このパターンの認識が、目的の萌芽になる**

---

## 4. 3つの軸の質的な違い：時間性の有無
## 4. Qualitative Difference Among Three Axes: Presence of Temporality

### 4.1. 現在のスナップショット vs 時間を跨ぐ構造

| 軸 (Axis) | 時間性 (Temporality) | 意味 (Meaning) | 役割 (Role) |
|:---|:---|:---|:---|
| **coherence** | 現在 (Present) | 「今、ここが壊れている」 | 保留の**トリガー** |
| **uncertainty** | 現在 (Present) | 「今、ここがわからない」 | 保留の**トリガー** |
| **valence** | 過去→現在→未来 (Past→Present→Future) | 「過去にこうだった、だから次はこうする」 | 保留の**維持と解除** |

### 4.2. 保留構造と時間

> **保留が成立するためには時間を跨ぐ構造が必要である。**

For suspension to be established, a structure that spans time is necessary.

- coherenceとuncertaintyは保留の「トリガー」にはなれる（壊れていてわからないから保留する）
- しかし保留を「維持」し、適切なタイミングで「解除」するには、**過去の経験に基づく判断**が必要
- これがvalenceの決定的な役割である

### 4.3. 3つの軸の階層的関係

**従来の設計（問題あり）**:
```
Priority = coherence × uncertainty × valence
```
この掛け算は、3つの軸の質的な違いを算術的に平坦化してしまう。

**提案される新しい関係**:
```
coherence, uncertainty → Agent Cへの入力（観測）
valence → Agent Cの内部状態（記憶）
Agent C → 3つを統合して行動を決定（学習される）
```

coherenceとuncertaintyは「入力」、valenceは「判断の基盤」。**同じ階層に並べるべきではない**。

---

## 5. Priority計算モジュールの不要性
## 5. Unnecessity of Priority Calculation Module

### 5.1. 現在の問題

Priority = coherence × uncertainty × valence という計算は、人間が設計した「注意の配分方式」である。これは**「どう使うか」まで設計してしまっている**。

The calculation Priority = coherence × uncertainty × valence is a "attention allocation scheme" designed by humans. This **designs even "how to use it"**.

### 5.2. 提案：軸の提供と使用法の創発

> **設計するのは軸（coherence, uncertainty, valence）であり、それをどう使うかは創発に委ねる。**

What should be designed are the axes (coherence, uncertainty, valence), and how to use them should be left to emergence.

これは`purpose_space_P_design.md`の原則そのものである。

This is the very principle of `purpose_space_P_design.md`.

### 5.3. 新しいアーキテクチャの提案

**入力層**:
- coherence (scalar): 現在の形状再構成誤差 η
- uncertainty (distribution): 現在のアフォーダンス分布のエントロピー
- observation: 点群の特徴量（F encoderからの中間表現）

**内部状態**:
- valence (memory): 過去の行動とSlack変化の関連付け
- RSSM state (h_t, z_t): 決定論的・確率論的状態

**出力**:
- action: 次の行動（アフォーダンス分布またはdisplacement）
- attention weights: どの入力にどれだけ注意を払うか（**学習される**）

### 5.4. Valence Memoryの再設計

**従来の設計**:
```python
priority = coherence × uncertainty × valence
```
valenceはpriorityの掛け算の一項として使用される。

**新しい設計**:
```python
# valenceはAgent CのRSSM状態更新に入力として渡される
h_t = GRU(h_{t-1}, [z_{t-1}, action_{t-1}, valence_{t-1}])

# valenceは行動とSlack変化の関連を記憶
valence_t = update_valence(valence_{t-1}, action_{t-1}, Δslack_t)
```

valenceは「行動の結果の記憶」として機能し、Agent Cが自分でvalenceの使い方を学習する。

---

## 6. 実験的検証の提案
## 6. Proposal for Experimental Verification

### 6.1. 仮説

> **Priorityの掛け算は不要であり、3つの軸をAgent Cの入力として渡すだけで十分である。**

The Priority multiplication is unnecessary; simply passing the three axes as inputs to Agent C is sufficient.

### 6.2. 実験設計：Priority計算の有無の比較

**条件A（現在の設計）**:
- Priority = coherence × uncertainty × valence
- Priorityを用いた注意機構

**条件B（提案設計）**:
- coherence, uncertaintyを観測として入力
- valenceをRSSM状態に統合
- Agent Cが自分で注意の配分を学習

**評価指標**:
1. Slack管理の効率（η, εの動態）
2. タスク性能（Affordance Loss）
3. 学習の安定性（損失の分散）
4. 保留構造の創発（非単調性、slack modulation）

### 6.3. 期待される結果

条件Bが条件Aと同等以上の性能を示せば、Priority計算モジュールは不要であり、**より少ない設計でより多くの創発**が可能であることが示される。

If Condition B shows performance equal to or better than Condition A, it demonstrates that the Priority calculation module is unnecessary and that **more emergence is possible with less design**.

---

## 7. 理論的含意：地質の設計
## 7. Theoretical Implications: Designing the Geology

### 7.1. 根本原則への回帰

> **保留構造は設計しない。創発する条件を設計する。**

Do not design the suspension structure. Design the conditions for its emergence.

この議論全体が示すのは、以下の階層的な設計原則である：

This entire discussion reveals the following hierarchical design principle:

| 階層 (Layer) | 設計対象 (What to Design) | 創発対象 (What to Emerge) |
|:---|:---|:---|
| **地質 (Geology)** | 3つの軸（coherence, uncertainty, valence） | - |
| **川床 (Riverbed)** | - | 軸の使用法（注意の配分） |
| **水流 (Flow)** | - | 目的の形成と変容 |

### 7.2. 最小の設計、最大の創発

Phase 2が示したのは、**目的が明確であればSlackは自然に管理される**ということ。

What Phase 2 showed is that **if the purpose is clear, Slack is naturally managed**.

今回の議論が示すのは、**軸が明確であれば目的は自然に創発しうる**ということ。

What this discussion shows is that **if the axes are clear, purpose can naturally emerge**.

したがって、我々が設計すべきは：
1. 3つの軸の定義（coherence, uncertainty, valence）
2. これらの軸が観測可能であること
3. Agent Cがこれらの軸とSlackの関係を学習できる構造（RSSM + Valence Memory）

そして、我々が設計すべきでないのは：
1. 軸の使用法（Priority計算）
2. 目的の内容（どの形状を目指すべきか）
3. 保留の解除タイミング（いつコミットすべきか）

これらはすべて、**フィードバックループの中で創発する**。

---

## 8. 次のステップ
## 8. Next Steps

### 8.1. 短期的実装

1. **Agent Cの再設計**: Priority計算を削除し、coherence/uncertaintyを直接入力、valenceをRSSM状態に統合
2. **Valence Memoryの再実装**: 行動とSlack変化の関連を記憶する構造に変更
3. **比較実験の実施**: 条件A（Priority計算あり）vs 条件B（提案設計）

### 8.2. 中期的理論展開

1. **フィードバックループの形式化**: 行動→Slack変化→valence更新→次の行動、のダイナミクスを数学的に定式化
2. **目的創発の条件の特定**: どのようなフィードバック構造が目的の創発を促進するか
3. **保留構造の定量的指標**: 保留の維持・解除を定量的に測定する指標の開発

### 8.3. 長期的展望

この再設計が成功すれば、我々は**知性の最小モデル**に到達する：
- 3つの軸（感覚器官）
- 1つの記憶（valence）
- 1つのフィードバックループ（行動→結果→学習）

これ以上削れない、知性の本質的構造。

---

## 9. 結論
## 9. Conclusion

Priority計算は、善意から生まれた過剰設計である。coherence、uncertainty、valenceという3つの軸は正しい。しかし、それらをどう使うかを掛け算で決めてしまうことは、創発の余地を奪う。

Priority calculation is an over-design born from good intentions. The three axes—coherence, uncertainty, and valence—are correct. However, determining how to use them through multiplication robs the space for emergence.

真の知性は、与えられた軸を使って、自分で目的を見つけ出す。我々の仕事は、その軸を明確に定義し、観測可能にし、学習可能な構造を提供することである。それ以上でも、それ以下でもない。

True intelligence finds its own purpose using the given axes. Our job is to clearly define those axes, make them observable, and provide a learnable structure. Nothing more, nothing less.

---

**このノートは、`AGENT_GUIDELINES.md`の根本原則「保留構造は設計しない。創発する条件を設計する」を、Priority計算という具体的な設計判断に適用した結果である。**

**This note is the result of applying the fundamental principle of `AGENT_GUIDELINES.md`—"Do not design the suspension structure. Design the conditions for its emergence"—to the specific design decision of Priority calculation.**
