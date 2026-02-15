# 随伴の非対称性と目的のフィルター
# The Asymmetry of Adjunction and the Purpose Filter

**理論ノート**
**Theoretical Note**

**日付**: 2026-02-15
**Date**: February 15, 2026

---

## 1. 概要 (Summary)

本稿は、我々の随伴モデル (Adjunction Model) の中核をなす理論的洞察、すなわち**随伴の本質的な非対称性**と、それが**目的のフィルター**として機能するメカニズムについて詳述する。この洞察は、`research_notes_consolidated.md` v6.0で簡潔に触れられているが、ここではその理論的背景と含意をより深く掘り下げる。

This paper details the core theoretical insight of our Adjunction Model: the **inherent asymmetry of adjunction** and the mechanism by which it functions as a **purpose filter**. While briefly mentioned in `research_notes_consolidated.md` v6.0, this note delves deeper into its theoretical background and implications.

---

## 2. 随伴の再定義：状態Cへの依存性
## 2. Redefining Adjunction: Dependency on State C

標準的な圏論の随伴 `F ⊣ G` は、エージェントと世界の相互作用を記述するには不十分である。なぜなら、エージェントの行動は、その内部状態**C**（例：疲労、意図、身体的制約）に決定的に依存するからだ。この依存性を取り込むことで、随伴構造に**本質的な非対称性**が明らかになる。

The standard categorical adjunction `F ⊣ G` is insufficient to describe agent-world interaction because an agent's actions are critically dependent on its internal state **C** (e.g., fatigue, intent, physical constraints). Incorporating this dependency reveals an **inherent asymmetry** in the adjunction structure.

| 要素 (Element) | 属性 (Attribute) | 説明 (Description) |
| :--- | :--- | :--- |
| **形状 (Shape)** | 環境側の不変量 (Environment-side Invariant) | エージェントの状態Cに**依存しない**。スーツケースは、エージェントが疲れていようがいまいが、スーツケースのままである。 |
| **行動 (Action)** | エージェント側の可変量 (Agent-side Variable) | エージェントの状態Cに**依存する**。元気なエージェントはスーツケースを「片手で持ち上げる」が、腕を怪我したエージェントは「転がす」を選択する。 |

この非対称性に基づき、関手FとGは以下のように再定義される。

Based on this asymmetry, the functors F and G are redefined as follows:

-   **関手 F(Shape, C) → Action**: ある形状と**現在の自己状態C**に基づき、可能な行動（アフォーダンス）を決定する。
-   **関手 G(Action, C) → Shape_Aspects**: ある行動を**現在の自己状態C**で実現するために、形状の**どの側面(Aspects)に注目すべきか**を選択する。

この再定義の核心は、**Gの役割**にある。Gは形状をゼロから「生成」するのではなく、無限の機能的側面を内包する多面体としての既存の形状から、現在の目的に関連する部分（例：「持ち手」「車輪」「容量」）を**選択(select)**または**注目(attend to)**する。したがって、この随伴構造全体が、エージェントの現在の状態Cという**「目的のフィルター」**を通して世界を解釈するプロセスそのものをモデル化しているのである。

The core of this redefinition lies in the **role of G**. G does not "generate" a shape from scratch, but rather **selects** or **attends to** the relevant aspects (e.g., "handle," "wheels," "capacity") from an existing shape, which is seen as a polyhedron with infinite functional facets. Therefore, the entire adjunction structure models the very process of interpreting the world through the **"purpose filter"** of the agent's current state C.

---

## 3. Slack：非対称性の定量化
## 3. Slack: Quantifying the Asymmetry

**Slack**とは、この目的のフィルターを通して世界と相互作用する際に必然的に生じる「ズレ」や「遊び」の定量化である。フィルターが存在しなければ、Slackは定義すらできない。

**Slack** is the quantification of the "gaps" or "play" that inevitably arise when interacting with the world through this purpose filter. Without the filter, Slack cannot even be defined.

-   **η (Unit Slack)**: `||Shape - G(F(Shape, C), C)||²`
    -   形状全体と、エージェントが注目した側面との差分。これは、現在の目的にとって**無関係であると判断された形状情報**の量であり、**知覚的余白 (perceptual margin)** と解釈できる。
    -   The difference between the entire shape and the aspects the agent attended to. This is the amount of shape information **deemed irrelevant** for the current purpose, interpretable as the **perceptual margin**.

-   **ε (Counit Slack)**: `||Action - F(G(Action, C), C)||²`
    -   当初意図した行動と、形状の特定の側面に注目した結果として再解釈された行動との差分。これは、**目的が世界との接触を通じて変容する可能性**の量であり、**意味的余白 (semantic margin)** と解釈できる。
    -   The difference between the initially intended action and the re-interpreted action resulting from attending to specific aspects of the shape. This is the amount of **potential for the purpose to be transformed through contact with the world**, interpretable as the **semantic margin**.

この構造的非対称性は、Slackの分布に決定的な影響を与える。知覚的余白(η)は物理的形状に制約されるため、その「遊び」には上限がある。一方、意味的余白(ε)は解釈の問題であるため、その「遊び」はエージェントの経験や創造性に応じて際限なく広がりうる。この理論的予測は、Phase 2 Slack実験において、εがηに比べて爆発的に増大（+417,837% vs +14%）した結果によって強力に裏付けられている。

This structural asymmetry has a decisive impact on the distribution of Slack. The perceptual margin (η) is constrained by physical shape, so its "play" has an upper bound. In contrast, the semantic margin (ε) is a matter of interpretation, so its "play" can expand indefinitely depending on the agent's experience and creativity. This theoretical prediction is strongly supported by the results of the Phase 2 Slack experiment, where ε increased explosively compared to η (+417,837% vs +14%).

---

## 4. 結論：創造性の源泉としての意味的余白
## 4. Conclusion: Semantic Margin as the Source of Creativity

エージェントの適応性と創造性の源泉は、知覚の精度（ηをゼロに近づけること）ではなく、**意味の再解釈能力（εを管理すること）**にある。Phase 2実験で観測された「有能感駆動行動」とは、エージェントがこの広大な意味的余白を利用して、「同じ形状に対して、より効率的な目的の解釈を見つけ出す」プロセスに他ならない。

The source of an agent's adaptability and creativity lies not in the precision of its perception (minimizing η), but in its **ability to reinterpret meaning (managing ε)**. The "competence-driven behavior" observed in the Phase 2 experiment is nothing other than the process of the agent utilizing this vast semantic margin to "find more efficient interpretations of purpose for the same shape."

したがって、随伴の非対称性という概念は、単なる数学的な定式化に留まらず、知性がどのようにして世界と関わり、意味を見出し、そして自らの行動を洗練させていくのかについての、根源的な説明を提供するものである。

Therefore, the concept of adjunction asymmetry is not merely a mathematical formalization; it provides a fundamental explanation of how intelligence engages with the world, finds meaning, and refines its own actions.
