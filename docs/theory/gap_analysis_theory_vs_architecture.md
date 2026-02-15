# 理論と実装のズレに関する分析
# Gap Analysis: Theory vs. Architecture

**分析ノート**
**Analysis Note**

**日付**: 2026-02-15
**Date**: February 15, 2026

---

## 1. 目的 (Objective)

本稿は、最新の理論的洞察（`adjunction_asymmetry_and_purpose_filter.md`）と、現在の実装（`ARCHITECTURE.md`）との間に存在する**争点とズレ**を明確にすることを目的とする。

This document aims to clarify the **points of contention and discrepancies** between the latest theoretical insights (`adjunction_asymmetry_and_purpose_filter.md`) and the current implementation (`ARCHITECTURE.md`).

---

## 2. 分析サマリー (Analysis Summary)

理論と実装の間に、**1つの根本的な争点**と、それに起因する**2つの具体的なズレ**が存在する。

There is **one fundamental point of contention** and **two specific discrepancies** arising from it between the theory and the implementation.

| 項目 (Item) | 理論 (Theory) | 実装 (Implementation) | ズレ (Gap) |
| :--- | :--- | :--- | :--- |
| **根本的な争点** | **随伴の動的な変調**<br>FとGは状態Cによって動的に変調されるべき | **随伴の静的なマッピング**<br>FとGは状態Cから独立した静的な関数である | **最大** |
| **具体的なズレ 1** | **`F(Shape, C)`**<br>状態Cがアフォーダンスの解釈を変える | **`F(Shape)`**<br>状態CはFの計算に関与しない | **大** |
| **具体的なズレ 2** | **`G(Action, C)`**<br>状態Cに応じて形状の「側面」を選択する | **`G(Action)`**<br>状態Cに関係なく、常に形状全体を「生成」する | **大** |

**結論として、現在のアーキテクチャは、理論が要求する「目的のフィルター」を随伴の内部に実装しておらず、エージェント層Cが外部からその役割を代替する、より疎結合な構造になっている。**

**In conclusion, the current architecture does not implement the "purpose filter" required by the theory *inside* the adjunction. Instead, it has a more loosely coupled structure where the agent layer C substitutes for that role from the outside.**

---

## 3. 詳細な分析 (Detailed Analysis)

### 3.1. 争点：随伴は「静的」か「動的」か (Contention: Is the Adjunction Static or Dynamic?)

-   **理論の主張**: 最新の理論では、随伴 `F ⊣ G` はエージェントの内部状態**C**に依存する**動的なプロセス**として定義される。`F(Shape, C)` と `G(Action, C)` は、Cという「目的のフィルター」を通して世界を解釈するメカニズムそのものである。

-   **実装の現状**: `ARCHITECTURE.md`および`src/models/`以下のコードを見ると、`FunctorF` (PointNet++) と `FunctorG` (FoldingNet) は、状態Cを入力として受け取らない。これらはPhase 1で事前学習された後、Phase 3では凍結される**静的な関数**である。`F`は形状のみを入力とし、`G`はアフォーダンス分布のみを入力とする。

-   **ズレの核心**: 理論が「FとG自体がCによって変形する」ことを要求しているのに対し、実装は「FとGは不変の知識ベースであり、Cはそれを利用するエージェントである」という立場を取っている。これは、**目的のフィルターが随伴の内部にあるべきか、外部にあるべきか**という根本的な設計思想の違いである。

### 3.2. 具体的なズレ (Specific Discrepancies)

#### 1. Functor F: 状態Cの不在 (Absence of State C in Functor F)

-   **理論**: `F(Shape, C)` は、同じ形状（スーツケース）でも、エージェントの状態（疲れているか）によって異なるアフォーダンス（「転がせる」）を出力することを要求する。
-   **実装**: `FunctorF.forward(pos, batch)` は、形状`pos`しか見ない。したがって、現在の実装では、同じ形状に対しては常に同じアフォーダンスしか出力できない。状態Cによるアフォーダンスの文脈的解釈は、Fの**後段**にいる`Agent C`の役割となっている。

#### 2. Functor G: 生成 vs 側面選択 (Generation vs. Aspect Selection)

-   **理論**: `G(Action, C)` は、ある行動（「持ち上げる」）と状態C（「急いでいる」）に基づき、形状の特定の**側面**（「持ち手」）に注目する、**選択(selection)**のプロセスである。
-   **実装**: `FunctorG`はFoldingNetであり、アフォーダンス分布から点群全体を**生成(generation)**する。これは、理論が要求する「既存の形状からの側面選択」とは全く異なるプロセスである。

---

## 4. このズレが意味すること (Implications of This Gap)

この理論と実装のズレは、失敗を意味するものではない。むしろ、**我々のモデルがどのように機能しているかについての、より深い理解**を与える。

-   **現在の成功の理由**: Phase 2実験で観測された「有能感駆動行動」は、**疎結合アーキテクチャの賜物**である可能性が高い。静的で安定したFとG（川床）が存在し、その上でエージェントCがSlack（ηとε）を手がかりに最適な行動系列（水流）を学習する。もしFとGがCによって常に変動していたら、学習は不安定になっていたかもしれない。

-   **次世代アーキテクチャへの課題**: 真の「目的のフィルター」を実現するためには、FとGのアーキテクチャを根本的に見直す必要がある。例えば、FiLM (Feature-wise Linear Modulation) 層のような技術を用いて、状態CのベクトルでFとGの内部計算を動的に変調する**「条件付き随伴 (Conditional Adjunction)」**の実装が考えられる。これは、`legacy_code`内に存在する`conditional_adjunction_v4.py`の設計思想に近い。

この分析は、現在のアーキテクチャの限界と、今後の研究開発の方向性を明確に示すものである。

This gap between theory and implementation does not signify a failure. Rather, it provides a **deeper understanding of how our model actually works**.

-   **Reason for Current Success**: The "competence-driven behavior" observed in the Phase 2 experiment is likely a result of the **loosely coupled architecture**. A static and stable F and G (the riverbed) exist, upon which Agent C learns the optimal action sequence (the water flow) using Slack (η and ε) as cues. If F and G were constantly fluctuating with C, the learning might have been unstable.

-   **Challenge for Next-Generation Architecture**: To realize a true "purpose filter," the architecture of F and G needs a fundamental overhaul. For example, one could implement a **"Conditional Adjunction"** using techniques like FiLM (Feature-wise Linear Modulation) layers to dynamically modulate the internal computations of F and G with the state vector C. This aligns with the design philosophy of `conditional_adjunction_v4.py` found in the `legacy_code`.

This analysis clarifies the limitations of the current architecture and points to future directions for research and development.
