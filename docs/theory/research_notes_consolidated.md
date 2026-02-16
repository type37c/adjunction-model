# 知性とは「遊び」を管理する能力である：随伴モデルと有能感駆動エージェントの探求
# Intelligence is Slack Management: An Inquiry into the Adjunction Model and Competence-Driven Agents

**研究ノート — v5.0**
**Research Note — v5.0**

**日付**: 2026-02-15
**Date**: February 15, 2026

---

## 要旨 (Abstract)

本研究は、知性の本質を「予測精度の最大化」ではなく**「不確実性を管理し、利用する能力」**として再定義することを試みる。我々は、圏論の**随伴(Adjunction)**を用いてエージェントの世界モデルを定式化し、そのモデル内に意図的に保持される「遊び」や「緩み」である**Slack**という概念を導入する。画期的な実験結果として、エージェントはこのSlackを未知の探索（好奇心）のためではなく、**タスク遂行能力の向上（有能感）**のために利用することを発見した。これは、従来の強化学習における探索と活用のトレードオフを覆し、**有能感駆動(competence-driven)**という新しいエージェントの動機付け原理を示唆する。最終的に、我々は「保留構造が創発する条件の設計」とは、環境の複雑化ではなく、**エージェントに対する制約を極限まで取り除くこと**であるという結論に到達した。

This research attempts to redefine the essence of intelligence not as the maximization of prediction accuracy, but as **the ability to manage and exploit uncertainty**. We formalize an agent's world model using the concept of **Adjunction** from category theory and introduce the notion of **Slack**, a quantifiable measure of the "play" or "looseness" intentionally preserved within that model. As a groundbreaking experimental result, we discovered that the agent utilizes this Slack not for exploring the unknown (curiosity), but for **enhancing its task-performance capabilities (competence)**. This finding challenges the conventional exploration-exploitation trade-off in reinforcement learning and suggests a new motivational principle for agents: **competence-driven behavior**. Ultimately, we conclude that designing the conditions for the emergence of suspension structures is not about complicating the environment, but about **removing constraints on the agent to an extreme degree**.

---

## 1. 問い：知性とは何か？ (The Question: What is Intelligence?)

現在の大規模言語モデル（LLM）は、膨大なデータから統計的パターンを学習し、驚異的な能力を発揮する。しかし、その知性には身体性がなく、現実世界との相互作用から意味を理解する**記号接地問題**が未解決のままである [1]。彼らは「コップ」という単語を知っていても、「コップを掴む」という行為に伴う物理的感覚や、それを落とした時の結果を知らない。

我々は、知性の本質を問い直す。もし、知性が「常に正しい予測をすること」ではなく、**「予測が破綻した時にどう対処するか」**という能力にあるとしたら？もし、知性が「エラーを最小化する」プロセスではなく、**「エラーや不確実性を意図的に保持し、利用する」**プロセスだとしたら？

この問いを追求するため、我々は**物理的意味的随伴モデル (Physical-Semantic Adjunction Model)**を提案する。このモデルの核心は、知性とは**Slack（遊び）の管理能力**である、という仮説にある。

---

## 2. 理論的枠組み：随伴、Slack、そして保留構造
## 2. Theoretical Framework: Adjunction, Slack, and the Suspension Structure

### 2.1 随伴(Adjunction)：世界と相互作用するモデル

我々は、エージェントと世界の相互作用を、圏論における**随伴 (Adjunction F ⊣ G)** という数学的構造を用いてモデル化する [2]。

-   **関手 F (Shape → Action)**: オブジェクトの形状から、可能な行動（アフォーダンス）を推論する。
-   **関手 G (Action → Shape)**: ある行動を実現するために必要な形状を推論する。

### 2.2 Slack：モデル内の「遊び」の定量化

理想的な世界では、`G(F(Shape))`は元の`Shape`と一致する。しかし、現実のモデルには必ずズレが生じる。我々はこのズレを「エラー」として最小化するのではなく、**Slack**と名付け、意図的に保持・測定する [3]。

-   **η (Unit Slack)**: `||Shape - G(F(Shape))||²` (知覚的な不確実性)
-   **ε (Counit Slack)**: `||Affordance - F(G(Affordance))||²` (意味的な曖昧さ)

### 2.3 保留構造(Suspension Structure)：Slackを管理する原理

Slackを管理し、知的な行動に繋げるための高次の原理が**保留構造 (Suspension Structure)**である [4]。我々は、**「地質・川床・水流」**の比喩を用いてこれを説明する。

| 要素 (Element) | 比喩 (Analogy) | 説明 (Description) |
| :--- | :--- | :--- |
| **随伴 F ⊣ G** | 川床 (Riverbed) | エージェントの現在の世界モデル。経験によって常に変化する。 |
| **行動・経験** | 水流 (Water Flow) | 川床の形を削り、変化させる力。 |
| **保留構造** | 地質 (Geology) | 川床がどのように形成・変形されるかを規定する、より深く不変の原理。 |

我々が設計するのは「川床」そのものではなく、**「地質」**である。つまり、我々はSlackを管理するための**創発的条件**を設計するのである。

---

## 3. 発見：Slackは有能感のために使われる
## 3. Discovery: Slack is Used for Competence, Not Curiosity

**Phase 2 Slack実験**において、モデルの訓練から形状再構成損失（ηの最小化）を意図的に取り除いた結果、以下の事実が判明した [3]。

| 指標 (Metric) | 結果 (Result) | 意義 (Significance) |
| :--- | :--- | :--- |
| **総Slack変化 (Total Slack Change)** | **+757%** | Slackは自発的に保持・増幅される。 |
| **η/ε 相関 (η/ε Correlation)** | **0.92** | 形の遊び(η)と意味の遊び(ε)は強く結合している。 |
| **Slack-KL 相関 (Slack-KL Correlation)** | **-0.99** | Slackは**探索ではなく活用**のために使われる。 |

最も衝撃的な発見は、SlackとKLダイバージェンス（探索行動の指標）の間に見られた**ほぼ完璧な負の相関 (-0.99)**である。これは、エージェントが獲得した「遊び」を、新しい可能性を探すため（好奇心）ではなく、**既存のタスクをより効率的に遂行するため（有能感）**に利用していることを示している。これは、**有能感 (competence)**、すなわち「うまくやれている」という感覚を最大化しようとする、新しい動機付け原理の存在を示唆する。

---

## 4. 実装とアーキテクチャ
## 4. Implementation and Architecture

この発見を可能にしたのは、**3段階の訓練プロセス**と、**エージェント層C**の設計である [5]。

1.  **Phase 1: 随伴の事前学習 (Adjunction Pre-training)**: 安定だが「堅い」世界モデル（川床）を獲得する。
2.  **Phase 2: Slackの保存とエージェントの協調学習 (Slack Preservation & Agent Co-training)**: 再構成損失を取り除き、Slackの増幅を許容しながら、エージェント層Cを同時に訓練する。**この段階で有能感駆動の振る舞いが創発する。**
3.  **Phase 3: エージェントのファインチューニング (Agent Fine-tuning)**: 世界モデルを固定し、特定のタスクに対してエージェント層Cのポリシーを最適化する。

---

## 5. 次の地平：時間的保留と目的の創発
## 5. The Next Horizon: Temporal Suspension and the Emergence of Purpose

Phase 2実験は、ある瞬間の**静的な保留**を捉えた。しかし、保留の本質は**時間的・文脈的な現象**である [7]。単語「走」は、それだけでは動作を確定させないが、「走れメロス」という文脈の中で初めて意味が立ち上がる。同様に、エージェントの知性も、時間の流れの中で創発するはずである。

Phase 2 experiments captured **static suspension** at a single moment. However, the essence of suspension is a **temporal and contextual phenomenon** [7]. The word "run" does not determine an action on its own, but its meaning emerges within the context of "Run, Melos, Run." Similarly, an agent's intelligence should emerge in the flow of time.

この洞察に基づき、我々は**目的創発・能動的組立実験 (Purpose-Emergent Active Assembly Experiment)**を、驚くほどシンプルな形に昇華させた。

Based on this insight, we have refined the **Purpose-Emergent Active Assembly Experiment** into a surprisingly simple form.

### 5.1 実験の最終形：余計なものを取り除く
### 5.1 The Final Form of the Experiment: Removing the Superfluous

実験の目的は、エージェントが自発的に単一の目標を選択し、それに向かって一貫した行動をとるか、すなわち**志向性(Intentionality)**の創発を検証することである。そのために、我々はエージェントに対する制約を極限まで取り除いた。

The goal is to verify the emergence of **intentionality**: whether the agent spontaneously selects a single goal and acts consistently towards it. To achieve this, we removed constraints on the agent to an extreme degree.

| 撤廃された制約 (Removed Constraints) | なぜ不要か (Why it's Unnecessary) |
| :--- | :--- |
| Progressive Revelation（段階的開示） | 曖昧さはエージェントの内部に生まれるべき (Ambiguity should arise within the agent) |
| 1つの正解形状（target_points） | 目標はエージェントが自ら選ぶべき (The agent should choose its own goal) |
| 再構成損失（L_recon） | Slackの増幅を許容するため (To allow for Slack amplification) |

その結果、実験は以下の2つの損失関数のみで構成される。

As a result, the experiment consists of only two loss functions.

1.  **目的損失 (Purpose Loss)**: `L_purpose = min(CD(pos, sphere), CD(pos, cube), ...)`
    -   「いずれかの既知の形に近づけばよい」という、極めて弱い制約。
    -   An extremely weak constraint: "It is sufficient to get closer to any of the known shapes."
2.  **コヒーレンス損失 (Coherence Loss)**: `L_coherence = -log(η + ε)`
    -   Slackをゼロにせず、情報量として維持するための正則化項。
    -   A regularization term to maintain Slack as information, rather than minimizing it to zero.

### 5.2 「何もしない」ことの設計
### 5.2 The Design of "Doing Nothing"

この驚くべき単純化は、我々のプロジェクトにおける最も重要な発見の一つに繋がった。

This surprising simplification led to one of the most important discoveries in our project.

> **「保留構造が創発する条件を設計する」とは、最終的に「余計なものを取り除くこと」だった。**
> **Designing the conditions for the emergence of a suspension structure ultimately meant "removing everything superfluous."**

我々は、安定した世界モデル（Phase 2で学習済みの「川床」）を用意し、エージェントに「どれかの形に近づけ」という漠然とした目的を与え、あとは**自由にさせる**。環境を複雑にしたり、スケジュールを与えたりする必要はなかった。エージェントが自らの行動の結果（`η(t)`の変動）を観測し、学習する自由を与えるだけで、目的ある行動が創発する土壌が整うのである。

We provide a stable world model (the "riverbed" learned in Phase 2), give the agent a vague objective to "approach one of the shapes," and then let it **be free**. There was no need to complicate the environment or provide schedules. Simply giving the agent the freedom to observe the consequences of its own actions (the fluctuations of `η(t)`) and learn from them prepares the ground for the emergence of purposeful behavior.

この目的の創発を駆動するのが、**Priority原理**である [6]。

Driving this emergence of purpoこの目的の創発を駆動するメカニズムは、当初**Priority原理**として定式化された [6]。しかし、2026年2月16日の詳細な理論的検討の結果、この原理はより洗練された形に昇華された。

**旧原理（問題点）**:
> priority = coherence × uncertainty × valence

この掛け算による定式化は、3つの軸の質的な違い（特にvalenceの時間性）を無視し、注意の配分方法という「どう使うか」の部分まで人間が設計してしまう**過剰設計**であったことが明らかになった。

**新原理：軸の提供と使用法の創発**

> **設計するのは軸（coherence, uncertainty, valence）であり、それをどう使うかは創発に委ねる。**

新しい原理では、Priority計算モジュールは廃止される。代わりに、coherenceとuncertaintyは「観測」として、valenceは「記憶」としてAgent Cに直接提供される。Agent Cは、これらの入力を統合し、どの情報に注意を払うべきかを**自ら学習する**。この注意の配分方法そのものが、創発の対象となるのである。この価値判断の蓄積と自己組織化こそが、目的の核となる。

The agent preferentially directs its attention to areas where its current model is broken (coherence), it is unsure what it is (uncertainty), and it has had positive outcomes from addressing similar breakdowns in the past (valence). This accumulation of value judgment (valence) forms the core of purpose.

---

## 6. 結論 (Conclusion)

本研究は、知性の探求において、エラー最小化というパラダイムから**Slack管理**という新しいパラダイムへの転換を提案する。我々は、エージェントが意図的に保持する「遊び」であるSlackが、単なるノイズではなく、適応性と能力向上のための重要なリソースであることを実証した。特に、Slackが好奇心ではなく**有能感**を駆動するという発見は、自律的エージェントの設計思想に大きな影響を与えるだろう。

This research proposes a paradigm shift in the study of intelligence, moving from error minimization to **Slack management**. We have demonstrated that Slack, the "play" intentionally preserved by an agent, is not mere noise but a critical resource for adaptability and capability enhancement. In particular, the discovery that Slack drives **competence** rather than curiosity will have profound implications for the design of autonomous agents.

そして、この探求の過程で、我々は「創発の設計」に関するより深い真理に到達した。それは、複雑なルールを追加することではなく、**本質的でない制約を注意深く取り除き、システムが自ら構造を形成するための「余白」を与えること**である。我々の次の挑戦は、この極限まで単純化された実験系において、いかにして安定的で長期的な**目的**が創発するかを観測し、記述することである。それは、真に自律的な知性の謎を解く鍵となるだろう。

And in the course of this inquiry, we have arrived at a deeper truth about the "design of emergence." It is not about adding complex rules, but about **carefully removing non-essential constraints and providing the "margin" for the system to form its own structure**. Our next challenge is to observe and describe how stable, long-term **purpose** emerges in this extremely simplified experimental system. This will be key to unlocking the mystery of truly autonomous intelligence.

---

## 7. 参考文献 (References)

[1] Harnad, S. (1990). The symbol grounding problem. *Physica D: Nonlinear Phenomena*, 42(1-3), 335-346.
[2] `docs/theory/research_notes_consolidated.md` (v3.0, internal reference).
[3] `results/phase2_slack/COMPREHENSIVE_ANALYSIS_REPORT.md` (internal reference).
[4] `docs/theory/suspension_and_confidence.md` (internal reference).
[5] `ARCHITECTURE.md` (internal reference).
[6] `docs/theory/purpose_space_P_design.md` (internal reference).
[7] `docs/archive/01_temporal_suspension.md` (internal reference).
