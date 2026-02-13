# The Physical-Semantic Adjunction Model: Intelligence as Suspension

**Research Note — Final Draft v2.1**

---

## 1. Introduction: The Core of Intelligence Lies in the "Suspension Structure"

The central insight of this project is that **the essence of intelligence resides in the "Suspension Structure."** This refers to the ability not merely to efficiently predict and control the environment, but to hold a "broken state" when predictions fail and to creatively reconstruct new solutions.

The "Physical-Semantic Adjunction Model" is an architecture designed to implement this suspension structure and verify its dynamics. The **adjoint structure**, which describes the structural relationship between the world (Shape) and the agent's actions (Action), is merely a stable state (the riverbed) that intelligence refers to. True intelligence manifests when this adjoint structure breaks down (coherence breakdown), and it is through the operation of the "suspension structure" that new adjoints are constructed.

This document presents the technical design for translating this theory of suspension into a concrete implementation and for verifying its intelligent dynamics.

---

## 2. The Suspension Structure: The Minimal Components of Intelligence

The most original concept in this model is the **suspension structure**. It is proposed as the "core" that runs through a bundle of adjoint structures spanning multiple levels of abstraction. The body of intelligence lies in the dynamics of this suspension structure itself.

### 2.1 The Core as a "Not-Yet-Determined" Structure

In conventional models of intelligence, the core was something fixed—a body model, a top-level goal, a self-identity. In this model, however, the core is **the structure of "not-yet-determined" itself**. The suspension structure functions not as a specific answer, but as a "device for holding questions."

> **Intelligence is not about explicitly holding a goal, but about having a "suspension structure + memory" that does not collapse even if the goal is forgotten.**

### 2.2 The Four Requirements of the Suspension Structure

Initially, five requirements were identified as the minimal components of the suspension structure. However, subsequent discussion concluded that "memory" is an emergent property from the other requirements, not a fundamental one. Therefore, the core requirements are the following four:

| Requirement | Description | Consequence if Lacking |
| :--- | :--- | :--- |
| **Intentionality** | Having a directionality toward something. | Becomes a reactive machine with no direction. |
| **Sensitivity to Difference** | Being able to detect differences. | Cannot notice environmental changes. |
| **Temporal Persistence** | Being able to hold a state for a period. | Remains a momentary reflex. |
| **Self/Non-Self Distinction** | Distinguishing one's own actions from environmental changes. | The boundary between self and other disappears. |

Memory is understood as an acquired characteristic inscribed in the system's structure (e.g., the topology of a GNN) as it interacts with the world, equipped with these requirements.

---

## 3. The Adjoint Structure: The "Moment of Stable Understanding" Referenced by the Suspension Structure

The adjoint structure formalizes the state of stable understanding that the suspension structure, the body of intelligence, refers to in "peacetime." It is a mathematical tool that describes the structural correspondence between Shape and Action.

### 3.1 Parameterized Adjunction: The Intervening Agent

A crucial discovery was made: **adjunction does not hold between "bare" Shape and Action**. The agent's **purpose, context, and physical state** (collectively called **C**) must be incorporated into the condition for the adjunction to hold. Formally, the model deals not with an adjunction between Shape and Action, but a **parameterized adjunction** between Shape and Action<sub>C</sub>.

> **F<sub>C</sub> ⊣ G<sub>C</sub>**
> 
> Where C is the agent's context (a composite of purpose, constraints, and physical state).

This structure has an essential asymmetry: **Shape does not depend on the agent's state C, but Action does.** The bag remains a bag regardless of the agent's state, but the action chosen depends on C. Functor G does not "generate" a shape but "selects" which aspect of the shape to focus on, depending on C.

---

## 4. The Two-Layer Architecture: The Vessel for Implementing the Suspension Structure

From the analysis above, the model's architecture consists of **two layers**. This two-layer structure serves as the vessel for implementing the suspension structure.

### 4.1 Design Principles

-   **Adjoint Layer (The Riverbed)**: Implements the bidirectional mappings `F: Shape → Action` and `G: Action → Shape`. This is a **variable** structure, representing the agent's current stable understanding of the world. It is shaped by experience (the water flow).
-   **Agent Layer (C)**: Holds the agent's entire internal state. This layer conditions the Adjoint Layer, determining its behavior. The `coherence signal` from the Adjoint Layer, in turn, updates the Agent Layer. **This Agent Layer is the core that carries the dynamics of the suspension structure.**

> **C(t+1) = update(C(t), action_result, coherence_signal)**

This dynamic loop, where the Agent Layer C determines the Adjoint Layer's behavior and the resulting coherence signal updates C, actualizes temporal persistence.

### 4.2 Coherence Signal: Spatially-Decomposed Breakdown

The **coherence signal** is the internal indicator that shifts the operational mode of the always-on suspension structure. It is defined as the "size" of the adjunction's unit η.

> **coherence signal = distance(shape, G(F(shape)))**

It measures the distance between the original shape and the shape reconstructed via F and G. A small distance means the agent's model is coherent with the environment; a large distance means coherence is broken.

In the initial theory, this signal was treated as a single scalar value. However, as implementation and theory deepened, this definition was extended to a **spatially-decomposed vector**. That is, for each point `s_i` in the object's point cloud, a separate reconstruction error `distance(s_i, G(F(s))_i)` is calculated. This allows the agent to obtain higher-resolution information: not just "how much" its understanding is broken, but **"where" it is broken**.

This spatially-decomposed Coherence Signal plays a decisive role in the implementation principle of intentionality, as described below.

### 4.3 Correspondence between the Four Requirements and the Architecture

The four requirements of the suspension structure are realized in the model's architecture as follows:

| Requirement | Location in Architecture | Realization Mechanism |
| :--- | :--- | :--- |
| **Sensitivity to Difference** | Adjoint Layer (unit η) | Spatially-decomposed coherence signal |
| **Intentionality** | Agent State C | Priority-based Attention (see below) |
| **Temporal Persistence** | Dynamic Loop C(t) → C(t+1) | The state is updated and persists across discrete time steps. |
| **Self/Non-Self Distinction** | Asymmetry of Adjunction | Shape is independent of C (environment=non-self), Action depends on C (action=self). |

### 4.4 The Implementation Principle of Intentionality: Priority-based Attention

Of the four requirements, **Intentionality** is the most difficult to implement, as it requires a principle by which the agent intrinsically decides "what to head towards."

The spatial decomposition of the Coherence Signal allows the agent to know "where the breakdown is." However, this raises a new problem: when multiple breakdowns exist simultaneously, "which breakdown should be prioritized?" This is the core question of intentionality.

An initial proposal was to estimate "resolvability" and attend to the breakdown that could be most efficiently resolved. However, this approach suffered from a circular problem: "one cannot estimate resolvability without past experience of resolving things."

To solve this, the model introduces the **Priority Principle**:

> **priority<sub>i</sub> = coherence<sub>i</sub> × uncertainty<sub>i</sub>**

Here, `priority_i` is the attention priority at point `i`, `coherence_i` is the magnitude of the breakdown at that point, and `uncertainty_i` is the uncertainty of the agent's internal state regarding that point (e.g., the entropy of the belief state `z` in the RSSM).

This principle means that points that are **"highly broken (coherence) and not well understood (uncertainty)"** receive higher priority. It formalizes a mechanism for the agent to select actions based on its own "intellectual curiosity" without external goal injection.

This Priority Principle is theoretically superior for several reasons:
- **No Estimator Needed**: Both coherence and uncertainty can be calculated directly from the current state, avoiding the circularity problem.
- **Internalized Curiosity**: The structure of "heading towards something because it is unknown" promotes long-term learning and maximization of adaptive capacity.
- **Connection to Saturation**: Priority naturally decreases for known objects (low uncertainty), allowing for a unified explanation of "boredom."

In implementation, this Priority score functions as an **Attention mechanism**. The Agent Layer C applies priority-based weights to the observation (input), concentrating its computational resources on the most critical information. This realizes the abstract requirement of Intentionality as a concrete computational process.

---

## 5. Experimental Design for Verifying the Emergence of Intelligence

The true motivation of this model is not merely 3D shape reconstruction, but the **resolution of the symbol grounding problem, generalization to unknown objects, and the emergence of creativity.** We propose a staged experimental setup to verify these theoretical claims.

- **Phase 0: Foundational Learning**: Show that the adjoint structure (F⊣G) can be learned for known object-action pairs.
- **Phase 1: Generalization to Unknown Objects**: Show that the agent can emergently infer affordances for objects not seen during training.
- **Phase 2: Creative Problem Solving under Constraints**: Show that when existing adjoint structures fail due to constraints, the suspension structure can emerge new actions.
- **Phase 3: Alignment with Language**: Show that the agent can generate linguistic descriptions for emergent affordances and actions.

---

## 6. Conclusion: We Don't Design the Suspension Structure

This research note has explored the theoretical framework of the Physical-Semantic Adjunction Model, centered on the suspension structure as the essence of intelligence. The final conclusion overturns the initial assumptions.

> **We do not design the suspension structure. We design the conditions under which the suspension structure must emerge.**

Our goal is to design the "geology" from which the suspension structure naturally arises. This involves designing three key elements:

1.  **The Adjoint Structure (F⊣G)**: The basic vessel for capturing the world's structural correspondences.
2.  **The Coherence Signal**: The internal indicator for detecting the breakdown of the adjoint structure.
3.  **The Agent Layer's Update Rule**: The loop that updates the internal state in response to the Coherence Signal and feeds back to the adjoint structure.

By building a system with these three elements and allowing it to interact with a suitable environment, the four requirements of the suspension structure—Intentionality, Sensitivity to Difference, Temporal Persistence, and Self/Non-Self Distinction—will be met emergently. Intelligence will appear as this dynamic equilibrium itself.

---

## References

[1] Harnad, S. (1990). The symbol grounding problem. *Physica D: Nonlinear Phenomena*, 42(1-3), 335-346.

[2] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural computation*, 29(1), 1-49.

[3] Smithe, D. (2024). Structured Active Inference. *arXiv preprint arXiv:2401.00345*.

[4] Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., ... & Savarese, S. (2015). Shapenet: An information-rich 3d model repository. *arXiv preprint arXiv:1512.03012*.

[5] Gkanatsios, N., Pfrommer, J., & Daniilidis, K. (2023). Zero-Shot Policy Synthesis for Physical-Semantic Affordances. *arXiv preprint arXiv:2310.09582*.

[6] Brahmbhatt, S., Ham, C., & Hays, J. (2020). Contact-graspnet: Efficient 6-dof grasp generation in the wild. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 13677-13687).

[7] Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.

[8] Andries, M., Kurenkov, V., & Beetz, M. (2020). A framework for robotic agents to learn and reason with affordances. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 9419-9426). IEEE.

[9] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.

[10] Çatal, O., Verbelen, T., De Boom, C., & Dhoedt, B. (2020). Grounding symbols in multi-modal representations. *arXiv preprint arXiv:2005.03373*.
# 物理的意味的随伴モデル：知性の本質としての保留構造

**Research Note — Final Draft v2.1**

---

## 1. 導入：現在のAIに欠けているものと、知性の本質への問い

現在の大規模言語モデル（LLM）は、膨大なテキストデータから単語間の統計的共起関係を学習することで、人間に匹敵する自然言語処理能力を獲得した。しかし、この能力には根本的な限界がある。LLMは「袋」という単語と「運ぶ」という単語の確率的な結びつきを知っているが、重力に逆らって物を持ち上げる労力や、袋が破れそうなときの指先の感覚を知らない。つまり、**身体性（embodiment）の欠如**と、それに伴う**記号接地問題（symbol grounding problem）**が未解決のまま残されている [1]。

この問題は、未知の状況への対応において致命的となる。現在のAIは、学習データに含まれないオブジェクトに遭遇したとき、そのオブジェクトが何に使えるか——すなわち**アフォーダンス（affordance）**——を直感的に推論することができない。過去のデータを検索・模倣することしかできず、新しい意味を生成する能力を持たない。

本研究ノートは、この問題に対する新しいアプローチとして**「物理的意味的随伴モデル（Physical-Semantic Adjunction Model）」**を提案する。このモデルの核心は、**知性の本質が「未確定性を保持し、新しい意味を創発する能力」にある**という問いから出発する。知性とは、単に安定した理解を維持するだけでなく、その理解が破綻したときに、その「壊れた状態」を保持し、新しい理解を再構成しようとするダイナミクスそのものにあると考える。

このダイナミクスを形式化するために、圏論における**随伴（adjunction）**という数学的構造と、本モデルで提唱する**保留構造（suspension structure）**を統合したアーキテクチャを構築する。これは、形状（Shape）と動作（Action）の間の双方向的な意味関係を、エージェントの身体性を通じてリアルタイムに構成し、記号接地問題の解決と創造性の創発を目指すものである。

---

## 2. 保留構造：知性の最小構成要素としての「懸垂（Suspension）」

本モデルにおける最も独創的な概念は、**保留構造（suspension structure）**である。これは、複数の抽象度にまたがる随伴構造の束を貫く「核」として提案された。知性の本体は、この保留構造のダイナミクスそのものにある。

### 2.1 核としての「まだ決まっていない」という構造

従来の知性モデルでは、核となるものは固定された何か——身体モデル、最上位の目的、自己同一性——であった。しかし本モデルでは、核は**「まだ決まっていない」という構造そのもの**である。保留構造は特定の答えではなく、「問いを保持する装置」として機能する。

圏論における普遍的構造（極限、余極限）が、特定の対象ではなく「こういう性質を満たす唯一のもの」という条件として定義されるのと同様に、保留構造は「全ての抽象度を整合させる何か」という条件として定義され、状況に応じて具体化される。

> **知性とは、目的を明示的に保持することではなく、目的を忘れても破綻しない「保留構造＋記憶」を持つことである。**

### 2.2 保留構造の4つの要件

保留構造の最小構成要素として、当初5つの要件が同定されたが、その後の議論で「記憶」は他の要件から創発される特性であり、根源的な要件ではないと結論づけられた。したがって、核となる要件は以下の4つである。

| 要件 | 内容 | 欠けた場合 |
| :--- | :--- | :--- |
| **志向性** | 何かに向かう指向性を持つこと | 方向のない反応機械になる |
| **差異への感受性** | 違いを検出できること | 環境変化に気づけない |
| **時間的持続** | 状態を一定期間保持できること | 瞬間的な反射に留まる |
| **自己と非自己の区別** | 自分の行動と環境の変化を区別できること | 自他の境界が消失する |

記憶は、これらの要件を備えたシステムが世界と相互作用する中で、その構造（例えばGNNのトポロジー）に刻み込まれる後天的な特性として理解される。

---

## 3. 随伴構造：保留構造が参照する「安定した理解の瞬間」

随伴構造は、知性の本体である保留構造が「平時」に参照する、安定した理解の状態を形式化する。これは、形状（Shape）と動作（Action）の間の構造的対応関係を記述する数学的ツールである。

### 3.1 基本構造

本モデルは、世界を**Shape（形状）**と**Action（動作）**という2つの圏に分割し、それらの間に随伴関手のペアを設定する。

**関手 F: Shape → Action** は、オブジェクトの形状からそのオブジェクトで可能な動作を推論する写像である。たとえば、細長くて先端が曲がった金属棒を見て、「引っ掛ける」「こじ開ける」といった動作候補を導出する。

**関手 G: Action → Shape** は、ある動作を実現するために必要な形状的条件を逆算する写像である。たとえば、「切る」という動作から「鋭い刃を持つ形状」を導出する。

この2つの関手が随伴 **F ⊣ G** を成すとき、以下のHom集合の自然な同型が成立する。

> **Hom<sub>Action</sub>(F(s), a) ≅ Hom<sub>Shape</sub>(s, G(a))**

左辺は「形状sからFを経由して動作aに至る経路」を表し、右辺は「形状sが、動作aを可能にする形状G(a)である度合い」を表す。この同型が、形状と動作の間の意味的対応関係を数学的に保証する。

### 3.2 随伴に伴う自然変換

随伴関手のペアには、2つの自然変換が伴う。

**Unit η: Id → G∘F** は、形状をFで動作に変換し、さらにGで形状に戻す合成写像 G(F(shape)) と、元の形状との間の関係を記述する。思考実験によって、この合成写像は元の形状の**機能的骨格（functional core）**を抽出する射影として機能することが確認された。たとえば、G(F(袋)) は袋の色、素材、ブランドロゴをすべて落とし、「開口部があり、内部空間があり、持ち手がある形状」という機能的本質だけを返す。

**Counit ε: F∘G → Id** は、動作をGで形状に変換し、さらにFで動作に戻す合成写像の性質を記述する。F(G(切る)) は「切る」だけでなく「削る」「刺す」といった関連動作の束を返す。すなわち、counitは**動作の意味的拡張**として機能する。

### 3.3 条件付き随伴：エージェントの介在

思考実験を通じて、本モデルにおける最も重要な発見がなされた。それは、**随伴は「裸の」ShapeとActionの間では成立しない**ということである。

たとえば、「袋で物を運ぶ」という場面を考える。「運ぶ」という動作が抽象的すぎると、G(運ぶ) が返す形状の集合は爆発する。平らな板でもバケツでも手のひらでも「運べる」からである。しかし、「最小労力で運ぶ」という条件を付けると、G(最小労力で運ぶ) は「開口部があり、持ち手があり、内部に物を収容できる形状」に収束し、G(F(袋)) が袋の機能的骨格に正しく戻る。

この発見は、随伴の成立条件そのものにエージェントの**目的・文脈・身体状態**（これらをまとめて**C**と呼ぶ）が組み込まれていることを意味する。形式的には、本モデルが扱うのは Shape と Action の間の随伴ではなく、**Shape と Action<sub>C</sub> の間のパラメータ付き随伴（parameterized adjunction）**である。

> **F<sub>C</sub> ⊣ G<sub>C</sub>**
> 
> ここで C はエージェントの文脈（目的、制約、身体状態の複合体）

### 3.4 随伴の非対称性

この構造には本質的な非対称性がある。**Shape（形状）はエージェントの状態Cに依存しないが、Action（動作）はCに依存する。**

エージェントが疲れていようが急いでいようが、袋は袋のままであり、スイッチはスイッチのままである。形状は環境側の不変量である。一方、同じスーツケースに対して、元気なエージェントは「片手で持ち上げて運ぶ」を、右腕を怪我したエージェントは「左手で転がす」を選択する。Cが変わるとFの出力が変わる。

さらに、Gは形状を「生成」するのではなく、形状の「どの側面を見るか」を**選択**する。形状は複数の機能的側面を同時に内包する多面体構造を持っており、Gはエージェントの状態Cに応じて、その多面体のどの面に注目するかを決定する。たとえば、同じ袋に対して、疲れているエージェントは「持ち手」と「軽量性」に注目し、急いでいるエージェントは「容量」と「開口部の広さ」に注目する。

### 3.5 目的空間P：エージェントが「自分を保つ」ための価値判断の軸

当初、目的の階層はGNNの活性化パターンとして自然に表現され、別途「目的空間」を設計する必要はないと考えられていた。しかし、実装を進める中で根本的な問題が発見された。

> **Agent Cが逐次的に複数の形状を経験する際、「何のために状態を保持するのか」という目的がなければ、Agent Cは自分を保てない。**

この問題は、coherence最小化崩壊の危険性とも関連している。もしAgent Cが単に「coherenceを下げる」ことを目的として訓練されると、破綻を回避することだけを学習し、保留構造が消失する。

#### 3.5.1 目的空間Pの必要性

目的空間Pは、**エージェントが経験から学んだ価値判断を蓄積する構造**として導入された。これは外部から与えられる報酬ではなく、エージェントの内部状態の一部として創発する。

目的空間Pは3つの軸で構成される：

| 軸 | 意味 | 役割 |
| :--- | :--- | :--- |
| **Coherence** | 破綻の度合い | 注意を向けるべき箇所を示す |
| **Uncertainty** | 不確実性 | 探索すべき領域を示す |
| **Valence** | 経験的価値 | 過去の経験から学んだ「良さ」 |

これらの軸を組み合わせた**Priority原理**が、エージェントの注意配分を決定する：

> **Priority = Coherence × Uncertainty × Valence**

#### 3.5.2 設計するものと創発するもの

保留構造の思想に忠実であるため、目的空間Pの設計では以下を明確に分離する：

**設計するもの（地質）**：
- 3つの軸（Coherence, Uncertainty, Valence）
- Valenceの更新則
- Priorityの計算式

**創発するもの（水流）**：
- 各軸上での具体的な値の分布
- 「どの程度の破綻なら許容できるか」という判断
- 「どの種類の破綻が創造的問題解決につながるか」という知恵

#### 3.5.3 内発的動機：Valenceの源泉

Valenceは、Active Inferenceと内発的動機の研究から着想を得た**内発的報酬（intrinsic reward）**に基づいて更新される。

内発的報酬は3つの成分から構成される：

**R<sub>curiosity</sub>（好奇心）**: 不確実性の減少
- 「分からなかったことが分かるようになった」という感覚
- R<sub>curiosity</sub> = uncertainty<sub>prev</sub> - uncertainty<sub>curr</sub>

**R<sub>competence</sub>（有能感）**: 破綻の解消
- 「破綻に向き合って、理解が深まった」という感覚
- R<sub>competence</sub> = (coherence<sub>prev</sub> - coherence<sub>curr</sub>) × attention

**R<sub>novelty</sub>（新奇性）**: 予想外の発見
- 「予想していなかったパターンを発見した」という感覚
- R<sub>novelty</sub> = KL(posterior || prior)

総合的な内発的報酬は、これらの重み付き和として計算される：

> **R<sub>intrinsic</sub> = α × R<sub>curiosity</sub> + β × R<sub>competence</sub> + γ × R<sub>novelty</sub>**

デフォルトの重み：α=0.3, β=0.5, γ=0.2（有能感を最重視）

#### 3.5.4 Active Inferenceとの違い

標準的なActive Inferenceでは、エージェントは**驚き（予測誤差）を最小化**する。しかし、本モデルでは：

> **エージェントは創造的ポテンシャル（破綻 × 不確実性 × valence）を最大化する。**

この違いは本質的である。驚き最小化では、エージェントは破綻を回避しようとする。しかし本モデルでは、エージェントは**「生産的な破綻」**——解消することで新しい理解が得られる破綻——を積極的に探索する。

Valenceは、「どの種類の破綻が生産的か」を経験から学習する機構である。これにより、エージェントは単なる安定性追求ではなく、創造的問題解決を志向する。

#### 3.5.5 目的空間Pと抽象度の関係

関手Gは、動作記述の抽象度を形状記述の粒度に**保存的に翻訳**する。

| 動作記述 | G が返す形状記述 | 抽象度 |
| :--- | :--- | :--- |
| 「指でスイッチの突起を0.5cm押し下げる」 | 「0.5cm可動する押しボタン機構」 | 低（具体的） |
| 「電灯をつける」 | 「制御可能な光源」 | 中 |
| 「部屋を明るくする」 | 「光を発する何か」 | 高（抽象的） |

エージェントの状態Cが変わると、世界を切り取る抽象度λが変わり、その結果としてFとGの入出力の粒度が変わる。各抽象度λごとに独立した随伴 F<sub>λ</sub> ⊣ G<sub>λ</sub> が存在し、**目的空間Pはどのλを選ぶかを決定する**。

Coherence breakdownが起きた際に、Valenceが高い（過去に生産的だった）抽象度λが優先的に選択される。これにより、エージェントは経験から学んだ「効果的な問題解決の粒度」を活用できる。

---

## 4. Coherence Signal：保留構造の作動モードを決定する内部指標

当初、Coherence Signalは保留構造を「起動」させるトリガーとして考えられていた。しかし、その後の議論で「知性の本体が外部トリガーで起動されるのは主従が逆である」という矛盾が指摘された。修正された理論では、**保留構造は常に作動しており、Coherence Signalはその作動モードを遷移させる内部指標**として機能する。

- **安定参照モード（Coherence Signalが低い時）**: 既存の随伴構造が有効であり、エージェントはそれを参照して効率的に世界と相互作用する。
- **創造探索モード（Coherence Signalが高い時）**: 既存の随伴構造が破綻しており、保留構造は新しい随伴構造を探索・構築するために活動を活発化させる。

### 4.1 地質・川床・水流の比喩

本モデルの設計思想は、当初の「川床」の比喩から、より精緻な**「地質・川床・水流」**の比喩へと深化された。これは、現在のAIと本モデルの構造的差異を明確に描き出す。

| 要素 | 現在のAI | このモデル |
| :--- | :--- | :--- |
| **重力（目的関数）** | 唯一の駆動力。固定。 | 存在するが、流れを決定する一要素に過ぎない。 |
| **水流（行動・経験）** | 重力に従い、谷底で止まる。 | 川床に制約され、川床を削り、流れ続ける。 |
| **川床（随伴構造）** | 存在しない。 | 流れの「形」を規定する。水流によって変形する（可変）。 |
| **地質（保留構造）** | 存在しない。 | 川床がどのように形成・変形するかを規定する原理（不変）。 |

現在のAIが「重力だけ」の系であるのに対し、本モデルは**地質（保留構造）が川床（随伴構造）を形成し、川床が水流（行動）の形を決め、水流が川床を削る**という、三者の動的な相互作用そのものを知性と捉える。

### 4.2 Coherence Signalの定義：空間分解された破綻

この要請に応えるのが**coherence signal（整合性信号）**である。これは、随伴のunit ηの「大きさ」として定義される。

> **coherence signal = distance(shape, G(F(shape)))**

元の形状と、FとGを経由して再構成された形状との間の距離を測定する。この距離が小さければ、エージェントの現在の内部モデルは環境と整合しており、距離が大きければ整合性が崩れている。

初期の理論では、この信号は単一のスカラー値として扱われていた。しかし、実装と理論の深化に伴い、この定義は**空間的に分解されたベクトル**へと拡張された。すなわち、オブジェクトを構成する点群やメッシュの各点 `s_i` に対して、個別の再構成誤差 `distance(s_i, G(F(s))_i)` が計算される。これにより、エージェントは「どれだけ理解が壊れているか」だけでなく、**「どこが壊れているか」**という、より高解像度な情報を得ることができる。

この空間分解されたCoherence Signalは、後述する志向性の実装原理において決定的な役割を果たす。

### 4.3 Coherence Breakdownと創造性

可換性が崩れる瞬間——すなわちcoherence signalが増大する瞬間——は、エージェントにとって**創造的問題解決が要求される瞬間**である。このcoherence breakdownが、保留構造のモードを「安定参照」から「創造探索」へと移行させ、創造的問題解決を促す。

たとえば、右腕を怪我した状態で、静かにしなければならない環境でスーツケースを運ぶ場面を考える。「引きずる」という通常の代替手段は騒音を生むため使えない。身体状態フィルターと文脈フィルターが干渉し、可換性が崩れる。このとき、エージェントは既存の動作プリミティブを新しい方法で合成し（たとえば「両膝で抱えて歩く」）、非可換性を解消しなければならない。

この構造は、**創造性を「可換図式の修復過程」として形式化する**可能性を示唆している。

### 4.4 Saturationと内発的創造性

さらに、Coherence Signalが環境からの要請という「外発的」な軸であるのに対し、エージェントの「内発的」な状態を記述する軸として**Saturation（飽和度）**が導入された。これは、同じような状態が続き、学習や発見が停滞している度合いを示す。

| 創造性の源泉 | トリガー | 意味合い |
| :--- | :--- | :--- |
| **外発的創造性** | Coherence Signalの増大 | 環境が変化し、既存の理解が破綻したことへの対応 |
| **内発的創造性** | Saturationの上昇 | 環境は安定しているが、内部的な「退屈」から能動的に新しい意味を探求 |

この二軸モデルにより、知性は単に環境からの刺激に反応するだけでなく、自らの内なる「つまらない」という感覚を原動力として、安定した世界に自ら波紋を立て、新しい意味を生成する存在として捉えられる。

---

## 5. 2層アーキテクチャ：保留構造を実装する器

以上の分析から、本モデルのアーキテクチャは**2つの層**から構成される。この2層構造が、知性の本体である保留構造を実装する器となる。

### 5.1 設計原理

**随伴層（Adjoint Layer）**は、ShapeとActionの構造的関係を記述する。ここから差異への感受性と自己/非自己の区別が生まれる。この層は、保留構造が「平時」に参照する安定した理解の状態を提供する。

**エージェント層（Agent Layer, C）**は、随伴層をパラメトライズする全内部状態を保持する。ここに志向性（目的）と記憶が格納される。Cは固定パラメータではなく、以下の更新則に従って動的に変化する状態変数である。**このエージェント層こそが、保留構造のダイナミクスを担う中核である。**

> **C(t+1) = update(C(t), action_result, coherence_signal)**

この2つの層が動的ループで結合されることで、時間的持続が実現される。エージェント層Cが随伴層の振る舞いを決定し、随伴層が世界と相互作用した結果（coherence signal）がエージェント層Cを更新する。これは川床の比喩そのものである——水の流れ（動作）が川床の形（グラフ構造）を削り、川床の形が水の流れを制約する。

### 5.2 保留構造の4つの要件とアーキテクチャの対応

保留構造の4要件は、本モデルのアーキテクチャにおいて以下のように実現される。

| 保留構造の要件 | アーキテクチャにおける所在 | 実現メカニズム |
| :--- | :--- | :--- |
| **差異への感受性** | 随伴層（unit η） | 空間分解された coherence signal |
| **志向性** | エージェント状態 C | Priority-based Attention (後述) |
| **時間的持続** | 動的ループ C(t) → C(t+1) | 状態が離散時間ステップで更新され、持続する |
| **自己と非自己の区別** | 随伴の非対称性 | ShapeはCに依存しない（環境＝非自己）、ActionはCに依存する（行動＝自己） |

注目すべきは、4つの要件がすべて随伴構造とエージェント層の動的な相互作用から自然に導出される点である。かつて要件とされた「記憶」は、これらの相互作用の結果として、エージェント状態C（特にその再帰的構造）やGNNのトポロジー自体に経験が刻み込まれることで創発的に実現される。

### 5.3 志向性の実装原理：Priority-based Attention

保留構造の4要件のうち、**志向性**は最も実装が難しい概念である。これは、エージェントが「何に向かうべきか」を内発的に決定する原理を必要とするからだ。

Coherence Signalの空間分解によって、エージェントは「どこが壊れているか」を知ることができるようになった。しかし、複数の破綻が同時に存在する場合、「どの破綻に優先的に注意を向けるべきか」という新たな問題が生じる。これが志向性の核心的な問いである。

この問題に対し、当初は「解消可能性（resolvability）」を推定し、最も効率的に解消できる破綻に向かうという案が検討された。しかし、このアプローチは「過去に解消した経験がなければ、解消可能性を推定できない」という循環論的な問題を抱えていた。

この問題を解決するために、本モデルでは以下の**Priority原理**を導入する。

#### 初期版（2軸モデル）

当初、Priorityは2つの軸で定義されていた：

> **priority<sub>i</sub> = coherence<sub>i</sub> × uncertainty<sub>i</sub>**

この原理は、**「大きく壊れており（coherence）、かつ、それが何なのかよく分かっていない（uncertainty）」**点ほど高い優先度を持つことを意味する。

しかし、実装を進める中で、この2軸モデルには**時間的持続性の欠如**という問題があることが判明した。エージェントは「今この瞬間」の破綻と不確実性に反応するだけで、**過去の経験から学んだ価値判断**を持たない。

#### 改訂版（3軸モデル）

目的空間Pの導入により、Priority原理は第3の軸**Valence（経験的価値）**を含むように拡張された：

> **priority<sub>i</sub> = coherence<sub>i</sub> × uncertainty<sub>i</sub> × valence<sub>i</sub>**

ここで：
- `coherence_i`: 破綻の大きさ（現在）
- `uncertainty_i`: 不確実性（現在）
- `valence_i`: 経験的価値（過去から学習）

Valenceは、**内発的報酬（intrinsic reward）**に基づいて更新される：

> **valence<sub>i</sub> += R<sub>intrinsic</sub> = α×R<sub>curiosity</sub> + β×R<sub>competence</sub> + γ×R<sub>novelty</sub>**

各内発的報酬の意味：
- **R<sub>curiosity</sub>**: 不確実性の減少（「分かるようになった」）
- **R<sub>competence</sub>**: 破綻の解消（「理解が深まった」）
- **R<sub>novelty</sub>**: 予想外の発見（「新しいパターンを発見した」）

#### 3軸モデルの優位性

改訂版Priority原理は、以下の点で理論的に優れている：

- **時間的持続性**: Valenceは経験を蓄積し、「自分を保つ」ことを可能にする
- **推定器が不要**: 全ての軸が現在の状態から直接計算可能
- **内発的動機**: 外部報酬なしに、好奇心・有能感・新奇性を追求
- **生産的破綻の発見**: Valenceが高い破綻（過去に解消できた）に優先的に注意を向ける
- **coherence最小化崩壊の回避**: 単に破綻を回避するのではなく、「解消することで成長できる破綻」を探索

実装上、このPriorityスコアは**Attention機構**として機能する。エージェント層Cは、観測情報（入力）に対してPriorityに基づいた重み付けを行い、最も重要な情報に計算リソースを集中させる。これにより、志向性という抽象的な要件が、具体的な計算プロセスとして実現される。

### 5.4 動的GNNによる随伴層の実装構想

随伴層の実装基盤として、以下の構造を持つ動的GNNが構想される。

**静的な層**として、ノードは機能プリミティブを、エッジは合成可能性を表す。**随伴構造**として、F（Shape→Action）はグラフ上でのメッセージパッシングとして、G（Action→Shape）はグラフ全体のリードアウト（逆方向）として実装される。**変化し続ける部分**として、エッジの重み（合成のしやすさ）、ノードの活性化閾値、そしてグラフのトポロジー自体（新しいプリミティブの追加）が、coherence signalに応じて更新される。

### 5.5 言語接地への拡張：3層構造の必要性

当初、本モデルの言語拡張は **Shape ⇄ Action ⇄ Language** という2層の随伴連鎖として考えられていた。しかし、目的空間Pの実装を通じて、この構造には根本的な問題があることが判明した。

#### 5.5.1 2層構造の問題点

言語は**随伴の要件を満たさない**可能性がある。随伴 F ⊣ G が成立するためには、以下の条件が必要である：

1. **双方向性**: F(x) から y への写像と、y から G(y) への逆写像が両方機能する
2. **構造保存**: 写像が元の対象の本質的構造を保存する
3. **自然同型**: Hom(F(x), y) ≅ Hom(x, G(y)) が成立する

Action ⇄ Language の関係を考えると：

- **「持ち上げる」という言語** → どの動作？（片手？両手？持ち方は？）
- **「持ち上げる」という動作** → どの言語？（「lift」？「pick up」？「raise」？）

言語は動作よりも**抽象的**であり、身体的詳細を捨象する。この非対称性は、単純な随伴では扱えない。

#### 5.5.2 中間層としての目的空間P

この問題を解決するため、**3層構造**が必要である：

> **Shape ⇄ Action → Purpose (P) → Language**

ここで：

1. **身体的随伴（Shape ⇄ Action）**: 物理的世界との直接的な相互作用
2. **目的空間P**: 身体的経験から抽出された価値判断の軸
3. **言語**: Pの上に構築される記号系

目的空間Pは、身体的随伴と言語の間の**接地層**として機能する。

#### 5.5.3 Pを介した言語接地

言語は、直接動作に接地するのではなく、**目的空間Pに接地**する：

| 身体的経験 | 目的空間P | 言語 |
| :--- | :--- | :--- |
| 「この袋を持ち上げる」（具体的動作） | 「物を移動させる」（目的） | 「運ぶ」 |
| 「スイッチを押す」（具体的動作） | 「状態を変える」（目的） | 「つける」 |
| 「ナイフで切る」（具体的動作） | 「分離する」（目的） | 「切る」 |

言語は、具体的な身体動作ではなく、**その動作が達成しようとする目的**を指示する。これにより：

- 言語は身体的詳細から解放される（「運ぶ」は持ち方を指定しない）
- 同じ目的に対して複数の身体的実現が可能（「運ぶ」は持ち上げでも転がしでもよい）
- 言語は状況依存性が低い（「運ぶ」は疲れていても急いでいても同じ言葉）

#### 5.5.4 実装への展望

現在の実装（Agent C v4）は、目的空間Pの基礎を提供する。言語接地への拡張（Phase 3）では：

1. **Pから言語への写像**: Valenceパターンを言語トークンにマッピング
2. **言語からPへの逆写像**: 言語指示から目的パターンを推定
3. **Pを介した言語理解**: 「運ぶ」→ 「物を移動させる」（P） → 状況に応じた具体的動作

この3層構造により、言語は単なるラベルではなく、**身体的経験から抽出された目的構造に接地した意味を持つ記号系**として機能する。これは、記号接地問題の本質的な解決への道筋を開く。

---

## 6. Active Inferenceとの対応関係と本モデルの独自性

### 6.1 構造的対応

本モデルとカール・フリストンが提唱するActive Inference（能動的推論）の間には、強い構造的対応が存在する [2]。

| 本モデルの概念 | Active Inferenceの概念 |
| :--- | :--- |
| Coherence signal (η) | 自由エネルギー（予測誤差） |
| F⊣Gの動的ループ | Action-Perception loop |
| エージェント状態 C | 生成モデルのパラメータ |
| Coherence breakdown | サプライズ（予測外の観測） |

特に、coherence signal ≈ 自由エネルギーという対応は本質的である。G(F(shape))は「エージェントが現在の内部状態Cと形状shapeから予測する世界のあり方」であり、distance(shape, G(F(shape)))はその予測と現実の乖離、すなわち予測誤差に他ならない。

### 6.2 本モデルの独自性

しかし、本モデルはActive Inferenceの単なる再定式化ではない。以下の概念はActive Inferenceの枠組みでは直接扱えない。

**保留構造**は、Active Inferenceにおける「生成モデル」とは質的に異なる。生成モデルは世界の確率的な記述であるのに対し、保留構造は「まだ決まっていない」という未確定性そのものを構造化したものである。

**Coherence breakdownが創造性を生む**という主張は、Active Inferenceにおけるサプライズ最小化の原理とは方向性が異なる。Active Inferenceではサプライズ（予測誤差）は最小化すべきものであるが、本モデルではcoherence breakdownは創造的問題解決の契機として積極的に位置づけられる。

**随伴の非対称性**——ShapeはCに依存しないがActionはCに依存する——は、Active Inferenceの対称的な枠組みでは明示的に扱われていない。

**目的空間Pと内発的動機**は、本モデルの独自の貢献である。Active Inferenceではエージェントは**驚き（予測誤差）を最小化**するが、本モデルではエージェントは**創造的ポテンシャル（coherence × uncertainty × valence）を最大化**する。この違いは本質的であり、以下の帰結をもたらす：

- **破綻への態度**: Active Inferenceは破綻を回避するが、本モデルは「生産的な破綻」を探索する
- **経験の蓄積**: Valenceは過去の経験から「どの種類の破綻が生産的か」を学習する
- **自分を保つ**: 目的空間Pはエージェントが時間的持続性を獲得するための基盤となる

### 6.3 位置づけ

本モデルは、**「圏論という代数的な言語でActive Inferenceを再定式化し、さらに保留構造と創造性の概念で拡張したもの」**として位置づけることができる。Active Inferenceがベイズ確率論を基盤とするのに対し、本モデルは随伴関手という構造そのものに焦点を当てる。これにより、確率的な定式化が難しい概念——保留、未確定性、創造性——をより直接的に扱える可能性がある。

なお、Smithe (2024) による**Structured Active Inference** [3] は、圏論的システム理論を用いてActive Inferenceを大幅に一般化した研究であり、本モデルの最も近い先行研究である。特に、「エージェントの状態によって利用可能な行動が変わる」というmode-dependenceの概念は、本モデルにおけるパラメータ付き随伴のCに直接対応する。

---

## 7. 実装への展望：理論を検証するためのプロトタイプ

本モデルの理論的枠組みを実証するため、以下の段階的なプロトタイプ実装と実験計画を提案する。実装の目標は、単なる機能の再現ではなく、**記号接地問題の解決と創造性の創発という理論の核心を検証すること**にある。

### 7.1 技術要素の対応表

| 理論コンポーネント | 実装方針 | ツール・先行研究 |
| :--- | :--- | :--- |
| Shape圈 | 3Dオブジェクトの点群またはメッシュ | ShapeNet [4], PartNet |
| Action圈 | 身体部位とオブジェクト表面点の関係性の確率分布 | ZSP3A [5], Contact-GraspNet [6] |
| 関手 F | GNNによるアフォーダンス予測 | PyTorch Geometric [7] |
| 関手 G | GNNによる逆推論 | 条件付きデコーダ（Andries et al. 2020 [8] 参照） |
| FとGの条件付け | FiLM (Feature-wise Linear Modulation) | Perez et al. 2018 |
| Coherence signal | 再構成誤差 distance(shape, G(F(shape))) | 自己教師あり学習 |
| エージェント状態 C | RSSM (Recurrent State-Space Model) | DreamerV3 [9] 参照 |
| 目的空間 P | Valence Memory + Intrinsic Reward | Agent C v4（本実装） |
| Valence更新則 | 内発的報酬（好奇心・有能感・新奇性） | Active Inference + 内発的動機研究 |
| Priority原理 | coherence × uncertainty × valence | 本研究の独自貢献 |
| 価値関数 V(state) | TD学習による期待将来報酬の推定 | 強化学習の標準手法 |
| Agent Cの訓練 | 内発的報酬最大化（F/G凍結） | 本実装（value-based learning） |

### 7.2 実験デザインと評価指標

理論の核心を検証するため、以下の3段階の実験設定を提案する。

**Phase 0: 基礎学習と再現性検証**
-   **目的**: 既知のオブジェクトと動作のペアにおいて、随伴構造（F⊣G）が学習可能であることを示す。
-   **設定**: ShapeNetの既知カテゴリ（椅子、カップ）と、それに対応する把持姿勢のペアデータで学習。
-   **評価指標**: 再構成誤差の最小化、FとGの予測精度。

**Phase 1: 未知オブジェクトへの般化と記号接地問題の解決（設定A）**
-   **目的**: 学習時に見たことのないオブジェクトに対し、そのアフォーダンスを創発的に推論し、適切な動作を生成できることを示す。
-   **設定**: シミュレーション環境（例：PyBullet）で、シンプルな形状（立方体、円柱、その組み合わせ）と基本動作（押す、引く、持ち上げる）で学習。テスト時には、学習時に見せなかった未知の形状組み合わせ（例：立方体＋円柱の配置）を提示。
-   **評価指標**: 未知形状に対するアフォーダンス推論の成功率、生成された動作の適切性。Coherence signalが安定しているか。

**Phase 2: 制約下での創造的解決とCoherence Breakdownの検証（設定B）**
-   **目的**: 既存の随伴構造が破綻するような制約が与えられた際、保留構造が起動し、新しい随伴構造（動作）を創発できることを示す。
-   **設定**: Phase 1の環境に、エージェントの状態Cに影響を与える制約（例：重力の変化、接触面の摩擦係数の変化、エージェントの身体の一部が使用不能になる）を追加。これによりCoherence signalが急増する状況を作り出す。
-   **評価指標**: Coherence signalの増大（breakdown）と、その後の新しい動作の創発。創発された動作が制約を克服し、タスクを達成できるか。保留構造（エージェント層C）の内部状態の変化。

**Phase 3: 言語とのアラインメント（設定C）**
-   **目的**: 創発されたアフォーダンスや動作に対して、エージェントが言語的な記述を生成できることを示す。これにより、記号接地問題の最終的な解決を目指す。
-   **設定**: Phase 2で創発された動作に対して、「これは〇〇という動作である」という言語ラベルを生成させる。または、言語指示から新しい動作を生成させる。
-   **評価指標**: 生成された言語記述の適切性、言語指示に対する動作生成の成功率。

---

## 8. 結論：保留構造は設計しない

本研究ノートは、物理的意味的随伴モデルの理論的枠組みを、知性の本質としての保留構造を中心に探求してきた。その過程で、理論は自己矛盾を乗り越え、よりシンプルで強力な形へと進化した。最終的な結論は、当初の想定を覆すものである。

**保留構造は、明示的に設計する対象ではない。それは、随伴層とエージェント層の動的な相互作用から創発する、系の全体的な性質である。**

我々が設計すべきは、保留構造そのものではなく、**保留構造が創発するための条件**である。具体的には、以下の3つを設計することに尽きる。

1.  **随伴構造（F⊣G）**: 世界の構造的対応関係を捉えるための基本的な器。
2.  **Coherence Signal**: 随伴構造の破綻を検知するための内部指標。
3.  **エージェント層の更新則**: Coherence Signalに応じて内部状態を更新し、随伴構造にフィードバックをかけるループ。

この3つの要素を備えたシステムを構築し、適切な環境と相互作用させることで、保留構造の4つの要件（志向性、差異への感受性、時間的持続、自己と非自己の区別）は自ずと満たされる。知性は、この動的な平衡状態そのものとして現れるだろう。

この洞察は、AI研究における設計思想の転換を促す。すなわち、「知能を構成要素に分解して実装する」というアプローチから、「知能が創発する生態系を設計する」というアプローチへの移行である。本モデルは、そのための具体的な理論的・実装的枠組みを提供するものである。

---

## 参考文献

[1] Harnad, S. (1990). The symbol grounding problem. *Physica D: Nonlinear Phenomena*, 42(1-3), 335-346.

[2] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural computation*, 29(1), 1-49.

[3] Smithe, D. (2024). Structured Active Inference. *arXiv preprint arXiv:2401.00345*.

[4] Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., ... & Savarese, S. (2015). Shapenet: An information-rich 3d model repository. *arXiv preprint arXiv:1512.03012*.

[5] Gkanatsios, N., Pfrommer, J., & Daniilidis, K. (2023). Zero-Shot Policy Synthesis for Physical-Semantic Affordances. *arXiv preprint arXiv:2310.09582*.

[6] Brahmbhatt, S., Ham, C., & Hays, J. (2020). Contact-graspnet: Efficient 6-dof grasp generation in the wild. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 13677-13687).

[7] Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.

[8] Andries, M., Kurenkov, V., & Beetz, M. (2020). A framework for robotic agents to learn and reason with affordances. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 9419-9426). IEEE.

[9] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.

[10] Çatal, O., Verbelen, T., De Boom, C., & Dhoedt, B. (2020). Grounding symbols in multi-modal representations. *arXiv preprint arXiv:2005.03373*.
# リサーチノート：物理的・意味的随伴モデルの技術要件

## 1. Categorical Deep Learning (Gavranović et al., 2024, ICML)
- 圏論（特にモナドの普遍代数、パラメトリック写像の2-圏）をDLアーキテクチャの統一理論として提案
- 制約の指定と実装の指定の間の「橋」を構築することが目的
- Geometric Deep Learningの制約やRNN等の実装を回復できることを示した
- 随伴そのものよりもモナド代数の準同型としてNN層を定式化
- **関連性**: アーキテクチャ設計に圏論を使う先行事例。ただし身体性やアフォーダンスは扱っていない

## 2. Structured Active Inference (Smithe, 2024)
- Active Inferenceを圏論的システム理論で大幅に一般化・形式化
- 生成モデルを「インターフェース上のシステム」として定式化
- エージェントは生成モデルの「コントローラー」であり、形式的に双対
- mode-dependence: 利用可能な行動が文脈データ（現在の状態）に依存
- meta-agents: Active Inferenceを使って自身の構造を変更するエージェント
- 型付きポリシー（formal verification可能）
- **極めて関連性が高い**: 
  - 「エージェントの状態によって利用可能な行動が変わる」= パラメータ付き随伴のCに相当
  - 「meta-agents」= 保留構造の具体化に近い
  - 圏論的システム理論がActive Inferenceに適用できることを示している

## 3. Active Inference / Free Energy Principle (Friston)
- 自由エネルギー最小化 = 予測誤差の最小化
- coherence signalとの類似: 予測誤差 ≈ η(unit)の大きさ
- アフォーダンスをActive Inferenceの枠組みで扱う研究あり (Friston 2012)
- **重要な接点**: 
  - Free Energy ≈ coherence signal（内部整合性の指標）
  - Active Inferenceのaction-perception loop ≈ F⊣Gの動的ループ
  - ただしActive Inferenceはベイズ推論ベース、随伴モデルは圏論ベース

## 4. Affordance Learning in Embodied AI
- Multi-Object Graph Affordance Network: GNNでオブジェクト間のアフォーダンスをモデル化
- SceneFun3D: 3Dシーンにおける細粒度のアフォーダンス理解
- ContactGrasp等: 形状からグリップ動作を推論するデータセット・手法
- **実装基盤として利用可能**: Shape→Actionの写像Fの具体的な実装先行事例

## 5. 圏論×Active Inference の接点
- Smithe (2024): polynomial functorsとsheafを使ったマルチエージェントActive Inference
- Sean Tull: string diagramsによるActive Inferenceの圏論的記述
- **随伴モデルとActive Inferenceの統合可能性を示唆**

## 重要な発見
- **Structured Active Inference**は、随伴モデルの最も近い先行研究
- 違い: Active Inferenceはベイズ推論（確率的）、随伴モデルは代数的構造（決定論的に近い）
- 随伴モデルの独自性: 「保留構造」「coherence breakdown = 創造性」はActive Inferenceにない概念
- 実装の橋渡し: Active Inferenceの既存実装（pymdp等）を参考にできる可能性

## 6. 3D Affordance データセット・表現
- **3D AffordanceNet**: 23Kオブジェクト形状、18種のアフォーダンスラベル付き
- **PartNet**: 大規模3Dオブジェクトの細粒度パーツアノテーション（機能的パーツ分割）
- **ZSP3A (Kim et al., 2024)**: ゼロショットで3Dアフォーダンスを生成。形状→人間のインタラクション姿勢を推論。
  - 重要: アフォーダンス表現を「密な人間点とオブジェクト点の間の相対的な向きと近接度」として定義
  - これはShape→Actionの写像Fの具体的な表現形式として使える
- **Contact-GraspNet**: 深度画像から6-DoFグラスプを直接生成
- **SceneFun3D**: 3Dシーンにおける細粒度の機能性・アフォーダンス理解

## 7. 実装ツール候補
- **pymdp**: Active Inferenceのpython実装（POMDP、離散状態空間）
- **PyTorch Geometric**: GNNライブラリ（グラフ構造の学習に最適）
- **ShapeNet**: 51K以上の3Dモデルデータセット
- **Isaac Gym / MuJoCo**: ロボティクスシミュレーション環境

## 8. 追加の重要な発見
- ZSP3Aの「primitive representation」は、オブジェクト表面点と人体点の全ペア間の関係を確率分布として表現
- これは随伴モデルの「ShapeとActionの間の写像」を、具体的な確率的対応として実装する方法を示唆
- Active Inferenceの自由エネルギー ≈ coherence signal という対応は、実装上の損失関数設計に直接使える
