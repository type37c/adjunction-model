# 目的機構（Module M）の設計案 / Purpose Mechanism (Module M) Design Proposal

## 哲学的基盤 / Philosophical Foundations

本設計は、以下の哲学的理論の調査に基づいている。

### 中心的な洞察 / Central Insight

7つの哲学的伝統を横断して、**目的は明示的な目標表現ではなく、動的な傾向性（tendency）である**という点で収束している。

> 「熟練した活動のエピソードにおいて、熟練した個人は明示的な目標を心に抱いているのではなく、むしろ環境によって誘引（solicit）され、状況への把握（grip）を改善するように行動する。」
> — Bruineberg & Rietveld (2014)

> "During those episodes of skilled activity, the skilled individual does NOT have an explicit goal in mind, but rather is solicited by the environment in such a way as to improve her grip on the situation."
> — Bruineberg & Rietveld (2014)

### 哲学から導出された3つの設計原則 / Three Design Principles from Philosophy

**原則1: 目的は目標表現ではなく動的傾向性である (Purpose as Dynamic Tendency)**

メルロ＝ポンティの「最大把握（maximum grip）」とBruineberg/Rietveldの「最適把握への傾向性（tendency toward optimal grip）」に基づく。目的機構は、明示的な目標を生成するのではなく、エージェントと環境の関係における「最適からの逸脱」を検出し、その逸脱を低減する傾向性を実装すべきである。

**原則2: 目的はランドスケープをフィールドに変換する (Purpose Transforms Landscape into Field)**

Rietveld/Kiversteinの「アフォーダンスのランドスケープ」と「関連するアフォーダンスのフィールド」の区別に基づく。F/Gが出力するアフォーダンスの全体（ランドスケープ）から、現在の関心事に基づいて関連するアフォーダンスのサブセット（フィールド）を選択する機構が必要である。

**原則3: 目的は階層的に構造化されている (Purpose is Hierarchically Structured)**

ハイデガーの「〜のために（in-order-to）→ 〜のための（for-which）→ 〜のために存在する（for-the-sake-of-which）」の参照連関に基づく。異なる時間スケールの行動準備性（action readiness）が、この階層の異なるレベルに対応する。

---

## アーキテクチャ設計 / Architecture Design

### 現在のアーキテクチャの問題 / Problems with Current Architecture

```
点群(N,3) → F → アフォーダンス表現(N,16) → 平均プーリング → Agent C → 行動
                         ↓
                    G → 再構成(N,3)
```

**問題点:**
1. F/Gは形状再構成のみを学習 → アフォーダンスの「ランドスケープ」を捉えていない
2. 平均プーリングが局所構造を消失させる → 「フィールド」への変換が不可能
3. 目的を表現・生成する機構が存在しない → 随伴の「目的を通した」部分が欠落

### 提案するアーキテクチャ / Proposed Architecture

```
                        ┌─────────────────────────────┐
                        │     Module M (目的機構)       │
                        │   Purpose Mechanism          │
                        │                              │
                        │  ┌─────────┐  ┌──────────┐  │
                        │  │ Concern │  │ Grip     │  │
                        │  │ State   │←→│ Monitor  │  │
                        │  │ (関心)   │  │ (把握度)  │  │
                        │  └────┬────┘  └─────┬────┘  │
                        │       │              │       │
                        │       └──────┬───────┘       │
                        │              │               │
                        │       ┌──────▼──────┐        │
                        │       │ Salience    │        │
                        │       │ Modulator   │        │
                        │       │ (顕著性変調) │        │
                        │       └──────┬──────┘        │
                        └──────────────┼───────────────┘
                                       │
                                       ▼ g (goal/concern vector)
                                       │
点群(N,3) → F(x, g) → アフォーダンス表現(N,d) → 注意機構 → Agent C → 行動
                ↓                                    ↑
           G → 再構成(N,3)                          │
                ↓                                    │
           η(再構成誤差) ──────────────────────────────┘
                ↓
           Grip Monitor へフィードバック
```

### Module M の3つのサブコンポーネント / Three Subcomponents of Module M

#### 1. Concern State（関心状態）/ 哲学的対応: ハイデガーの「〜のために存在する」

**哲学的根拠**: ハイデガーの参照連関（Verweisungszusammenhang）において、すべての道具的存在は最終的に「〜のために存在する（Worumwillen）」に帰着する。エージェントの関心事がなければ、アフォーダンスは「関連する」ものとして現れない。

**設計**: Concern Stateは、エージェントの現在の「関心事」を表現する潜在ベクトルである。これは明示的な目標（「カップを持ち上げろ」）ではなく、より抽象的な傾向性（「環境との相互作用を深めたい」「新しい可能性を探索したい」）を表現する。

```python
class ConcernState(nn.Module):
    """
    哲学的対応 / Philosophical Correspondence:
    - Heidegger: Worumwillen (for-the-sake-of-which)
    - Enactivism: Autopoietic self-maintenance concern
    - Bratman: Partial plans as commitment structures
    
    Concern Stateは固定された目標ではなく、
    エージェントの経験に基づいて動的に更新される。
    Bratmanの「部分的計画」のように、
    状況に応じて埋められる未完成の構造である。
    """
    def __init__(self, concern_dim, hidden_dim):
        super().__init__()
        # 関心状態は時間的に持続する（GRUで更新）
        # Concern state persists over time (updated via GRU)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=concern_dim)
        
        # 初期関心は「自己維持」に相当する基本的な傾向性
        # Initial concern corresponds to basic autopoietic tendency
        self.base_concern = nn.Parameter(torch.randn(concern_dim))
    
    def forward(self, feedback, prev_concern=None):
        """
        feedback: Grip Monitorからのフィードバック
        prev_concern: 前のタイムステップの関心状態
        """
        if prev_concern is None:
            prev_concern = self.base_concern.unsqueeze(0).unsqueeze(0)
        
        # 関心状態を更新（Thompsonの「進行中であること」に対応）
        # Update concern state (corresponds to Thompson's "being in progress")
        new_concern, _ = self.gru(feedback.unsqueeze(0), prev_concern)
        return new_concern.squeeze(0)
```

#### 2. Grip Monitor（把握度モニター）/ 哲学的対応: メルロ＝ポンティの「最大把握」

**哲学的根拠**: メルロ＝ポンティは、身体が常に「最大把握（maximum grip）」に向かう傾向性を持つと論じた。Bruineberg/Rietveldはこれを自由エネルギー最小化と接続した。把握度モニターは、エージェントの現在の「把握度」を評価し、最適からの逸脱を検出する。

**設計**: Grip Monitorは、ηの変化パターン、行動の結果、環境の状態変化を統合して、「現在の把握度」を評価する。把握度が低い（逸脱が大きい）場合、Concern Stateの更新を促進し、Salience Modulatorの感度を上げる。

```python
class GripMonitor(nn.Module):
    """
    哲学的対応 / Philosophical Correspondence:
    - Merleau-Ponty: Maximum grip / optimal distance
    - Bruineberg/Rietveld: Free energy as dis-attunedness
    - Di Paolo: Adaptivity (evaluating better/worse conditions)
    
    把握度は「最適からの逸脱」として計算される。
    逸脱が大きい = 把握度が低い = 関心の更新が必要。
    これはBruineberg/Rietveldの
    「自由エネルギー = 不調和度」に対応する。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # ηの変化、行動結果、環境状態を統合
        # Integrates η changes, action results, and environment state
        self.integrator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 把握度の履歴を保持（時間的文脈）
        # Maintains grip history (temporal context)
        self.grip_history = nn.GRU(hidden_dim, hidden_dim)
    
    def forward(self, eta, delta_eta, action_result, grip_history=None):
        """
        eta: 現在の再構成誤差（不調和度の指標）
        delta_eta: ηの変化量（把握度の改善/悪化）
        action_result: 行動の結果（環境からのフィードバック）
        
        Returns:
            grip_deviation: 最適からの逸脱度（スカラー）
            grip_feedback: Concern Stateへのフィードバック（ベクトル）
        """
        combined = torch.cat([eta, delta_eta, action_result], dim=-1)
        grip_state = self.integrator(combined)
        
        if grip_history is not None:
            grip_state, new_history = self.grip_history(
                grip_state.unsqueeze(0), grip_history
            )
            grip_state = grip_state.squeeze(0)
        else:
            new_history = None
        
        return grip_state, new_history
```

#### 3. Salience Modulator（顕著性変調器）/ 哲学的対応: SIFの「フィールド」

**哲学的根拠**: Rietveld/Kiversteinの「関連するアフォーダンスのフィールド」は、ランドスケープ全体から現在の関心事に基づいて「際立つ」アフォーダンスのサブセットである。Gibsonの「招待（invitation）」概念も同様に、アフォーダンスが主観的に関連性を持つ場合にのみ行動を誘引することを示す。

**設計**: Salience Modulatorは、Concern Stateとアフォーダンス表現を受け取り、各アフォーダンスの「顕著性（salience）」を計算する。これにより、F(x)のランドスケープがAgent Cにとっての「フィールド」に変換される。

```python
class SalienceModulator(nn.Module):
    """
    哲学的対応 / Philosophical Correspondence:
    - Rietveld/Kiverstein: Landscape → Field transformation
    - Gibson/Reed: Affordance → Invitation transformation
    - Heidegger: Circumspection (Umsicht) — looking around
      to take in relevant references
    
    顕著性変調は、注意機構（Attention）として実装される。
    Concern Stateがクエリ、アフォーダンス表現がキー/バリューとなる。
    これにより、目的に応じて異なるアフォーダンスが「際立つ」。
    """
    def __init__(self, concern_dim, affordance_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=affordance_dim,
            num_heads=num_heads,
            kdim=affordance_dim,
            vdim=affordance_dim
        )
        # Concern Stateをクエリ空間に射影
        # Project Concern State into query space
        self.concern_to_query = nn.Linear(concern_dim, affordance_dim)
    
    def forward(self, concern_state, affordance_landscape):
        """
        concern_state: 現在の関心状態 (B, concern_dim)
        affordance_landscape: F(x)の出力 (B, N, affordance_dim)
        
        Returns:
            field: 関連するアフォーダンスのフィールド (B, affordance_dim)
            salience_weights: 各点の顕著性重み (B, N)
        """
        # Concern Stateをクエリに変換
        query = self.concern_to_query(concern_state)  # (B, affordance_dim)
        query = query.unsqueeze(0)  # (1, B, affordance_dim)
        
        # アフォーダンスランドスケープをキー/バリューに
        key = affordance_landscape.permute(1, 0, 2)  # (N, B, affordance_dim)
        value = key
        
        # 注意機構でフィールドを生成
        field, salience_weights = self.attention(query, key, value)
        
        return field.squeeze(0), salience_weights.squeeze(1)
```

---

## F/Gの改修 / Modifications to F/G

### FunctorF の改修: 目的条件付き + 近傍構造

```python
class FunctorF_v2(nn.Module):
    """
    改修点 / Modifications:
    1. 近傍構造の導入（点ごとの独立処理 → 局所構造を捉える）
    2. 目的ベクトルgによる条件付け（将来的にModule Mと接続）
    
    哲学的対応:
    - Heidegger: 道具は孤立して存在しない。
      常に他の道具との参照連関の中にある。
      → 点も孤立して処理すべきではない。近傍との関係が重要。
    - Heidegger: 道具の存在は「〜のために」によって構成される。
      → 目的gがアフォーダンス表現を構成する。
    """
    def __init__(self, input_dim=3, affordance_dim=32, goal_dim=16, k=16):
        super().__init__()
        self.k = k  # 近傍点の数
        
        # 局所特徴抽出（近傍構造を考慮）
        # Local feature extraction (considering neighborhood structure)
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 点 + 近傍との相対位置
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 目的による条件付け（FiLM変調）
        # Goal conditioning (FiLM modulation)
        self.goal_to_gamma = nn.Linear(goal_dim, 128)
        self.goal_to_beta = nn.Linear(goal_dim, 128)
        
        # アフォーダンス表現への射影
        # Projection to affordance representation
        self.to_affordance = nn.Sequential(
            nn.Linear(128, affordance_dim),
            nn.ReLU()
        )
    
    def forward(self, pos, goal=None):
        """
        pos: 点群座標 (B, N, 3)
        goal: 目的ベクトル (B, goal_dim) — Module Mから。Noneの場合は無条件。
        """
        # 近傍探索
        # k-nearest neighbor search
        dists = torch.cdist(pos, pos)  # (B, N, N)
        _, idx = dists.topk(self.k, dim=-1, largest=False)  # (B, N, k)
        
        # 近傍との相対位置を計算
        # Compute relative positions to neighbors
        neighbors = torch.gather(
            pos.unsqueeze(2).expand(-1, -1, pos.size(1), -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, N, k, 3)
        relative = neighbors - pos.unsqueeze(2)  # (B, N, k, 3)
        
        # 局所特徴: 点の座標 + 近傍との相対位置の平均
        local_context = relative.mean(dim=2)  # (B, N, 3)
        local_input = torch.cat([pos, local_context], dim=-1)  # (B, N, 6)
        
        features = self.local_encoder(local_input)  # (B, N, 128)
        
        # 目的による条件付け（FiLM変調）
        # Goal conditioning via FiLM modulation
        if goal is not None:
            gamma = self.goal_to_gamma(goal).unsqueeze(1)  # (B, 1, 128)
            beta = self.goal_to_beta(goal).unsqueeze(1)    # (B, 1, 128)
            features = gamma * features + beta
        
        affordances = self.to_affordance(features)  # (B, N, affordance_dim)
        return affordances
```

---

## 随伴との接続 / Connection to Adjunction

### 随伴の再解釈 / Reinterpretation of Adjunction

Module Mの導入により、随伴の構造は以下のように再解釈される。

**元の随伴**: F ⊣ G（物理空間 ⇄ アフォーダンス空間）

**拡張された随伴**: 目的gを通した随伴

```
物理空間 P ──F(·,g)──→ アフォーダンス空間 A
                              │
                        Salience Modulator
                              │
                              ▼
                     フィールド空間 Φ ──Agent C──→ 行動空間 Act
                              │                        │
                              │                        ▼
                     Grip Monitor ←──────────── 環境変化
                              │
                              ▼
                     Concern State 更新 → 新しい g
```

**哲学的解釈**: 

ハイデガーの用語で言えば、F(·,g)は「配視（Umsicht）」に相当する。配視とは、道具連関の中で「〜のために」の参照を見渡すことである。目的gが変われば、同じ物理的対象から異なるアフォーダンスが「見える」。

メルロ＝ポンティの用語で言えば、このループ全体が「志向弧（intentional arc）」に相当する。過去の経験（Concern Stateの履歴）、現在の状況（F(x,g)の出力）、未来の可能性（Agent Cの行動選択）が統合されている。

Bruineberg/Rietveldの用語で言えば、η（再構成誤差）は「不調和度（dis-attunedness）」に相当し、Grip Monitorがこれを評価して「最適把握への傾向性」を実装する。

### 単位（unit）と余単位（counit）の再定義 / Redefinition of Unit and Counit

**単位 η: P → G(F(P, g))**

物理的対象Pを、目的gのもとでアフォーダンス空間に写し、再び物理空間に戻す。この「往復」の誤差が、「この目的のもとで、この対象がどれだけ理解されているか」を測定する。

**余単位 ε: F(G(A), g) → A**

アフォーダンス表現Aから物理的対象を再構成し、それを再びアフォーダンス空間に写す。この「往復」の誤差が、「このアフォーダンス表現がどれだけ物理的に実現可能か」を測定する。

---

## 訓練戦略 / Training Strategy

### Phase 1.5: F/Gの再訓練（目的条件付き）

1. PyBulletで「形状 + 行動 → 結果」のデータを収集
2. F(x, g)を訓練: gは行動の種類を表すワンホットベクトル（初期段階）
3. 再構成損失 + 行動予測損失で訓練
4. Step 1を再実行してηの改善を確認

### Phase 2.1: Module Mの統合

1. Module M（Concern State + Grip Monitor + Salience Modulator）を実装
2. Agent Cと統合してend-to-endで訓練
3. タスク報酬を使用（ただし、Grip Monitorの出力を内発的報酬として併用）
4. 段階的にタスク報酬の比重を下げ、内発的報酬の比重を上げる

### 損失関数 / Loss Function

```
L = λ_recon · L_recon     # 再構成損失（F/Gの基本能力維持）
  + λ_task  · L_task      # タスク損失（Agent Cの行動学習）
  + λ_grip  · L_grip      # 把握度損失（Grip Monitorの校正）
  + λ_div   · L_diversity # 多様性損失（Concern Stateの崩壊防止）
```

---

## 哲学的理論との対応表 / Correspondence Table

| コンポーネント | 哲学的概念 | 出典 | 機能 |
|---|---|---|---|
| Concern State | Worumwillen（〜のために存在する） | Heidegger | エージェントの根本的な関心事を表現 |
| Concern State | 部分的計画 | Bratman | 状況に応じて埋められる未完成の目的構造 |
| Concern State | 自己維持の関心 | Di Paolo | 基本的な傾向性としての自己維持 |
| Grip Monitor | 最大把握 / 最適把握 | Merleau-Ponty / Bruineberg | 最適からの逸脱を検出 |
| Grip Monitor | 不調和度（自由エネルギー） | Bruineberg/Rietveld | η ≈ 自由エネルギー ≈ 不調和度 |
| Grip Monitor | 適応性 | Di Paolo | より良い/悪い条件の評価 |
| Salience Modulator | フィールド（関連するアフォーダンス） | Rietveld/Kiverstein | ランドスケープからフィールドへの変換 |
| Salience Modulator | 招待（invitation） | Gibson/Reed | アフォーダンスを行動誘引に変換 |
| Salience Modulator | 配視（Umsicht） | Heidegger | 参照連関の中で関連する道具を見渡す |
| F(x, g) | 条件付きアフォーダンス知覚 | Heidegger | 目的によってアフォーダンスの現れ方が変わる |
| F(x, g) の近傍構造 | 道具連関（Zeugzusammenhang） | Heidegger | 道具は孤立せず、他の道具との関係の中にある |
| η | 不調和度 / 逸脱 | Bruineberg/Rietveld | エージェント-環境の調和度の指標 |
| Agent C | 運動志向性（motor intentionality） | Merleau-Ponty | 「私はできる（I can）」としての身体的行動 |
| 訓練ループ全体 | 志向弧（intentional arc） | Merleau-Ponty | 過去・現在・未来の統合 |
| エピソードリセット | 道具の故障（breakdown） | Heidegger | 手前存在から眼前存在への移行 → リセット |

---

## 実装の優先順位 / Implementation Priority

### 即座に実装可能 / Immediately Implementable
1. **FunctorF_v2**: 近傍構造の導入（goal=Noneで現在と互換）
2. **Salience Modulator**: 注意機構による平均プーリングの置換

### Phase 1.5で実装 / Implement in Phase 1.5
3. **FunctorF_v2のgoal条件付け**: PyBulletデータでの再訓練
4. **Grip Monitor**: ηの変化パターンの評価

### Phase 2.1で実装 / Implement in Phase 2.1
5. **Concern State**: Agent Cとの統合訓練
6. **Module M全体の統合**: end-to-end訓練

---

## 未解決の問題 / Open Questions

1. **Concern Stateの初期化**: 自己維持の「基本的な関心」をどう表現するか？エナクティヴィズムの自己産出（autopoiesis）に対応するが、人工システムでは「自己維持」の意味が不明確。

2. **Grip Monitorの校正**: 「最適な把握」の基準をどう設定するか？教師なしで学習可能か？

3. **Concern Stateの崩壊**: 訓練中にConcern Stateが一つの固定パターンに崩壊する可能性。多様性損失で防げるか？

4. **エピソード間のConcern State**: エピソードリセット時にConcern Stateもリセットすべきか？ハイデガーの「故障」概念に基づけば、リセットは「手前存在から眼前存在への移行」に相当し、新しい理解の契機となる。

5. **随伴の三角恒等式**: Module Mの導入後も、随伴の三角恒等式（triangle identities）が成立するか？成立しない場合、随伴の理論的枠組みをどう修正するか？
