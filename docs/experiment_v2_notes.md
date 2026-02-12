# Prioritization V2 実験結果分析（OnlineLearner無効化版）

## 実験条件
- OnlineLearner: **無効化**（F/G重みは訓練後固定）
- Agent Cへの観測: **有効**（Fの中間特徴量をobsとして渡す）
- 訓練: 10エポック（obsを渡しながら）

## 定量的結果

| 指標 | Cube (Known) | Torus (Novel) | Ratio |
|:---|---:|---:|---:|
| Coherence Signal | 0.1151 | 0.4944 | **4.29x** |
| Attention (‖Δh‖) | 1.1784 | 1.1950 | **1.01x** |
| Priority (mean) | 18.3212 | 3.0314 | **0.17x** |
| Uncertainty | 34.0559 | 34.1487 | 1.00x |
| KL Divergence | 0.0426 | 0.0625 | **1.47x** |

## 改善された点

1. **KL > 0**: Agent Cが「世界を見ている」。posteriorとpriorが分離している。
2. **Coherenceが持続**: Torusの新規性が消えない（4.29x）。OnlineLearnerがF/Gを直接適応させないため。
3. **KL比率1.47x**: 新規形状に対してKLが高い = Agent Cの信念がより大きく修正される。

## 新たに露呈した問題

1. **Attention比率が1.01x（v1の1.11xより悪化）**: Agent Cの内部状態変化が
   形状の違いを反映していない。

2. **Priority比率が逆転（0.17x）**: Cubeの方がPriorityが圧倒的に高い。
   これはCoherenceが低い（0.11）のにPriorityが高い（18.3）という矛盾。
   原因: Cubeの直後にTorusが来るため、Cubeの前のcoherence_spatial_prev（Torusの高い値）が
   Cubeの試行に流入している。

3. **Uncertaintyが依然として1.00x**: RSSMのuncertaintyが形状を区別できていない。

## 根本的な問題の考察

ユーザーの予測通り、「Agent CがF/Gの振る舞いを変えられるほど強くConditioningされているか」
という問題が露呈した。

具体的には:
- FiLMのscale/shiftがAgent Cのcontext vectorから生成されるが、
  context vectorの変化がFiLMパラメータの変化に十分に反映されていない
- Agent Cの内部状態(h)は変化しているが、それがcontextを経由してF/Gの出力を
  有意に変えるほどの影響力を持っていない
- KLは非ゼロだが非常に小さい（0.04-0.09）。Agent Cが観測を受け取っているが、
  それを十分に活用していない

## Priority逆転の原因

priority_i = coherence_i × uncertainty_i

coherence_spatial_prevは「前の形状」の値が渡される。
- Cube試行時: coherence_spatial_prev = Torusの高い空間的coherence → 高いPriority
- Torus試行時: coherence_spatial_prev = Cubeの低い空間的coherence → 低いPriority

つまり、Priorityが「現在の形状」ではなく「前の形状」の情報を反映している。
これは実験設計の問題であり、Priorityの計算自体は正しい。
