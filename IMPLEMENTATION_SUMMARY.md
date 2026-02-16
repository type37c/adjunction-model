# Phase 2.5 Valence Role Experiment: Implementation Summary

**Date**: 2026-02-16  
**Status**: ✅ Implementation Complete

## Overview

Phase 2.5 Valence Role Experimentの実装が完了しました。このプロジェクトは、2026年2月16日の議論で明らかになった「Phase 2 Slackにおいてvalenceとpriorityが訓練を駆動していなかった」という発見に基づき、valenceの真の効果を測定するための実験です。

## Implemented Components

### 1. Core Modules

| File | Description | Status |
|:---|:---|:---|
| `src/models/priority_v2.py` | Priority計算（valenceなし版） | ✅ Complete |
| `src/models/priority_v3.py` | Priority計算（valence対応版） | ✅ Complete |
| `src/models/valence_v3.py` | ValenceMemoryV3（行動とΔηの記憶） | ✅ Complete |
| `src/models/agent_c_v3.py` | AgentCV3（創発的valence使用） | ✅ Complete |

### 2. Experiment Scripts

| File | Description | Status |
|:---|:---|:---|
| `experiments/run_valence_experiment.py` | 実験実行スクリプト（3条件） | ✅ Complete |
| `experiments/analyze_valence_experiment.py` | 結果分析スクリプト | ✅ Complete |
| `experiments/phase2_valence_experiment/README.md` | 実験ドキュメント | ✅ Complete |

### 3. Documentation

| File | Description | Status |
|:---|:---|:---|
| `NEW_PLAN.md` | 開発計画 | ✅ Complete |
| `TODO.md` | Phase 2.5セクション追加 | ✅ Complete |
| `docs/theory/priority_and_valence_reconsidered.md` | 理論的背景 | ✅ Complete |

## Experimental Design

### Three Conditions

| Condition | Implementation | Key Difference |
|:---|:---|:---|
| **Condition 1: Baseline** | `alpha_curiosity=0.0` | Valence更新なし（Phase 2 Slack再現） |
| **Condition 2: Emergent** | `AgentCV3` | Valenceを直接RSSMに入力、使い方は創発 |
| **Condition 3: Designed** | `alpha_curiosity=1.0` | Priority = coherence × uncertainty × valence |

### Research Questions

1. **Condition 1 vs 3**: Valenceは訓練を改善するか？
2. **Condition 2 vs 3**: 創発的使用は設計的使用より優れているか？

## Key Design Decisions

### 1. ValenceMemoryV3の設計

従来のV2（intrinsic reward based）から、よりシンプルな「行動とSlack変化の記憶」に変更。

```python
# V2: valence = f(R_intrinsic) where R_intrinsic = α*R_curiosity + β*R_competence + γ*R_novelty
# V3: valence = memory of (action, Δη) pairs
```

### 2. AgentCV3の設計

Priority計算モジュールを削除し、coherence、uncertainty、valenceを直接RSSMに入力。

```python
# Extended observation: [coherence_scalar, uncertainty, original_obs]
# Context: [h, z, valence] → context_net → context for F and G
```

### 3. 実験の最小変更原則

Phase 2 Slackの安定した枠組みをベースとし、変数を1つ（valence）だけ変更することで、効果測定を明確化。

## Implementation Status

### Completed ✅

- [x] Priority v2/v3のリファクタリング
- [x] ValenceMemoryV3の実装
- [x] AgentCV3の実装
- [x] 実験スクリプトの作成
- [x] 分析スクリプトの作成
- [x] ドキュメントの整備

### Pending ⏳

- [ ] **Condition 2の統合**: `AdjunctionModelV3`の作成が必要
  - `AdjunctionModel`を継承し、`AgentC`の代わりに`AgentCV3`を使用
  - `conditional_adjunction.py`の対応する変更
- [ ] **実験の実行**: 3条件×50エポック（CPU で 2-3時間）
- [ ] **結果の分析**: 比較グラフと統計分析
- [ ] **理論文書の更新**: 実験結果に基づく改訂

## Next Steps

### Immediate (Day 1)

1. **AdjunctionModelV3の実装**
   ```python
   # src/models/adjunction_model_v3.py
   class AdjunctionModelV3(AdjunctionModel):
       def __init__(self, ...):
           # Use AgentCV3 instead of AgentC
           self.agent = AgentCV3(...)
   ```

2. **Condition 2の統合**
   - `run_valence_experiment.py`の`run_condition_2()`を実装

### Short-term (Day 2-3)

3. **実験の実行**
   ```bash
   python experiments/run_valence_experiment.py
   ```

4. **結果の分析**
   ```bash
   python experiments/analyze_valence_experiment.py
   ```

### Medium-term (Day 4-5)

5. **ドキュメントの更新**
   - `priority_and_valence_reconsidered.md`に実験結果を追記
   - `TODO.md`のステータス更新

6. **次の実験の計画**
   - 実験結果に基づき、Purpose-Emergent実験の再設計を検討

## Technical Notes

### Compatibility

すべての新しいモジュールは既存のコードベースと互換性があります：

- `priority_v2.py`, `priority_v3.py`: 既存の`priority.py`と同じインターフェース
- `valence_v3.py`: `valence_v2.py`と同じインターフェース
- `agent_c_v3.py`: `agent_c.py`と同じインターフェース

### Testing

PyTorchが未インストールのため、単体テストは実行していませんが、すべてのモジュールに`if __name__ == '__main__':`ブロックでテストコードを含めています。

### Dependencies

- PyTorch
- NumPy
- Matplotlib
- 既存のプロジェクト依存関係

## Theoretical Implications

この実装は、以下の理論的洞察を実験的に検証するためのものです：

1. **損失関数が目的空間として機能する**: Phase 2では`L_aff`だけでSlack管理が成立した
2. **Valenceの時間性**: Valenceだけが時間を跨ぐ構造を持つ
3. **創発 vs 設計**: 軸の使い方を設計するのではなく、創発に委ねるべき

実験結果は、これらの仮説を支持または反証し、プロジェクトの次の方向性を決定します。

---

**実装者**: Manus AI  
**レビュー**: Pending  
**最終更新**: 2026-02-16
