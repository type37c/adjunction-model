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
