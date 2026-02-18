# ドキュメント再編成計画 (2026-02-18)

## 1. 現状の問題点

現在、リポジトリ内のドキュメントは複数の異なる視点や開発段階（Slack管理、価値関数ベース、保留構造など）から書かれており、情報が散在・重複・一部陳腐化している。特に、以下の文書間の関係が不明確である。

- **出発点:** `upload/Clipboard_0_E02FBB0F` (物理的意味的随伴モデル Research Note v2.0)
- **方針:** `UPDATED_PROJECT_DIRECTION.md` (3つのPDFを統合したロードマップ)
- **既存:** `README.md`, `ARCHITECTURE.md`, `AGENT_GUIDELINES.md` など

これにより、新規参入者（人間またはAIエージェント）がプロジェクトの全体像と現在の最優先事項を正確に把握することが困難になっている。

## 2. 再編成の目的

- **単一の信頼できる情報源 (Single Source of Truth)** を確立する。
- 情報を**理論（Why）**、**アーキテクチャ（How）**、**ロードマップ（What）**に明確に分離・整理する。
- 古い情報をアーカイブし、現在の開発方針との整合性を取る。
- `README.md` を、プロジェクトの核心ゴールと現在の状況を正確に反映した、最適な入り口として再構築する。

## 3. 新しいドキュメント構造案

### 3.1. トップレベル

- **`README.md` (大幅改訂):**
  - **核心ゴール:** 「未知の形状に対してアフォーダンスを推測できるエージェントの開発」を明確に記述。
  - **核心コンセプト:** 「川床(F/G)と水(Agent C)の二重学習原理」「軌跡としての意味」といった最新の概念を導入。
  - **現状:** Phase A（汎化の基盤確立）の進行中であること、Phase 1実験が実行中であることを記述。
  - **ポインタ:** `docs/` 以下の主要文書への明確なリンクを提供する。

- **`AGENT_GUIDELINES.md` (維持・強化):**
  - 開発の普遍的なルールブックとして維持。
  - 「保留構造は設計しない。創発する条件を設計する」という根本原則を最上部に強調する。

- **`ARCHITECTURE.md` (v3.0へ改訂):**
  - `UPDATED_PROJECT_DIRECTION.md` の内容を完全に統合し、v3.0とする。
  - **二重学習原理:** F/G（損失関数ベース）とAgent C（価値関数ベース）の役割分担を明確に図解。
  - **フィルター機構:** Agent CがF/Gを変調するのではなく、「関わり方を変える」アーキテクチャとして図解。
  - **二重のタイムスケール:** Agent Cの高速な学習とF/Gの低速な学習の概念を導入。
  - **軌跡ベースの認知機能:** 休息の創発、記憶の引き出しといった概念をアーキテクチャに位置づける。

### 3.2. `docs/` ディレクトリ

- **`docs/THEORY.md` (新規作成):**
  - `upload/Clipboard_0_E02FBB0F` (Research Note v2.0) の内容をほぼそのまま格納。
  - プロジェクトの理論的・哲学的背景を記述する「憲法」と位置づける。
  - `SUSPENSION_STRUCTURE_PRINCIPLES.md` と `TEMPORAL_PERSISTENCE_PRINCIPLES.md` の核心的な洞察（5要件、睡眠の必要性）を統合・吸収する。

- **`docs/ROADMAP.md` (新規作成):**
  - `UPDATED_PROJECT_DIRECTION.md` のロードマップ部分（Phase A, B, C）を抽出して格納。
  - `TODO.md` は具体的なタスク管理ファイルとし、`ROADMAP.md` は高レベルな戦略文書と位置づける。

- **`docs/archive/` (整理):**
  - `UPDATED_PROJECT_DIRECTION.md` (ROADMAP.mdとARCHITECTURE.mdに吸収されるため)
  - `SUSPENSION_STRUCTURE_PRINCIPLES.md` (THEORY.mdに吸収されるため)
  - `TEMPORAL_PERSISTENCE_PRINCIPLES.md` (THEORY.mdに吸収されるため)
  - `NEW_PLAN.md`, `THREE_PHASE_TRAINING_PLAN.md` (古いため)
  - その他、現在のアーキテクチャやロードマップと矛盾する古い文書を移動。

## 4. 実行計画

1. **ファイル作成と移動:**
   - `docs/THEORY.md` と `docs/ROADMAP.md` を新規作成。
   - `upload/Clipboard_0_E02FBB0F` の内容を `docs/THEORY.md` に書き込む。
   - `UPDATED_PROJECT_DIRECTION.md` のロードマップ部分を `docs/ROADMAP.md` に書き込む。

2. **内容の統合と改訂:**
   - `ARCHITECTURE.md` を `UPDATED_PROJECT_DIRECTION.md` のアーキテクチャ部分と統合し、v3.0に改訂する。
   - `docs/THEORY.md` に `SUSPENSION_STRUCTURE_PRINCIPLES.md` と `TEMPORAL_PERSISTENCE_PRINCIPLES.md` の内容を統合する。

3. **トップレベルの改訂:**
   - `README.md` を新しい構造に合わせて全面的に書き換える。

4. **アーカイブ:**
   - 役割を終えた古いファイルを `docs/archive/` に移動する。

5. **コミット:**
   - `docs: Reorganize documentation for clarity and consistency` というコミットメッセージで変更をコミットする。
