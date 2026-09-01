# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-07-23 17:05 JST
- 作業環境: 自宅Mac（Codex）
- ブランチ: `agent/add-handoff-docs`
- 基準コミット: `6214f2f`

## 今回の目的

研究室PCと自宅Macの間で作業文脈を引き継げるよう、リポジトリ共通のエージェント指示と引き継ぎ文書を追加する。

## 実施した変更

- Codex向けのリポジトリ指示を追加した。
- Claude Code向けに、共通指示と引き継ぎ文書を参照する入口を追加した。
- 長期・短期の引き継ぎ文書テンプレートを `docs/` に追加した。

## 主要な変更ファイル

- `AGENTS.md`: Codex向けの作業開始時、引き継ぎ更新時、push前の共通手順。
- `CLAUDE.md`: Claude Code向けの入口と共通文書への参照。
- `docs/PROJECT_SUMMARY.md`: プロジェクト全体を記録する長期引き継ぎテンプレート。
- `docs/RECENT_WORK.md`: 直近作業を記録する短期引き継ぎ文書。

## 判断と理由

- エージェントが自動検出しやすいよう、`AGENTS.md` と `CLAUDE.md` はリポジトリ直下に配置した。
- 性質の異なる引き継ぎ文書をまとめるため、`PROJECT_SUMMARY.md` と `RECENT_WORK.md` は `docs/` に配置した。
- 指示の二重管理を避けるため、`CLAUDE.md` は詳細を重複させず `AGENTS.md` を正本として参照する構成にした。

## テスト・確認結果

- `git diff --check`: 問題なし。
- 文書のみの変更のため、コードテストは対象外。

## 未完了・既知の問題

- `docs/PROJECT_SUMMARY.md` の各項目はまだ未記入。
- 作業ツリーに今回の変更とは無関係な既存の `.DS_Store` 変更があり、commit対象から除外する。

## 次に行うこと

1. リポジトリの実装と実験結果を確認し、`docs/PROJECT_SUMMARY.md` を記入する。
2. 次回以降のpush前に、実作業に合わせて `docs/RECENT_WORK.md` を置き換える。

## 再開時の注意

- まず `AGENTS.md`、`docs/PROJECT_SUMMARY.md`、`docs/RECENT_WORK.md` を読む。
- `.DS_Store` の変更は意図を確認できるまでcommitしない。
