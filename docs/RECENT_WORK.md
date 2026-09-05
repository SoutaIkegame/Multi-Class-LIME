# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-05（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`（PR #5）
- 基準コミット: `f7cbe43`

## 今回の目的

Fisherを提案から降ろす方針が決まった後、「分析だけでなく新しい手法を出したい」というユーザー依頼を受け、Contrastive LIMEの改良案を4つ考案・実装・評価した（提案A〜C＋組み合わせ）。全て完走。その後「今までの実験を詳細にmdでまとめて」という依頼で`docs/EXPERIMENT_LOG.md`を新規作成した。

## 実施した変更と主要な変更ファイル

1. `src/surrogates.py`: `fit_joint_contrastive`（提案A、MultiTaskLasso）、`fit_ovo_logistic`（提案C、ソフトラベル・ロジスティック回帰）を追加。
2. `src/run_shared_support_experiment.py`: 提案Aの評価列（`joint_lasso`/`joint_refit`）と循環残差指標を追加。
3. `src/run_pair_kernel_experiment.py`（新規）: 提案B「対比認識カーネル」の評価。
4. `src/run_logistic_target_experiment.py`（新規）: 提案Cの評価。
5. `src/run_combined_bc_experiment.py`（新規）: B・Cの組み合わせが積み上がるかの検証。
6. `docs/EXPERIMENT_LOG.md`（新規）: フェーズ1〜11の全実験を、`results/*.csv`から再集計した数値付きで詳細に記録。
7. `docs/PROJECT_SUMMARY.md`: 研究ストーリーの組み替えを「推奨」から「確定方針」に更新し、提案A〜C＋組み合わせの結果を追記。主要な構成に新規スクリプト・`EXPERIMENT_LOG.md`を追記。

## 重要な判断とその理由

1. **提案A（全クラス同時group lasso）は不採用と確定**。真の係数復元・fidelity・安定性の全軸で明確に劣る（`EXPERIMENT_LOG.md`フェーズ8に数値あり）。原因は「全クラスへの共有制約が着目ペアの精度を損なう」という、Fisherの全クラス共通$S_W$と同型の問題。
2. **提案C（OVO logistic）を単体の主軸に据えた**。B・Cは似た改善パターン（全体・穏やかな領域fidelity改善、極端領域無風）だが、Cの方が安定性改善がより一貫している（9/9 vs Bの6/9）。
3. **B+C組み合わせは「指標によって最適解が変わる」という結論にした**。fidelityは積み上がるが安定性は打ち消し合う（`EXPERIMENT_LOG.md`フェーズ11）。単純に「組み合わせが最良」とは言えないため、両方併記する形にした。
4. **`EXPERIMENT_LOG.md`を新設し、PROJECT_SUMMARY.mdとの役割を分離した**。PROJECT_SUMMARY.mdは要約・現状、EXPERIMENT_LOG.mdは各実験の目的・方法・数値・結論の一次記録。数値は全て`results/*.csv`から再集計し、会話中の記憶に頼らず正確性を確保した。

## 実行したテスト・確認結果

- 提案A・B・C・組み合わせの全実験、フルグリッド（9セル×20シード、該当するものはK2水準）で完走（exit code 0）。
- `docs/EXPERIMENT_LOG.md`の全数値は、`results/*.csv`をpandasで再集計して転記（アドホック集計スクリプトで検証済み、恒久スクリプト化はしていない）。

## 未完了・既知の問題・未検証事項

- ソフト版Fisherのstabilityフルグリッド未検証（前回からの持ち越し）。
- `src/investigate_reversal.py`の統計的枠組みへの移行未実施（前回からの持ち越し）。
- ラベルのみ黒箱でのFisher(hard) vs OVR比較は未実施（Fisherに残る唯一の原理的な居場所候補、ユーザー判断待ちのまま）。
- 中間発表（2026-09-12）向けのスライド（`docs/MIDTERM_PRESENTATION_DRAFT.md`）は新ストーリーに未反映。
- 実データ未検証、インスタンス選択の妥当性検証は引き続き未実施。

## 次に行うこと

1. 中間発表用に、確定方針（Contrastive+C主軸、B/共有支持集合はオプション、Fisherは分析章）をスライドへ落とし込む。
2. 提案C（および必要ならB）を`docs/OVO_LIME_METHODS.md`に正式な手法定式化として追記する。
3. 実データセットでの再現性確認。
