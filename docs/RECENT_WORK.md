# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-05（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`（PR #5）
- 基準コミット: `add6c6f`

## 今回の目的

「Fisher版を提案したい。うまく行かないなら分析して何を変えればよいか」というユーザー依頼に対し、(1) Fisherが負ける原因の部品分解、(2) 診断に基づく提案（二段構成）の実装、(3) 正直な対照との比較、を行った。両方の実験が完走し、結論が出た。

## 実施した変更と主要な変更ファイル

1. `src/diagnose_fisher_direction.py`（新規）: Fisher方向の失敗要因分解。完走。結論は`PROJECT_SUMMARY.md`「Fisher方向の失敗要因分解」に恒久記載。
2. `src/surrogates.py`: `shared_support_fisher_soft` / `shared_support_ridge` / `fit_contrastive_on_support` を追加（二段構成サロゲート）。
3. `src/run_shared_support_experiment.py`（新規）: 提案の評価。ロジスティック黒箱（約2分）とRF黒箱（約25分）で完走。結果は`results/shared_support_{logistic,rf}_{results,stats}.csv`（gitignore対象）。要約は`PROJECT_SUMMARY.md`「共有支持集合（二段構成）の評価」。
4. `docs/OVO_LIME_METHODS.md`: 実装状況表を更新（OVO Fisher-LIMEは診断内で`pair_*`として実装・評価済み）。

## 重要な判断とその理由

1. **Fisherを「提案」から降ろし「分析章」に位置づけ直すことを推奨**（`PROJECT_SUMMARY.md`に記載）。根拠：係数推定器としては$S_W$尺度の偏りで回帰に本質的に勝てない（診断）、共有集合の供給源としてもRidge集約と全指標で区別できない（今回）。
2. **「共有支持集合＋Contrastive再フィット」自体は有用な一貫性ノブとして残す**。RF黒箱でper-pair Lassoに対しfidelity・特徴集合安定性・overlapで有意に優れ、代償（着目ペア固有特徴の取りこぼし、真top-K再現率−4〜5pt）も定量化できた。ただし供給源はRidge集約で十分。
3. Fisher-metric ridge（$S_W$を罰則行列にする案）は不採用（診断で$S_W$尺度自体が偏りの源）。

## 実行したテスト・確認結果

- 両実験ともフルグリッド（9セル×20シード×K=2水準）完走、exit code 0。Holm補正後の有意セル数は`PROJECT_SUMMARY.md`の数値どおり（集計スクリプトはアドホック、`stats.csv`から再計算可能）。

## 未完了・既知の問題・未検証事項

- **ラベルのみ黒箱**（`predict_proba`が無い状況）でのFisher(hard) vs OVR(0/1ラベル回帰)の比較は未実施。Fisherに残る唯一の原理的な居場所候補。ユーザーの判断待ち。
- 中間発表（2026-09-12）向けのストーリー組み替えは文書に書いたが、スライド（`docs/MIDTERM_PRESENTATION_DRAFT.md`）には未反映。
- 特徴重要度の順位は生の係数単位で比較（標準化効果ではない）。注記が必要。
- 前回からの持ち越し：ソフト版Fisherのstabilityフルグリッド未検証、`investigate_reversal.py`の統計化未実施、実データ未検証、インスタンス選択の妥当性。

## 次に行うこと

1. ユーザーと「Fisherを分析章に降ろす」方針を確認する。
2. 確認が取れたら、ラベルのみ黒箱の検証を1本追加するか決める（既存コードで1スクリプト、数分）。
3. 中間発表ドラフトを新ストーリー（問題設定 → Contrastive提案 → 一貫性ノブ → Fisherの分析）に合わせて改訂する。
