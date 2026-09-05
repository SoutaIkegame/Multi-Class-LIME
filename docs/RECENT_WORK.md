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

ユーザーから、直近の実験・ドキュメントに対する詳細な外部レビュー（5点）を受けた。査読レベルの精度を要求する、正当な指摘だった。全て検証の上、コード・ドキュメント両方を修正した。

## 指摘5点と対応

1. **忠実性が学習に使った摂動上の成績になっている（in-sample）**：`run_pair_kernel_experiment.py`, `run_combined_bc_experiment.py`, `run_logistic_target_experiment.py`のfidelity/extreme/moderateは、いずれも学習に使ったZそのもので符号一致率を測っていた。学習内の適合度としては有効だが、未知の近傍への汎化を示す証拠にはならない。**対応**：`run_combined_bc_experiment.py`に独立held-out摂動（`Z_test`、同じインスタンス・同じカーネルで新規に引き直し）での評価を追加（`*_fidelity_test`等）。standard/kernel(B)/logistic(C)/combined(B+C)の4手法を一度に再検証。フルグリッド実行済み、結果は`docs/EXPERIMENT_LOG.md`フェーズ11.5に記載。他2スクリプトはこの再検証に一本化し、それぞれのdocstringに「fidelityはin-sample、`run_combined_bc_experiment.py`の`*_test`列を正とせよ」という注記を追加。
2. **「Fisherである限りギャップは消えない」は未証明**：フェーズ6の結果は特定の近傍・重み付け・正則化のもとでの結果であり、Fisherの一般的な限界を証明したものではない。**対応**：`docs/EXPERIMENT_LOG.md`フェーズ6、`docs/PROJECT_SUMMARY.md`、`src/surrogates.py`のコメントを「この設定ではFisherが劣った」という限定表現に修正。
3. **提案A（不採用）の敗因説明が実装と矛盾**：「フェーズ7は着目ペアだけで共有、提案Aは全クラスで共有だから負けた」と説明していたが、フェーズ7の`shared_support_fisher_soft`/`shared_support_ridge`も実際には全クラスの方向・係数を集約しており、この対比は誤り。**対応**：正しい違い（独立最適化の事後集約 vs 結合最適化）に修正。`docs/EXPERIMENT_LOG.md`フェーズ8、`docs/PROJECT_SUMMARY.md`を訂正。
4. **「ロジスティック損失は有界」は数学的に誤り**：有界なのは損失$L(s,q)=\log(1+e^s)-qs$ではなく、勾配$\sigma(s)-q\in[-1,1]$。**対応**：`docs/EXPERIMENT_LOG.md`フェーズ10、`docs/PROJECT_SUMMARY.md`の該当箇所を修正（`src/surrogates.py`の`fit_ovo_logistic`docstringは元々「bounded gradient」と正しく書かれていたため修正不要）。
5. **「有意差なし」→「固有の価値なし」、観測→確定原因という言い切り**：フェーズ7の「Fisherに固有の価値はない」、フェーズ6の「サンプル飢餓が原因」、フェーズ9の「副作用のない改善」を、観測結果と未証明の仮説を区別する表現に修正。

## 実施した変更と主要な変更ファイル

1. `src/run_combined_bc_experiment.py`: held-out評価（`Z_test`）を追加。フルグリッド再実行済み。
2. `src/run_pair_kernel_experiment.py`, `src/run_logistic_target_experiment.py`: docstringにin-sampleである旨の注記とheld-out版への参照を追加（再実行はしていない、`run_combined_bc_experiment.py`の`*_test`列が同じ比較をカバーするため）。
3. `src/surrogates.py`: フェーズ6のコメントを限定表現に修正。
4. `docs/EXPERIMENT_LOG.md`: 冒頭に「0.0 訂正記録」を新設。フェーズ6・7・8・9・10・11の該当箇所を訂正。フェーズ11.5（held-out再検証）を追加（本セッション末尾で確定）。
5. `docs/PROJECT_SUMMARY.md`: 訂正記録を追記、該当する結論の文言を修正。

## 重要な判断とその理由

held-out再検証は`run_combined_bc_experiment.py`一本に集約し、`run_pair_kernel_experiment.py`と`run_logistic_target_experiment.py`は再実行しなかった。理由：後者2つが個別に検証していたstandard/kernel(B)、standard/logistic(C)の比較は、`run_combined_bc_experiment.py`が4手法（standard/kernel/logistic/combined）を同時に扱うため完全に包含される。二重に実験を回すより、1本のheld-out版を「正」として位置づけ、他2本のdocstringに参照を追加する方が保守性が高いと判断した。

## 実行したテスト・確認結果

`run_combined_bc_experiment.py`のheld-out版をフルグリッド（9セル×20シード）で再実行。結果（詳細は`docs/EXPERIMENT_LOG.md`フェーズ11.5）：

- **B・Cそれぞれ単体のfidelity改善はheld-outでも生き残る**（standard比、全体fidelity+0.003〜0.005、4〜6/9セルで有意）が、in-sampleの9/9からは有意性・効果量とも縮小した。
- **「B+Cを組み合わせるとさらに積み上がる」というin-sample結論は再現されなかった**——combinedとkernel単体・logistic単体の差はheld-outで0〜1/9セルしか有意にならない。in-sample版で見えていた上乗せは学習サンプルへの適合度の見かけ上の差だった可能性が高く、撤回した。
- 極端領域fidelityはtrain・test問わず一貫して無風。
- 方向の安定性（resamplingベースで元々in-sampleの問題を受けない）は変わらず：logistic単体が最も一貫して改善（8/9）、combinedはlogistic単体より悪化（6/9有意）。
- **結論**：Cを単体の主軸として推奨、Bの追加併用は積極的には推奨しない、という方針に修正。

## 未完了・既知の問題・未検証事項

- フェーズ1〜8（`run_experiment.py`, `run_contrastive_experiment.py`, `run_fidelity_experiment.py`, `run_extreme_regime_experiment.py`, `run_shared_support_experiment.py`のRF側fidelity）も同じin-sample構造を持つ。今回はフェーズ9〜11（現在採用を検討している提案B/C/combined）のみ優先してheld-out化した。過去のフェーズも同様に再検証する価値があるが、未着手。
- 提案Aの敗因（独立最適化の事後集約 vs 結合最適化のどちらが真因か）は仮説のまま、切り分け実験は未実施。
- 前回からの持ち越し：ソフト版Fisherのstabilityフルグリッド未検証、`investigate_reversal.py`の統計化未実施、ラベルのみ黒箱での比較未実施、実データ未検証。

## 次に行うこと

1. **中間発表スライド（`draft_slides.pptx`、ユーザーに送付済み）の訂正が必要**：スライド10（提案Aの敗因説明が実装と矛盾）、スライド11・12（B+C併用を推奨する内容だが、held-out再検証で「B+Cが積み上がる」という結論は撤回済み）を、本セッションの訂正内容に合わせて作り直す必要がある。まだ未対応。
2. 時間が許せば、フェーズ1〜8のfidelityもheld-out化する（優先度高い、`今後の大きな課題`参照）。
3. 提案Aの敗因仮説（独立最適化の事後集約 vs 結合最適化）を切り分ける追加実験を検討する。
