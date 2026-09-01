# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-01（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/lime-multiclass-consistency-u9jgn9`（PR #1）
- 基準コミット: `626360f`

## 今回の目的

前回コミット（`626360f`、実験スキャフォールドのみで未完走）から続き、one-vs-rest LIME vs Fisher LIMEのconsistency・stability比較実験を完走させ、途中で見つかった理論的な誤りを修正しながら、実測ベースで正確な主張を組み立てる。

## 実施した変更と主要な変更ファイル

1. **`src/metrics.py`**: `transitivity_violation_rate`（推移律違反率）は理論的に常に0になる無意味な指標と判明したため、docstringで「使わない理由」を明記した上でコードは残置（削除はしていない）。新規に`mean_pairwise_feature_overlap`（クラス間の上位K特徴量集合のJaccard重なり）、`sum_to_one_deviation_topk`（top-K切り詰め後のsum-to-one逸脱）を追加。
2. **`src/surrogates.py`**: `fit_onevsrest`に`intercept`を追加返却。真のLasso特徴選択を行う`fit_onevsrest_lasso`（LIMEの`lasso_path`選択モード相当、二分探索でK個以上の非ゼロ係数を持つ最疎なLasso解を求め、上位K個を採用）を追加。`fit_fisher`に one-vs-rest形式の per-class 方向`onevsrest_direction(c) = S_W^{-1}(μ_c - μ_¬c)`を追加。**ソフトラベル版**`fit_fisher_soft`を追加（`π_i・f_c(z_i)`を重みとして使い、ハードラベルのargmaxを使わない）。`top_k_indices`ヘルパーを追加。
3. **`src/run_experiment.py`**: グリッドを`n_features=[8,14,20]`（冗長特徴量の余地を確保するため`[5,10,20]`から変更）×`n_classes=[3,4,5]`×`K∈{0.3,0.6}×n_features`に拡張。データ生成を`n_redundant = n_features - max(3,n_classes)`で相関の強い特徴量を含むよう変更。`transitivity_violation_rate`は使用停止。
4. **`src/investigate_reversal.py`**（新規）: `n_classes=5`付近でFisherの優位性が逆転する原因を切り分ける診断スクリプト。`n_informative`を3に固定して冗長特徴量予算をクラス数から分離し、ハード版・ソフト版Fisherを同時比較。

## 重要な判断とその理由

実装・実験の過程で、当初の理論的主張のいくつかが**再検証の結果、成り立たない/条件付きだった**ことが判明した。修論の記述に直結するため、時系列で記録する。

1. **「推移律が崩れる」は成り立たない**（既知、前回記録済み）。実数の全順序の自明な性質のため。
2. **sum-to-oneは全特徴量共有の独立回帰なら厳密に成立**（既知、前回記録済み）。lime公式パッケージの`feature_selection`でクラスごとに異なる特徴量が選ばれると崩れる。
3. **【今回】feature overlap（LIMEtreeの本当の主張の操作化）の第1回実験は null 結果だった**。全特徴量Ridge＋事後top-K切り詰めでは、one-vs-rest(0.516) vs Fisher(0.503)でほぼ差がなく、Fisherの優位性を示せなかった。
4. **【今回】真のLasso選択＋相関の強い冗長特徴量に設計変更したところ、Fisherの優位性が復活**（18セル中13でFisherが上回る）。特に`n_features=20, n_classes=3`（冗長特徴量が最多）で最大の差（K=6: one-vs-rest 0.392 vs Fisher 0.752）。ただし`n_classes=5`付近で逆転する例外あり。
5. **【今回・重要な発見】逆転の原因は「冗長特徴量予算」ではなく「ハードラベルによるクラス欠測」だった**。`n_informative`を固定して予算を切り分けても逆転は解消せず、代わりに`min_class_count_avg`（局所近傍で最も少数派のクラスのサンプル数）を調べたところ、n_classes=4で平均0.625（つまり近傍にそのクラスが1つも出現しない試行が多数）という深刻な欠測が判明。`fit_fisher`はhard_labelsで存在しないクラスの重心を計算できず、その回だけそのクラスが説明から丸ごと消える。one-vs-restの回帰は連続値ターゲットなので欠測しない。
6. **【今回】ソフトラベル版は欠測は完全に解消するが、feature overlap自体は必ずしも改善しない**。`n_present_soft_avg`は全条件で厳密にn_classesと一致（欠測ゼロを確認）。しかし肝心のfeature overlapは、10条件中2条件（n_classes=7）でのみソフト版が最良で、残り8条件ではソフト版がハード版より悪化し、一部（n_classes=5, n_perturb=300）ではone-vs-restにも負けた。考えられる理由：ソフト版の重心は全ての点から薄く寄与を受けるため、黒箱の確率分布があまり尖っていない（RandomForestにありがち）とクラス間の分離が弱まり、上位K特徴量の選択がノイズに敏感になる。**「ソフト版に切り替えれば良い」という単純な結論は成り立たない、バイアス・分散的なトレードオフとして報告すべき**。
7. **stability（Q1③の分散蓄積）は、全ての実験設定を通じて一貫してFisher優位**。データ生成やグリッドを変えても、競合ペア方向ベクトルの分散比は一度も逆転せず、条件によって約15倍〜100倍超の差があった。**現時点で最も頑健で説得力のある差別化ポイント**。

## 実行したテスト・確認結果

- 全特徴量版グリッド実験（`n_features=[5,10,20]`）完走。結果は上書きされたため`results/`には残っていない（旧`run_experiment.py`の出力）。
- Lasso選択＋相関特徴量版グリッド実験（`n_features=[8,14,20]`）完走。`results/experiment_results.csv`, `results/experiment_summary.csv`に保存。
- `src/investigate_reversal.py`（ハード版のみ）実行、その後ソフト版を追加して再実行。`results/reversal_investigation.csv`に保存。

## 未完了・既知の問題・未検証事項

- ソフトラベル版がなぜ`n_classes=7`でのみ有効なのかは仮説段階（重心の分離度が弱まる、という説明）で、直接的な検証（例えば重心間距離やS_Bのトレースを比較する）はまだ行っていない。
- feature overlap実験は`RandomForestClassifier`・`make_classification`の合成データのみで検証。実データや別の黒箱モデルでの再現性は未検証。
- 忠実性（fidelity）の評価はまだ未着手（TODO、以前から継続）。
- `docs/PROJECT_SUMMARY.md`を今回の内容に合わせて更新済み（本コミットに含む）。

## 次に行うこと

1. ソフト版が`n_classes=7`でのみ有効な理由を、重心間距離・S_Bの直接比較で検証する。
2. ハード版とソフト版のハイブリッド（例えば局所サンプルが少ないクラスだけソフトにフォールバックする）を検討する。
3. 忠実性（fidelity）指標の実験を追加する。
4. 実データセットでの再現性確認。
