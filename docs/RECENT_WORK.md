# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-03（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`
- 基準コミット: `118d8f0`

## 今回の目的

ユーザーから「今までの実験がちょっと適当すぎるからやり直そう」という指摘を受け、優先度を確認したところ「統計的な厳密さ不足」が最優先事項だった。具体的には、それまでの実験ドライバが各グリッドセルにつきデータセット・分類器の抽選を1回しか行わず、信頼区間や有意差検定なしに平均値だけを報告していた問題を解消する。

## 実施した変更と主要な変更ファイル

1. **`src/stats_utils.py`**（新規）: 統計ヘルパーモジュール。
   - `bootstrap_ci`: パーセンタイル・ブートストラップによる平均・95%信頼区間。
   - `paired_wilcoxon`: 対応ありWilcoxon符号順位検定＋matched-pairs rank-biserial効果量。
   - `holm_bonferroni`: 多重比較のHolm-Bonferroni補正（step-down）。
   - `compare_methods`: 上記を組み合わせ、グリッドセルごとにseedレベル平均を独立サンプルとして扱った統計比較表を作る高水準関数。**擬似反復（同一データセット内の相関したインスタンスを独立サンプル扱いすること）を避ける設計**であることをdocstringに明記。
2. **`src/run_experiment.py`, `src/run_contrastive_experiment.py`, `src/run_fidelity_experiment.py`, `src/run_extreme_regime_experiment.py`**: 各グリッドセルにつき独立したデータセット抽選（`make_classification`＋`RandomForestClassifier`の再学習）を`N_DATASET_SEEDS=20`回繰り返すよう変更。各`run_one_cell`呼び出しは元々`rng`から新しい`random_state`を引いて完全に独立したデータセットを生成する設計だったため、`main()`側でこの呼び出しをseedループで囲むだけで反則なく反復できた。各行に`seed`列を追加し、seed単位でグループ化した上で`compare_methods`により統計検定を実行。生データは`results/*_results.csv`（行数が従来比20倍）、統計比較表は新規の`results/*_stats.csv`に出力。

## 重要な判断とその理由

1. **スコープを統計的厳密性のみに限定した**（ユーザーへの優先度確認で明示的に選択）。インスタンス選択方法（マージン最小8点のみ）、実データ未検証、グリッドの粗さは今回は対象外として残した。
2. **`src/investigate_reversal.py`は対象外とした**。診断スクリプトであり、他の4本とは異なる設計軸（n_informativeを固定してn_classesを6〜7まで広げる）を持つため、今回の枠組みへの統合は別タスクとした。ただし、今回の`run_experiment.py`再検証結果との整合性確認が必要な状態になった（下記参照）。
3. **N_DATASET_SEEDS=20を選択**。1セルあたりの実行時間（run_experiment.py: 約5秒、run_contrastive_experiment.py: 約8秒、他は1〜1.5秒）から、20反復×9セルでも全4本合計で1時間以内に収まると判断。対応ありWilcoxon検定にとって20ペアは検出力として妥当な下限。
4. **seedレベル平均を独立サンプルの単位とした**（インスタンス単位ではない）。同一データセット内の8インスタンスは同じ分類器・同じデータ抽選から来ており独立ではないため、まずインスタンスをseedごとに平均してから、そのseed平均（20個）を独立サンプルとして検定・信頼区間に使う設計にした。

## 実行したテスト・確認結果

- 縮小グリッド（1セル、seed数3、インスタンス数3）でのスモークテストを4本全てで実施し、エラーなく統計比較表まで出力されることを確認。
- フルグリッド（n_features×n_classes = 3×3 = 9セル、K=2水準、`N_DATASET_SEEDS=20`）を4本ともバックグラウンドで完走（全てexit code 0）。実行時間: run_experiment.py 約14分、run_contrastive_experiment.py 約23分、run_fidelity_experiment.py 約3分、run_extreme_regime_experiment.py 約4分。

## 主要な結果（全てWilcoxon対応あり検定、20独立データセットseed、Holm-Bonferroni補正後のp値で判定）

1. **stability（正規化、Fisher hard vs OVR、`experiment_stats.csv`/`contrastive_stats.csv`）**: 全9グリッドセルでFisherが統計的に有意に不安定（p≈0.000002、20シード全てが同じ方向、効果量=-1.0）。以前「3セルのみのアドホック検証」だったhard版の結論が、フルグリッド・統計的検定で確定した。
2. **fidelity（Hellinger損失、絶対確率、`fidelity_stats.csv`）**: 全9セル×2手法（hard/soft）でOVRが統計的に有意に優位（p≈0.000002）。Fisher softはhardより全9セルで有意に優位。以前のアドホックな結論を厳密に確認。
3. **【訂正】feature overlap（OVR vs Fisher hard、`experiment_stats.csv`）**: 以前「クラス数が多いと逆転する」と記載していたが、統計的再検証では、Fisherは検証した全18セル×Kで数値上一貫してOVRより高い（優位）。有意性はクラス数増加につれ失われ、n_classes=5の多くのセルで非有意になるが、**OVRが有意に上回るケースは0件**。正確には「逆転」ではなく「クラス数増加に伴う優位性の消失（検出力不足ではなく、差そのものが小さくなる）」。
4. **fidelity（ペア符号一致率、Contrastive追加、`contrastive_stats.csv`）**: OVR vs Fisher(hard)は9セル中4セル（n_classes・n_featuresが大きい側）で有意にOVR優位、残りは非有意。OVR vs Contrastiveは9セル中8セルで非有意（差は0.002〜0.004の極小）——「ほぼ完全な同点」を統計的に確認。Fisher vs Contrastiveは9セル中4セルで有意にContrastive優位。
5. **stability（正規化、Contrastive追加、`contrastive_stats.csv`）**: FisherはOVR・Contrastiveの両方に全9セルで有意に劣る。OVR vs Contrastiveは9セル中3セルでContrastiveがわずかに有意に安定、残りは非有意。
6. **feature overlap（Contrastive追加、`contrastive_stats.csv`）**: OVR vs Contrastiveは18セル×K中3セルのみ有意（小さい差）。Fisher vs Contrastiveはn_classes=3・高n_featuresのセルで有意にFisher優位（クラス間 vs ペア間という比較単位の不一致は未解消のまま）。
7. **極端確率領域fidelity（`extreme_regime_stats.csv`）**: Fisher(hard)は全9セルで統計的に有意に劣化（p<0.003）——「サンプル飢餓」仮説を統計的に確認。ContrastiveはOVRに対し9セル中6セルで有意に優位（クラス数・次元数が大きいセルに集中、n_classes=3では非有意）。**新しい発見**：穏やかな領域では逆にContrastiveがOVRよりわずかに、しかし統計的に有意に劣る場合がある（9セル中3セルで有意、差は-0.002〜-0.008）。極端領域での優位性は無償ではなく、穏やかな領域での小さなコストと引き換えの可能性がある。

## 未完了・既知の問題・未検証事項

- ソフト版Fisherのstability（正規化）は依然としてフルグリッド未検証（今回の4本のstability比較はいずれもhard版のみが対象）。
- `src/investigate_reversal.py`（n_classes 6〜7での「逆転」診断）はまだ単一シードのアドホック診断のまま。今回の`run_experiment.py`の結果（n_classes 3〜5の範囲では「逆転」ではなく「優位性の消失」）とinvestigate_reversal.pyの主張（n_classes 6〜7でのより明確な逆転）は異なるn_classesレンジ・異なる設計（n_informative固定）を見ているため直接矛盾はしないが、再検証するまでは両立するとも矛盾するとも言い切れない。
- インスタンス選択（マージン最小8点のみ）、実データ未検証、グリッドの粗さ（n_features 3点、n_classes 3点のみ）は今回のスコープ外。
- 効果量（matched-pairs rank-biserial、-1〜+1）は「20シード中何シードが同じ方向を向いているか」を表す指標であり、実質的な効果の大きさ（`mean_diff`の絶対値）とは別物。両方を見て解釈する必要がある。

## 次に行うこと

1. ソフト版Fisherのstabilityを、今回と同じ`N_DATASET_SEEDS`方式でフルグリッド検証する。
2. `investigate_reversal.py`を同じ統計的枠組みで再検証し、n_classes 6〜7での「逆転」主張を確認・修正する。
3. 4手法×5指標（fidelity絶対値/ペア符号、stability、feature overlap、極端領域fidelity）の、統計的に裏付けられた結果を統合し、修論の主張として文章化する。
4. 実データセットでの再現性確認。
5. インスタンス選択方法（マージン最小のみ）の妥当性検証。
