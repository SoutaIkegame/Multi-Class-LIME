# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-05（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`（PR #5）
- 基準コミット: `bbafa2b`

## 今回の目的

「Fisher版を提案したい。今のアルゴリズムがうまく行かないなら分析して何を変えればよいか考える」というユーザー依頼。前回までにFisher(hard)がほぼ全指標で負けることは分かっていたので、(1) 負ける原因を部品ごとに分解し、(2) その診断に基づいて提案版を設計・実装し、(3) 正直な対照と比較する。

## 実施した変更と主要な変更ファイル

1. **`src/diagnose_fisher_direction.py`**（新規）: Fisher方向$S_W^{-1}(\mu_c-\mu_d)$を部品分解する診断。黒箱はロジスティック回帰（真の係数既知）、20シード×9セル、`stats_utils.compare_methods`で対応あり検定。完走済み。
2. **`src/surrogates.py`**: 二段構成サロゲートを追加。`shared_support_fisher_soft`（Fisher soft方向の集約で全クラス共通top-K集合）、`shared_support_ridge`（対照：OVR Ridge係数の集約）、`fit_contrastive_on_support`（集合に制限したContrastive Ridge）。
3. **`src/run_shared_support_experiment.py`**（新規）: 提案の評価。`pair_lasso`（per-pair Contrastive Lasso、独立選択）vs `fisher_select`（提案）vs `ridge_select`（対照）。黒箱ロジスティック（真のtop-K再現率、Spearman）とRF（ペア符号fidelity、特徴集合安定性＝再サンプリング間Jaccard、方向安定性、ペア横断overlap）。**このコミット時点ではフルグリッド実行中（未完了）**。

## 重要な判断とその理由

1. **診断の結論**（`PROJECT_SUMMARY.md`に恒久記載）：$S_W^{-1}$は必要、クラス横断プーリングはほぼ無料、ハードラベルが最大の損失源、残る差はクラス内散布を尺度に使うことに内在する偏り。→ Fisherは係数推定器としては回帰に勝てないが、共有構造の供給源としては精度コストなしで使える。
2. **提案を二段構成にした理由**：上の診断から、Fisherに残された役割は「どの特徴を使うか（全クラス共通）」であり、「どれだけの重みか」はContrastive回帰に任せるのが正しい分業。LIMEtreeの「共通構造」の定義に直接対応する。
3. **`OVO_LIME_METHODS.md`のFisher-metric ridge案（$S_W$を罰則行列にする）は不採用**。診断(5)で$S_W$尺度そのものが偏りの源だと分かったため、それを回帰に持ち込む設計は改悪になる。
4. **対照としてRidge集約版を必ず入れた**。Fisher選択が「単に密なランキングを共有しただけ」の効果と区別できなければ、Fisher固有の価値はないと結論すべき。スモークテストでは両者の選択がかなり一致していた。

## 実行したテスト・確認結果

- `diagnose_fisher_direction.py`: フルグリッド完走（約15秒）。主要数値は`PROJECT_SUMMARY.md`参照。
- `run_shared_support_experiment.py`: 縮小グリッドでスモークテスト済み（両黒箱ともエラーなく統計表まで出力）。フルグリッドはバックグラウンド実行中（logistic：数分、RF：30分程度の見込み）。

## 未完了・既知の問題・未検証事項

- **`run_shared_support_experiment.py`のフルグリッド結果は未取得**。結果次第で提案の位置づけが変わる：Fisher選択がRidge選択に勝てば「Fisher固有の共有構造に価値あり」、同等なら「共有構造自体には価値があるがFisherである必要はない」、per-pair Lassoに劣れば「共有はfidelityコストが大きい」。
- 特徴重要度の順位は生の係数単位（|θ_j|）で比較しており、標準化効果（|θ_j|·std_j）では測っていない。他のground-truth実験と整合させるための選択だが、注記が必要。
- 前回からの持ち越し：ソフト版Fisherのstabilityフルグリッド未検証、`investigate_reversal.py`の統計化未実施、実データ未検証、インスタンス選択の妥当性。

## 次に行うこと

1. `run_shared_support_experiment.py`の結果を読み、`PROJECT_SUMMARY.md`に提案の評価結果を恒久記載する。
2. 結果に応じて提案の最終形を決める（Fisher選択 vs Ridge選択の差が出なければ、提案は「共有支持集合＋Contrastive回帰」として書き、Fisherは診断上の位置づけに留める）。
3. 中間発表向けに「Fisherが負ける理由の分解」を1枚にまとめる。
