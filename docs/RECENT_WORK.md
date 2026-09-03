# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-03（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`
- 基準コミット: `9abeced`（統計的厳密性強化コミット）

## 今回の目的

前回セッションで、他分野の関連研究（LIMEtree, SLISEMAP, AIM, CLIMAX, Rahnama et al., TERP）が使う評価指標を整理し、現行の4指標（sum-to-one, feature overlap, stability, fidelity）と対応させたところ、**Rahnama et al. (2024)型のground-truth検証（黒箱自体が線形モデルで真の係数が既知の場合に、サロゲートがその係数を復元できるかを見る）が未実装**だと分かった。今回はこれを実験として追加した。

## 実施した変更と主要な変更ファイル

1. **`src/metrics.py`**: `pairwise_coef_spearman(true_coef, est_coef)`を追加。真の係数ベクトルと推定係数ベクトルのSpearman順位相関。Ridge/LDAの縮小推定は係数の「大きさ」に偏りを生むが「順位」は保つはずという理由で、生の大きさではなく順位相関を使う（Rahnama et al.と同じ方針）。
2. **`src/run_groundtruth_experiment.py`**（新規）: 黒箱をこれまでの`RandomForestClassifier`から`LogisticRegression(multi_class='multinomial')`に差し替えた実験ドライバ。多項ロジスティック回帰は対数オッズについて厳密に線形（$\log(p_c/p_d)=(\theta_c-\theta_d)\cdot z+\text{const}$がどこでも成立）なので、真の係数$\theta_{c^*}-\theta_{c'}$がどの点でも既知になる。この真の係数と、OVR・Fisher(hard)・Fisher(soft)・Contrastiveそれぞれの推定係数のSpearman順位相関を測る。前回整備した統計的枠組み（`N_DATASET_SEEDS=20`、`stats_utils.compare_methods`）をそのまま流用。結果は`results/groundtruth_results.csv`、統計比較は`results/groundtruth_stats.csv`。

## 重要な判断とその理由

1. **黒箱をLogisticRegressionに差し替えた**（他の実験群はRandomForestのまま）。RandomForestには閉形式の「真の局所係数」が存在しないため、ground-truth検証にはそもそも使えない。この実験だけ黒箱の種類が異なる点に注意（他の実験群との直接比較はできない、これは別軸の検証）。
2. **順位相関（Spearman）を使い、生の係数の大きさは比較しない**。Fisherの$S_W^{-1}(\mu_c-\mu_d)$やRidgeの縮小推定には自然な尺度がないため、大きさの比較は意味を持たない。RELATED_WORKで整理したRahnama et al.の方針をそのまま踏襲。
3. **既存のデータ生成設計（相関の強い冗長特徴量あり）をそのまま維持した**。真の$\theta$自体がこの冗長性の影響を受けた値になる（多項ロジスティック回帰も相関特徴間で任意に重みを分配する）ため、サロゲート側の推定にも同じ困難が課される公平な設定になっている。

## 実行したテスト・確認結果

- 縮小グリッド（1セル、seed数3〜5）でのスモークテストで正常動作を確認。
- フルグリッド（9セル×`N_DATASET_SEEDS=20`）を実行、約15秒で完走（LogisticRegressionはRandomForestよりはるかに高速）。

## 主要な結果（全てWilcoxon対応あり検定、20独立データセットseed、Holm-Bonferroni補正後のp値で判定）

全グリッド平均のSpearman ρ（1に近いほど真の係数ランキングを正しく復元）:

| 手法 | 平均 ρ | グリッド内の範囲 |
|---|---|---|
| Contrastive | 0.999 | 0.998〜1.000 |
| Fisher (soft) | 0.983 | 0.981〜0.987 |
| OVR | 0.976 | 0.972〜0.980 |
| Fisher (hard) | 0.953 | 0.941〜0.966 |

統計的な優劣（`results/groundtruth_stats.csv`）:
- **Contrastive vs OVR / Fisher(hard) / Fisher(soft)**: 全9セルでContrastiveが有意に優位（p≦0.0003）。**Contrastiveにとって今までで最も明確な勝ちどころ**。
- **OVR vs Fisher(hard)**: 全9セルでOVRが有意に優位（p≦0.012）。
- **Fisher(soft) vs Fisher(hard)**: 全9セルでFisher(soft)が有意に優位（p≦0.0002）。
- **Fisher(soft) vs OVR**: 9セル中8セルでFisher(soft)が有意に優位（n_features=8, n_classes=3のみ非有意）。

総合順位 Contrastive > Fisher(soft) > OVR > Fisher(hard) が、ほぼ全ペア・全セルで統計的に確定した。

理論的な解釈: Contrastiveは対数オッズを直接回帰するため、黒箱が対数オッズについて線形である限り真の係数をほぼ完璧に復元できる（数学的に予想通り）。OVRは生の確率（softmaxで非線形）を回帰するため一致度が下がる。Fisher(hard)が最下位なのは、stability・極端領域fidelityで既に見えていた「ハードラベルによるサンプル飢餓」問題の別角度での再確認と考えられる。

## 未完了・既知の問題・未検証事項

- この実験は黒箱をLogisticRegressionに変えているため、他の実験（RandomForest黒箱）と直接同じ土俵で比較はできない。「黒箱が線形なら」という条件付きの結果である点に注意。
- 黒箱が非線形（本来のRandomForest）な場合、真の局所係数という概念自体が定義できないため、この検証軸を非線形黒箱に拡張する方法はまだ考えていない。
- ソフト版Fisherのstability（正規化）は依然としてフルグリッド未検証（前回からの持ち越し）。
- `src/investigate_reversal.py`はまだ単一シードのアドホック診断のまま（前回からの持ち越し）。
- インスタンス選択（マージン最小8点のみ）、実データ未検証、グリッドの粗さは引き続きスコープ外。

## 次に行うこと

1. グラウンドトゥルース検証の結果を、修論の主張の中心的な柱として位置づけることを検討する（Contrastiveの最も明確な優位性のため）。
2. ソフト版Fisherのstabilityを、今回整備した統計的枠組みでフルグリッド検証する。
3. `investigate_reversal.py`を同じ統計的枠組みで再検証する。
4. one-vs-rest, Fisher(hard/soft), Contrastiveの全指標（fidelity絶対値/ペア符号、stability、feature overlap、極端領域fidelity、グラウンドトゥルース復元）の結果を統合し、修論の主張として文章化する。
5. 実データセットでの再現性確認。
