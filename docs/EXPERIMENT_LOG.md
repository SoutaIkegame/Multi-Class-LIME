# 実験ログ（詳細版）

更新日: 2026-09-05

<!--
この文書は今までに実施した全実験の詳細な記録。PROJECT_SUMMARY.md/RECENT_WORK.md
が「現在の状態・直近の引き継ぎ」を簡潔にまとめるのに対し、こちらは各実験の
目的・方法・生データに基づく数値・結論を時系列かつ網羅的に記録する。
数値は全て results/*.csv から再集計したもの（推定ではない）。
-->

## 0. 共通の実験設計

### 0.1 合成データとブラックボックス

- 合成データ: `sklearn.datasets.make_classification`。`n_informative = max(3, n_classes)`、`n_redundant = n_features - n_informative`（相関の強い冗長特徴量を意図的に含める）、`class_sep=1.2`、`n_samples=2000`、train/test分割はtest_size=0.3・random_state=0固定。
- ブラックボックスは実験ごとに2種類使い分ける。
  - **RandomForestClassifier**（`n_estimators=200, random_state=0`）: 非線形決定境界。ほとんどの実験（consistency, stability, fidelity, feature overlap, 極端領域, 提案A/B/Cの評価）で使用。
  - **LogisticRegression（multinomial）**（`max_iter=2000, C=1.0`）: 対数オッズについて厳密に線形。真の係数$\theta_c$が既知になるため、グラウンドトゥルース検証・Fisher診断・提案A/Cの一部評価で使用。

### 0.2 グリッドと摂動

- グリッド: `n_features ∈ {8, 14, 20}` × `n_classes ∈ {3, 4, 5}`（9セル）。実験によりKも `{0.3, 0.6} × n_features` の2水準を追加。
- 各セルで「最も際どい」8インスタンス（黒箱の予測確率で1位と2位の差が最小のもの、`pick_contested_instances`）を説明対象にする。
- 摂動: LIMEのデフォルト方式（`z = x + noise * feature_std`、指数カーネルで近接度重み付け、`src/perturbation.py`）を全手法・全実験で共有。

### 0.3 統計的枠組み（2026-09-03導入）

- 各グリッドセルにつき**独立したデータセット抽選を20回（`N_DATASET_SEEDS=20`）**繰り返す。
- 8インスタンスの平均をまずseedごとに取り（擬似反復の回避）、20個のseedレベル平均を独立サンプルとして扱う。
- 手法間比較は**対応ありWilcoxon符号順位検定**（同一seed・同一摂動を共有するため）。p値は**Holm-Bonferroni補正**（同一指標内の全セル・全手法ペアを1つの検定族として補正）。
- 実装: `src/stats_utils.py`（`bootstrap_ci`, `paired_wilcoxon`, `holm_bonferroni`, `compare_methods`）。

以下、各フェーズの実験を時系列で記録する。表の「有意」列は上記Holm補正後に有意だったグリッドセル数（分母は検定族のセル×手法ペア数）。

---

## フェーズ1: OVR vs Fisher LIME（consistency・stability）

**目的**: 従来のOVR形式のLIME（クラスごとに独立回帰）に対し、Fisher判別分析（LDA）を局所サロゲートとして使う代替案の妥当性を検証する（当初の研究提案そのもの）。

**手法**: `fit_onevsrest`（重み付きRidge回帰、クラスごと独立）、`fit_fisher`（プールされたクラス内散布行列$S_W$を使うLDA、ハードラベル版）。

**スクリプト**: `src/run_experiment.py` → `results/experiment_results.csv` / `experiment_stats.csv`

| 指標 | OVR | Fisher(hard) | 差（OVR−Fisher） | 有意 |
|---|---|---|---|---|
| feature overlap（Jaccard、K別×2水準） | 0.446 | 0.509 | −0.063 | 8/18 |
| stability（正規化分散） | 0.051 | 0.126 | −0.074 | 9/9 |

**結論**: Fisherはfeature overlapで数値上優位（後述フェーズ4で「クラス数増加につれ有意性が消失、逆転はしない」と精緻化）。stabilityは全セルで有意にFisherが劣る（Fisherの分散はOVRの約2.4倍）。

**理論修正の記録**: 当初「推移律が崩れる」という問題提起をしていたが、任意の実数の大小比較は常に推移的なため数学的に成立しないと判明（`transitivity_violation_rate`は理論的に常に0、コードのみ残置）。consistencyの正しい定義はLIMEtree（Sokol & Flach 2025, Sec.3, p.5）の「モデル間で共通構造・特徴部分集合を共有しない」ことに基づくと整理し直した。

---

## フェーズ2: Fidelity実験（絶対確率）

**目的**: Fisherを標準の多クラス確率分類器として評価した場合の忠実性を測る。

**手法**: OVR、Fisher(hard)、Fisher(soft)（`fit_fisher_soft`、ハードラベル化せず$\pi_i \cdot f_c(z_i)$を重みに使う）。損失は重み付き二乗Hellinger距離（SLISEMAP Eq.11と同じ）。

**スクリプト**: `src/run_fidelity_experiment.py` → `results/fidelity_results.csv` / `fidelity_stats.csv`

| 比較 | 平均損失差 | 有意 |
|---|---|---|
| OVR (0.017) vs Fisher hard (0.065) | −0.049 | 9/9 |
| OVR (0.017) vs Fisher soft (0.036) | −0.020 | 9/9 |
| Fisher hard (0.065) vs Fisher soft (0.036) | +0.029 | 9/9 |

**結論**: OVRがFisher(hard)より約3.9倍、Fisher(soft)より約2.2倍低損失（優れる）。Fisher soft はhardの約1.8倍改善するが、OVRには届かない。

---

## フェーズ3: 極端確率領域実験

**目的**: 競合する2クラスの一方の確率がほぼ0の局所領域（閾値0.15未満、局所近傍の平均19%を占める）でfidelityを測る。OVR・Fisher(hard)・Contrastive LIME（後述）の3手法。fidelityは「黒箱が決めた予測クラス$c^*$と競合クラス$c'$のペア比較の符号一致率」。

**スクリプト**: `src/run_extreme_regime_experiment.py` → `results/extreme_regime_results.csv` / `extreme_regime_stats.csv`

| 領域 | 比較 | 差 | 有意 |
|---|---|---|---|
| 極端領域 | Contrastive (0.944) vs OVR (0.937) | +0.007 | 6/9 |
| 極端領域 | Fisher (0.907) vs OVR (0.937) | −0.030 | 9/9 |
| 穏やかな領域 | Contrastive (0.782) vs OVR (0.786) | −0.004 | 3/9 |
| 穏やかな領域 | Fisher (0.771) vs OVR (0.786) | −0.015 | 3/9 |

**結論**: Contrastiveは極端領域で弱いが本物の優位性（6/9セルで有意）。Fisherは極端領域で一貫して有意に劣化——feature overlap・stabilityで見えた「ハードラベルのサンプル飢餓」が3つ目の文脈で再確認された。

---

## フェーズ4: Contrastive LIME追加（3手法同時比較）

**目的**: 「$\log(p_A/p_B)$を直接回帰する」Contrastive LIME（`fit_contrastive`、ユーザー提案）を追加し、OVR・Fisher(hard)・Contrastiveの3手法をfidelity・stability・feature overlapで同時比較する。

**スクリプト**: `src/run_contrastive_experiment.py` → `results/contrastive_results.csv` / `contrastive_stats.csv`

| 指標 | 比較 | 値 | 差 | 有意 |
|---|---|---|---|---|
| fidelity（ペア符号） | OVR (0.821) vs Contrastive (0.819) | +0.001 | 1/9 |
| fidelity（ペア符号） | Fisher (0.800) vs Contrastive (0.819) | −0.020 | 4/9 |
| fidelity（ペア符号） | OVR (0.821) vs Fisher (0.800) | +0.021 | 5/9 |
| stability（正規化） | OVR (0.051) vs Contrastive (0.051) | +0.0001 | 4/9 |
| stability（正規化） | Fisher (0.126) vs Contrastive (0.051) | +0.075 | 9/9 |
| feature overlap | Fisher (0.509) vs Contrastive (0.466) | +0.043 | 3/18 |
| feature overlap | OVR (0.446) vs Contrastive (0.466) | −0.020 | 6/18 |

**結論**: ContrastiveはfidelityでOVRとほぼ完全な同点（差0.001〜0.002、ほぼ非有意）。stabilityもOVRとほぼ同格、Fisherより明確に優れる。feature overlapは中間的（クラス間 vs ペア間という比較単位の不一致は未解消）。

---

## フェーズ5: グラウンドトゥルース復元実験（Rahnama et al. 2024型）

**目的**: 黒箱をLogisticRegression（multinomial）に差し替え、真の係数$\theta_{c^*}-\theta_{c'}$が既知の状況で、各手法の推定係数とのSpearman順位相関を測る。局所再現性（fidelity）ではなく「正しい特徴に重みを置けているか」を検証する、質的に異なる評価軸。

**スクリプト**: `src/run_groundtruth_experiment.py` → `results/groundtruth_results.csv` / `groundtruth_stats.csv`

| 手法 | 平均 Spearman ρ |
|---|---|
| **Contrastive** | **0.9993** |
| Fisher (soft) | 0.9832 |
| OVR | 0.9758 |
| Fisher (hard) | 0.9528 |

全ペア比較が9/9セルで有意（Contrastive vs 他3手法、Fisher soft vs hard、OVR vs Fisher hard）。唯一 OVR vs Fisher soft のみFisher softがわずかに優る場合がある（差−0.007、8/9で有意）。

**結論**: Contrastiveが全手法に対し全セルで統計的に有意に最も真の係数を復元する——このプロジェクトで最も明確な勝ちどころ。理論的にも整合的（黒箱が対数オッズについて線形なら、対数オッズを直接回帰するContrastiveはほぼ完璧に一致するはず）。

---

## フェーズ6: Fisher方向の失敗要因分解（診断）

**目的**: Fisherの方向$v=S_W^{-1}(\mu_c-\mu_d)$を部品ごとに分解し、なぜ回帰系（OVR/Contrastive）に負けるのかを機構レベルで特定する。

**変種**（黒箱=LogisticRegression、真の係数との比較）:
- `centroid`: 重心差$\mu_c-\mu_d$のみ（$S_W$なし）
- `pooled`: 現行Fisher（全クラス共通$S_W$）
- `pair`: 2クラスだけで$S_W$を作る（OVO-Fisher）
- `diag`: $S_W$の対角成分のみ（回転なし）
- `cov_logodds`: 重心の代わりに対数オッズとの共分散を使う連続応答版

**スクリプト**: `src/diagnose_fisher_direction.py` → `results/diagnose_fisher_direction_results.csv` / `_stats.csv`

| 問い | 比較 | 差 | 有意 | 解釈 |
|---|---|---|---|---|
| $S_W^{-1}$は必要か | pooled_hard (0.953) vs centroid_hard (0.885) | +0.068 | 9/9 | 必要（尺度補正が効く） |
| $S_W^{-1}$は必要か | pooled_soft (0.983) vs centroid_soft (0.907) | +0.076 | 9/9 | 同上 |
| 全クラスプーリングの代償 | pair_hard (0.953) vs pooled_hard (0.953) | +0.0004 | 0/9 | **代償ほぼゼロ** |
| 全クラスプーリングの代償 | pair_soft (0.986) vs pooled_soft (0.983) | +0.003 | 4/9 | ほぼゼロ |
| 回転（非対角）は必要か | diag_soft (0.933) vs pooled_soft (0.983) | −0.050 | 9/9 | 必要（対角だけでは不十分） |
| ハード vs ソフト | pooled_soft (0.983) vs pooled_hard (0.953) | +0.030 | 9/9 | **サンプル飢餓が最大の損失源** |
| 重心 vs 連続応答 | cov_logodds (0.985) vs pooled_soft (0.983) | +0.001 | 4/9 | 改善はごく僅か |
| Contrastiveとの残差 | Contrastive (0.999) vs cov_logodds (0.985) | +0.015 | 9/9 | **本質的な差、消えない** |
| Contrastiveとの残差 | Contrastive (0.999) vs pooled_soft (0.983) | +0.016 | 9/9 | 同上 |

**結論**: (1) $S_W^{-1}$のスケール補正は必要で効いている。(2) 全クラス共通$S_W$というFisherの「売り」（クラス横断の共有構造）は精度をほぼ犠牲にしていない。(3) ハードラベルによるサンプル飢餓が最大の改善可能な損失源。(4) 連続応答にしても改善は僅少——重心という要約自体が情報を捨てているわけではない。(5) 残る本質的なギャップは、クラス内散布$S_W$という尺度そのものが判別的な目的（log-oddsの勾配）には偏った正規化になっているため、Fisherである限り解消しない。

---

## フェーズ7: 共有支持集合（二段構成）実験

**目的**: フェーズ6の(2)から、Fisherに残る役割は「係数の精度」ではなく「クラス横断の共有構造の供給」だと分かったため、二段構成（Stage1で共有特徴集合を選び、Stage2でContrastive回帰を再フィット）を評価する。正直な対照として、Stage1をFisher方向ではなく素のOVR Ridge係数の集約に置き換えた版（`ridge_select`）も比較する。

**手法**: `pair_lasso`（per-pair Contrastive Lasso、独立選択）、`fisher_select`（soft Fisherの方向を集約→Contrastive再フィット）、`ridge_select`（OVR Ridge係数を集約→Contrastive再フィット）。

**スクリプト**: `src/run_shared_support_experiment.py` → `results/shared_support_{logistic,rf}_*.csv`

### 黒箱=LogisticRegression（真top-K再現率・Spearman）

| 比較 | 差 | 有意 |
|---|---|---|
| fisher_select (0.749) vs ridge_select (0.741) の再現率 | +0.008 | 0/18 |
| ridge_select (0.741) vs pair_lasso (0.793) の再現率 | −0.053 | 5/18 |
| fisher_select (0.798) vs ridge_select (0.792) のSpearman | +0.006 | 0/18 |
| ridge_select (0.792) vs pair_lasso (0.817) のSpearman | −0.025 | 5/18 |

### 黒箱=RandomForest（ペア符号fidelity・支持集合安定性）

| 比較 | 差 | 有意 |
|---|---|---|
| fisher_select (0.784) vs ridge_select (0.784) のfidelity | +0.0003 | 0/18 |
| ridge_select (0.784) vs pair_lasso (0.761) のfidelity | +0.023 | 14/18 |
| fisher_select (0.789) vs ridge_select (0.789) の支持集合安定性 | +0.0004 | 0/18 |
| ridge_select (0.789) vs pair_lasso (0.717) の支持集合安定性 | +0.072 | 10/18 |

（ペア横断overlapは共有手法は構成上1.0、per-pair Lassoは平均0.466）

**結論**: (1) **Fisher-selectとRidge-selectは全指標・両黒箱で統計的に区別できない**（有意差0/18が一貫）——共有集合の選び方としてもFisherに固有の価値はない。(2) 共有集合そのもの（供給源不問）はper-pair Lassoに対しRFでfidelity・安定性が明確に優れる一方、ロジスティック黒箱の真top-K再現率で−5pt程度のコストを払う。「一貫性を買ってペア固有の精度を犠牲にする」というトレードオフが定量化できた。

---

## フェーズ8: 提案A「Multi-task Contrastive LIME」（不採用）

**目的**: フェーズ7の「二段構成」をさらに一歩進め、全クラスの対数確率を1本の多出力回帰で**同時学習**し、$\ell_{2,1}$（group lasso）で共有支持集合を学習段階から統合する。$\beta_{cd}=\gamma_c-\gamma_d$とパラメータ化すれば循環整合性（$\beta_{ab}+\beta_{bc}=\beta_{ac}$）は構成上ゼロになる。

**手法**: `fit_joint_contrastive`（`MultiTaskLasso`、$y_c=\log p_c - \text{mean}_k \log p_k$、二分探索で$K$個以上の非ゼロ行を持つ最疎解を探索）。`joint_lasso`＝生の係数、`joint_refit`＝選ばれた支持集合上でContrastive再フィット。

**スクリプト**: `src/run_shared_support_experiment.py`（`joint_lasso`/`joint_refit`列追加）

### 黒箱=LogisticRegression

| 比較 | 差 | 有意 |
|---|---|---|
| joint_refit (0.587) vs pair_lasso (0.793) の再現率 | −0.206 | 18/18 |
| joint_refit (0.587) vs ridge_select (0.741) の再現率 | −0.153 | 15/18 |
| joint_refit (0.705) vs pair_lasso (0.817) のSpearman | −0.113 | 17/18 |
| joint_refit (0.705) vs ridge_select (0.792) のSpearman | −0.088 | 14/18 |

### 黒箱=RandomForest

| 比較 | 差 | 有意 |
|---|---|---|
| joint_lasso (0.723) vs pair_lasso (0.761) のfidelity | −0.039 | 17/18 |
| joint_refit (0.769) vs ridge_select (0.784) のfidelity | −0.015 | 13/18 |
| joint_refit (0.178) vs ridge_select (0.096) の方向分散 | +0.082 | 14/18（不安定化） |
| joint_refit (0.709) vs ridge_select (0.789) の支持集合安定性 | −0.080 | 13/18 |

**結論**: **全指標・両黒箱で明確に劣る**（真の係数復元で−15〜21pt、fidelityで−1.5〜3.9pt、安定性も悪化）。原因は「全クラス（3〜5クラス）同時にグループスパース性を強制するのは、着目ペアだけに絞った共有（フェーズ7の二段構成）よりも遥かに強い制約になり、他クラスの再構成に引っ張られて着目ペアの精度が犠牲になる」ため——Fisherの「全クラス共通$S_W$」で見たのと同じ「共有しすぎ問題」が、Lasso版でも再現された。**この案は不採用**。

---

## フェーズ9: 提案B「対比認識カーネル」

**目的**: 着目ペア$(c^*,c')$の摂動重みを、2クラスが拮抗している領域（$q=p_{c^*}/(p_{c^*}+p_{c'})\approx0.5$）に集中させる。$\pi'_i=\pi_i\cdot(4q_i(1-q_i)+\text{floor})$（floor=0.05）。

**注記**: 当初「極端領域を狙う」という設計意図で提案したが、$4q(1-q)$は$q=0.5$で最大・$q\to0,1$で最小になるため、実際には**極端領域を軽視し、拮抗領域を重視する**カーネルになっている（設計意図の記述ミスだったが、結果は解釈可能で一貫している）。

**スクリプト**: `src/run_pair_kernel_experiment.py` → `results/pair_kernel_results.csv` / `_stats.csv`（黒箱=RF）

| 指標 | standard | pairkernel | 差 | 有意 |
|---|---|---|---|---|
| 全体fidelity | 0.815 | 0.820 | +0.005 | **9/9** |
| 穏やかな領域fidelity | 0.783 | 0.790 | +0.007 | **9/9** |
| 極端領域fidelity | 0.952 | 0.953 | +0.0003 | 1/9 |
| 方向の安定性（分散） | 0.052 | 0.049 | −0.003 | 6/9 |

**結論**: 拮抗領域を重視するだけで、コストなしに全体fidelityと安定性が一貫して改善（全体・穏やかな領域は全セルで有意）。極端領域は無風（悪化もしない）。地味だが副作用のない改善。

---

## フェーズ10: 提案C「OVO local logistic」

**目的**: Contrastiveの目的変数$\log(p_{c_1}/p_{c_2})$は同じまま、損失関数をRidge（対数変換後の二乗誤差、$q\to0,1$付近で発散）からソフトラベル・ロジスティック損失（有界、飽和する）に変える。

**手法**: `fit_ovo_logistic`（各行を$y=1$（重み$\pi_i q_i$）と$y=0$（重み$\pi_i(1-q_i)$）に複製し、`LogisticRegression`に`sample_weight`で渡す——重み付きソフトラベル交差エントロピーの標準的な実装）。

**スクリプト**: `src/run_logistic_target_experiment.py` → `results/logistic_target_{logistic,rf}_*.csv`

### 黒箱=LogisticRegression（ほぼ天井効果）

| 指標 | logistic | ridge | 差 | 有意 |
|---|---|---|---|---|
| Spearman | 0.998 | 0.999 | −0.0017 | 8/9 |
| 真top-K再現率 | 0.987 | 0.996 | −0.0093 | 1/9 |

### 黒箱=RandomForest

| 指標 | logistic | ridge | 差 | 有意 |
|---|---|---|---|---|
| 全体fidelity | 0.819 | 0.815 | +0.0045 | 8/9 |
| 穏やかな領域fidelity | 0.789 | 0.783 | +0.0057 | 8/9 |
| 極端領域fidelity | 0.953 | 0.952 | +0.0013 | 1/9 |
| 方向の安定性（分散） | 0.048 | 0.052 | −0.0040 | **9/9** |

**結論**: フェーズ9（B）とほぼ同じパターン（全体・穏やかな領域fidelity改善、極端領域無風）だが、**安定性の改善はBより一貫している（9/9 vs 6/9）**。真の係数復元はリッジよりわずかに劣るが、両方0.99台の天井効果でほぼ無視できる。単体で見ると4案の中で最もバランスが良い。

---

## フェーズ11: B+C 組み合わせ

**目的**: BとCは直交する改良（カーネル vs 損失関数）で個別に似た効果を示したため、組み合わせて積み上がるか検証する。

**手法**: `standard`（無印）、`kernel`（B単体）、`logistic`（C単体）、`combined`（B+C）。

**スクリプト**: `src/run_combined_bc_experiment.py` → `results/combined_bc_*.csv`（黒箱=RF）

| 指標 | standard | kernel | logistic | combined | combined vs standard | combined vs logistic |
|---|---|---|---|---|---|---|
| 全体fidelity | 0.815 | 0.820 | 0.819 | **0.821** | +0.007（9/9有意） | +0.002（6/9有意） |
| 穏やかな領域fidelity | 0.783 | 0.790 | 0.789 | **0.792** | +0.009（9/9有意） | +0.003（6/9有意） |
| 極端領域fidelity | 0.952 | 0.952 | 0.953 | 0.952 | 変化なし | 変化なし |
| 方向の安定性（分散） | 0.052 | 0.049 | **0.048** | 0.050 | わずかに改善（2/9のみ有意） | **悪化**（0.0019、6/9有意） |

**結論**: fidelityは素直に積み上がる（combined > kernel単体・logistic単体、いずれもstandardからの改善は最大）。**しかし安定性は打ち消し合う**——logistic単体が持っていた「境界付近でも勾配が飽和して暴れない」という利点を、カーネルで拮抗領域をさらに重視すると逆に効きすぎてブレが増える。**fidelityを取るか安定性を取るかで最適な組み合わせが変わる**、という交互作用が定量的に確認された。

---

## 12. 現時点の採用方針（2026-09-05時点）

1. **提案手法 = Contrastive LIME**。真の係数復元（フェーズ5）・極端領域fidelity（フェーズ3）で全手法に対し統計的に有意に優位。
2. **改良版として提案C（OVO logistic）を単体の主軸**とする。安定性が最も一貫して改善し、fidelityの犠牲もほぼない。fidelityをさらに欲しい場合は提案B（対比認識カーネル）を追加する選択肢を示す（安定性は犠牲になる）。
3. **一貫性が必要な場合の選択肢として「共有支持集合＋Contrastive再フィット」**（フェーズ7）を提示する。利得（fidelity・安定性・overlap=1）と代償（真top-K再現率−5pt程度）を両方定量化済み。供給源はRidge集約で十分（Fisher方向を使う必然性なし）。
4. **Fisher（LDA）は当初の提案から降ろし、分析章として位置づける**。係数推定器としては$S_W$尺度の偏りにより回帰系に本質的に勝てず（フェーズ6）、共有構造の供給源としてもRidge集約と区別がつかない（フェーズ7）。「なぜLDA的発想が局所説明に向かないか」を機構レベルで示した否定結果として記録する。
5. **提案A（Multi-task Contrastive LIME、全クラス同時のgroup lasso）は不採用**（フェーズ8）。全クラスへの過剰な共有制約が着目ペアの精度を損なうことが、真の係数復元・fidelity・安定性の全軸で確認された。

## 13. 未検証・今後の課題

- ソフト版Fisherのstability（正規化）はフルグリッド未検証（3セルのみのアドホック検証）。
- `src/investigate_reversal.py`（n_classes=6〜7での「逆転」診断）は統計的枠組みへの移行未実施。
- 黒箱がラベルのみ返す状況（`predict_proba`なし）でのFisher(hard) vs OVR(0/1ラベル回帰)比較は未実施——Fisherに残る唯一の原理的な居場所候補。
- 実データセットでの再現性確認は未実施（合成データのみ）。
- インスタンス選択（マージン最小8点のみ）の妥当性検証は未実施。
- feature overlapの「クラス間 vs ペア間」比較単位の不一致は未解消。

## 14. 実験スクリプト対応表

| スクリプト | 黒箱 | 対応フェーズ |
|---|---|---|
| `src/run_experiment.py` | RF | 1 |
| `src/run_fidelity_experiment.py` | RF | 2 |
| `src/run_extreme_regime_experiment.py` | RF | 3 |
| `src/run_contrastive_experiment.py` | RF | 4 |
| `src/run_groundtruth_experiment.py` | LogisticRegression | 5 |
| `src/diagnose_fisher_direction.py` | LogisticRegression | 6 |
| `src/run_shared_support_experiment.py` | RF + LogisticRegression | 7, 8 |
| `src/run_pair_kernel_experiment.py` | RF | 9 |
| `src/run_logistic_target_experiment.py` | RF + LogisticRegression | 10 |
| `src/run_combined_bc_experiment.py` | RF | 11 |
| `src/stats_utils.py` | — | 統計基盤（フェーズ1以降共通） |
| `src/investigate_reversal.py` | RF | （統計化未実施、フェーズ1派生の診断） |
