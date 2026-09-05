# Project Summary

<!--
この文書はプロジェクト全体の現在像を示す長期的なまとめです。
一時的な作業履歴ではなく、別端末・別セッションでも必要になる恒久情報を記載してください。
-->

## プロジェクトの目的

多クラス分類問題におけるLIMEの拡張を検討する修論研究。従来のLIMEの多クラス対応（one-vs-rest：クラスごとに独立した線形サロゲートをフィットする）が抱える問題点を整理し、代替手法としてFisher判別分析（LDA）を局所サロゲートとして使う方式を提案・検証する。

## 現在の状態

- 理論設計は一通り完了。摂動生成・黒箱への問い合わせ・近接度重み付けはLIMEと共通のまま、「サロゲートをフィットする」ステップだけをFisher LDAに置き換える設計。
- one-vs-rest LIME vs Fisher LIMEのconsistency・stability比較実験を実装・完走済み（グリッド実験2回、診断実験1回）。
- **重要な理論修正あり**：当初の「推移律が崩れる」という問題提起は数学的に成立しないことが判明（任意の実数の大小比較は常に推移的なため）。
- **consistencyの正しい定義はLIMEtree（Sokol & Flach 2025）の一次資料に基づく**：「モデル同士が共通構造を共有しない・異なる特徴量部分集合を使う」ことが矛盾した説明の原因（p.5）。sum-to-oneや推移律そのものではない。
- **査読済み文献を中心に関連研究を再整理済み**（`docs/RELATED_WORK.md`）。pairwise条件付き確率、log-ratio、多クラス同時サロゲート、共有特徴選択、LDAの局所説明への導入にはそれぞれ先行研究がある。一方、「LIMEの局所摂動上でユーザー指定の任意クラス対の$\log(p_c/p_d)$を疎な線形モデルとして直接学習し、各代替手法と統一条件で比較する」という組合せに完全一致する査読済み手法は、2026-09-03時点の調査では未確認。
- **統計的厳密性の強化（2026-09-03）**：それまでの4本の実験ドライバは、各グリッドセルにつきデータセット・分類器の抽選が1回だけで、8インスタンスの平均を信頼区間・検定なしで報告していた（＝そのデータセットがたまたま非典型だった可能性を区別できなかった）。`src/stats_utils.py`を新設し、各グリッドセルを**独立したデータセット抽選20回（N_DATASET_SEEDS=20）**で反復し、seedごとにインスタンス平均を取ってから（擬似反復の回避）ブートストラップ信頼区間・対応ありWilcoxon符号順位検定・Holm-Bonferroni多重比較補正を適用する方式に、主要4実験（`run_experiment.py`, `run_contrastive_experiment.py`, `run_fidelity_experiment.py`, `run_extreme_regime_experiment.py`）を作り直した。以下の結論はこの統計的枠組みで再検証済み（詳細は`docs/RECENT_WORK.md`参照）。`src/investigate_reversal.py`（診断スクリプト）はこの反復方式にまだ移行しておらず、単一シードのままである点に注意。
- 実測で確認済みの結論（詳細は`docs/RECENT_WORK.md`参照）：
  - **stability（正規化ペア方向ベクトルの分散）**：**Fisher（ハード版）はone-vs-restより不安定**という結果が、20独立シード全てで一貫して再現された（全9グリッドセルでp≈0.000002、Holm補正後も有意）。ソフトラベル版のstabilityは今回のフルグリッド再検証の対象外のまま（依然としてアドホックな3セル検証のみ、フルグリッドでの再現は未実施）。
  - feature overlap・fidelity実験は尺度の問題を抱えていない（feature overlapは順位のみ使うため尺度不変、fidelityは両手法とも適切な単位の確率値に変換してから比較しているため）。
  - **【訂正】feature overlap（LIMEtreeの主張の直接的な操作化）**：以前は「クラス数が多いと逆転する（OVRが優位になる）」と記載していたが、統計的に厳密な再検証では、Fisherは検証した全18グリッドセル×K（n_classes=3〜5の範囲）で数値上は一貫してOVRより高い（優位）。ただし有意性はクラス数が増えるほど失われ、n_classes=5の多くのセルで非有意になる。**OVRが統計的に有意に上回るケースは1つも観測されなかった**——正確には「逆転」ではなく「優位性の消失（クラス数増加につれ差が検出できなくなる）」。より高いクラス数（6〜7）での真の逆転を主張する`investigate_reversal.py`の診断結果は、異なる設計（n_informative固定）かつ単一シードのままなので、この訂正と直接矛盾はしないが再検証が必要。
  - ソフトラベル版Fisher（`fit_fisher_soft`）はクラス欠落は解消するが、feature overlap自体は多くの条件でむしろ悪化する（クラス間の分離が弱まるため）。単純な優劣ではなくトレードオフとして扱うべき。
  - **fidelity（忠実性）は測り方で結論が変わる**。Fisherを標準の多クラス確率分類器として評価すると one-vs-rest に大きく劣る（Hellinger損失、全9セル×hard/soft版で統計的に有意にOVRが優位、p≈0.000002）。しかし提案アルゴリズムが実際に使う量（黒箱が決めた予測クラス$c^*$と競合クラス$c'$のペア比較の符号一致率）で測り直すと、OVRはFisher(hard)より9セル中4セルで有意に優位（n_classes・n_featuresが大きいセルに集中）——**「ほぼ互角」ではなく「小さいグリッドでは互角、大きいグリッドではOVRがわずかに有意に優位」**という、以前より精緻化された結論。**Fisherの忠実性の弱さは「絶対確率値としての解釈」でより顕著**、という切り分けは維持。
  - **現時点の全体像**：ハード版Fisherに明確な優位性はない（stability・fidelity(絶対値・ペア符号どちらも)のいずれも統計的に有意に劣る。feature overlapのみ数値上は優位を保つが有意性は不安定）。ソフト版はfidelity(絶対値)でハード版より統計的に有意に優れる（全9セル）が、stabilityはフルグリッド未検証のまま。「Fisherが勝つ」という単純な主張ではなく、**指標ごとに条件付きで一長一短がある**、という正直な立ち位置は変わらない。
  - **4手法目「Contrastive LIME」**：one-vs-restを2本フィットして引くのではなく、$\log(p_A/p_B)$を直接回帰する方式。fidelity（ペア符号一致率）はOVRとほぼ完全な同点（9セル中8セルで非有意、差は0.002〜0.004）——統計的に確認済み。stabilityはOVRとほぼ同格（3/9セルで有意にContrastiveがわずかに安定、それ以外は非有意）、Fisherより全9セルで有意に安定。feature overlapはOVRとほぼ非有意差（3/18で有意、小さい）、Fisherとは低クラス数・高次元セルで有意差あり（比較単位がクラス間 vs ペア間で異なる点は未解消）。
  - **極端確率領域での検証**：競合する2クラスの一方の確率が0に近い局所領域でfidelityを測ると、Contrastiveのone-vs-restに対する優位性は全9セル中6セルで統計的に有意（クラス数・次元数が大きいセルに集中、n_classes=3では非有意）。**新しい発見**：この優位性は無償ではなく、穏やかな領域ではContrastiveがOVRよりわずかに、しかし統計的に有意に劣る場合がある（9セル中3セルで有意、差は-0.002〜-0.008）——極端領域での優位性と引き換えに穏やかな領域で小さなコストを払っている可能性。一方、**Fisher(hard)はこの極端領域で全9セルにおいて統計的に有意に劣化**（p<0.003）しており、feature overlap・stabilityでも見られた「ハードラベルによるサンプル飢餓」が3つ目の独立した文脈・厳密な検定で再確認された。
  - **Fisher方向の失敗要因分解（2026-09-05、`diagnose_fisher_direction.py`）**：真の係数とのSpearman ρで、重心差のみ0.885/0.907（hard/soft）→ pooled $S_W^{-1}$ 0.953/0.983 → Contrastive 0.999。統計的に確定した内訳：(1) $S_W^{-1}$は必要（+0.05〜0.08、単位補正）。(2) **クラス横断のプーリングはほぼ無料**（ペア限定$S_W$との差≤0.005、多くのセルで非有意）——共有構造という売りは精度を犠牲にしていない。(3) **ハードラベルが最大の損失源**（soft化で+0.02〜0.04）。(4) 重心を連続応答（log-oddsとの共分散）に置き換えても+0.005程度。(5) **残りの差0.985→0.999は本質的**：クラス内散布$S_W$を尺度に使うこと自体が、判別的な目的（log-oddsの勾配、正しい尺度は総散布$S_T$）に対して偏った正規化になる。$S_W=S_T-S_B$はクラス間方向を過剰補正する。これはFisherの定義に内在するので、Fisherを係数推定器として使う限り消えない。**帰結**：Fisherの残された役割は「係数」ではなく「共有構造（特徴集合）」。$S_W$をRidgeの罰則行列にする案（`OVO_LIME_METHODS.md`のFisher-metric系）は、この尺度偏りを回帰に持ち込むだけなので不採用。
  - **グラウンドトゥルース検証（Rahnama et al. 2024型、2026-09-03追加）**：黒箱をRandomForestから多項ロジスティック回帰に差し替え（＝真の対数オッズ係数$\theta_{c^*}-\theta_{c'}$が既知になる）、各手法の推定係数と真の係数のSpearman順位相関を測定。これは今までの「fidelity（局所再現性）」とは質的に異なる軸で、「正しい特徴に重みを置けているか」を直接検証する。**結果はContrastiveの最も明確な勝ちどころになった**：全グリッド平均でContrastive ρ=0.999、Fisher(soft) ρ=0.983、OVR ρ=0.976、Fisher(hard) ρ=0.953という順位が、ほぼ全ペア・全9セルで統計的に確定した（Contrastive vs 他3手法：全9セルで有意にContrastiveが優位、p≦0.0003）。理論的にも整合する：Contrastiveは対数オッズを直接回帰するため、黒箱が対数オッズについて線形（多項ロジスティック回帰）である限り真の係数をほぼ完璧に復元できる。OVRは生の確率（softmaxで非線形）を回帰するため一致度が下がる。Fisher(hard)は最下位で、これまでの実験（stability・極端領域fidelity）で見えていた「ハードラベルのサンプル飢餓」問題が別角度から再確認された。

## 主要な構成

- `src/perturbation.py`: one-vs-rest LIMEとFisher LIMEに共通の摂動サンプリング（LIMEのデフォルト方式：Gaussianノイズ×特徴量標準偏差、指数カーネルで近接度重み付け）。両手法を公平に比較するため、この摂動生成ステップだけは完全に共有する設計。
- `src/surrogates.py`: サロゲートフィッティング関数群。
  - `fit_onevsrest`: クラスごとに独立な重み付きRidge回帰（全特徴量使用、intercept込み）。
  - `fit_onevsrest_lasso`: クラスごとに独立な重み付きLasso選択（`lasso_path`スタイル、二分探索でK個以上の非ゼロ係数を持つ最疎解を求めて上位K個を採用）。真の特徴量部分集合選択を再現するため、feature overlap実験で使用。
  - `fit_fisher`: 3クラス（以上）共通のプールされたクラス内散布行列S_Wを使うFisher LDAサロゲート（ハードラベル版）。shrinkage正則化あり。ペア方向`v(X,Y)=S_W^{-1}(μ_X-μ_Y)`と one-vs-rest形式の`onevsrest_direction(c)=S_W^{-1}(μ_c-μ_¬c)`を計算するヘルパーを返す。
  - `shared_support_fisher_soft` / `shared_support_ridge` / `fit_contrastive_on_support`（2026-09-05、提案）: 二段構成サロゲート。Stage 1で全クラス共通のtop-K特徴集合を1つ選び（Fisher soft方向の集約、または対照としてOVR Ridge係数の集約）、Stage 2でその集合に制限したContrastive log-odds Ridgeを各ペアにフィットする。ペア横断のfeature overlapは構成上1。
  - `fit_fisher_soft`: ソフトラベル版。`π_i・f_c(z_i)`を重みとして使い、argmaxによるハードラベル化を行わない。クラスが局所近傍から丸ごと欠落する問題を解消するが、feature overlap自体は改善しないことがある（トレードオフ、詳細は`docs/RECENT_WORK.md`）。
  - `fit_contrastive`: 「Contrastive LIME」。$\log((p_{c1}+\varepsilon)/(p_{c2}+\varepsilon))$を目的変数にした重み付きRidge回帰（ペアごとに独立フィット、Fisherの共有S_Wは使わない）。
  - `fit_contrastive_lasso`: 同じ目的変数でのLasso選択版（feature overlap実験用）。
  - `top_k_indices`: 上位K個（絶対値）の特徴量インデックス集合を返すヘルパー。
- `src/metrics.py`: consistency・stability指標。
  - `sum_to_one_deviation` / `sum_to_one_deviation_topk`: 全特徴量時と top-K切り詰め後のsum-to-one逸脱。
  - `mean_pairwise_feature_overlap`: LIMEtreeの主張（共通構造の有無）を操作化した、クラス間top-K特徴量集合の平均Jaccard重なり。
  - `total_variance`: stability指標（ペア方向ベクトルの分散のtrace）。**警告**：Fisherとone-vs-restの出力ベクトルは尺度が異なるため、これを直接比較するのは誤り（詳細は`docs/RECENT_WORK.md`）。
  - `total_variance_normalized`: 尺度不変のstability指標（単位ベクトルに正規化してから分散を取る）。**手法間の比較には必ずこちらを使う**。
  - `mean_norm`: ベクトルの平均大きさ（尺度差を確認するための参考情報）。
  - `transitivity_violation_rate`: **理論的に常に0になるため実験では未使用**。docstringに理由を明記した上でコードのみ残置。
- `src/stats_utils.py`: 実験ドライバ共通の統計ヘルパー。`bootstrap_ci`（パーセンタイル・ブートストラップ信頼区間）、`paired_wilcoxon`（対応ありWilcoxon符号順位検定＋matched-pairs rank-biserial効果量）、`holm_bonferroni`（多重比較補正）、`compare_methods`（グリッドセルごとにseedレベル平均を独立サンプルとして扱い、これらを組み合わせて統計比較表を作る高水準関数）。擬似反復（同一データセット内の複数インスタンスを独立サンプル扱いすること）を避ける設計上の理由はモジュールのdocstringに詳しく記載。
- `src/run_experiment.py`: 次元数×クラス数×Kのグリッドで one-vs-rest / Fisher(hard) を比較する実験ドライバ。各グリッドセルにつき独立したデータセット抽選を`N_DATASET_SEEDS=20`回繰り返す。完走済み、生データは`results/experiment_results.csv`、統計比較は`results/experiment_stats.csv`に出力。
- `src/investigate_reversal.py`: feature overlapでFisherの優位性が縮小・逆転する条件（高クラス数、n_classes=6〜7）の原因を切り分ける診断スクリプト。ハード版・ソフト版Fisherを同時比較する。**まだ`N_DATASET_SEEDS`方式の統計的厳密化の対象外**（単一シードのアドホック診断のまま）。
- `src/fidelity.py`: 忠実性（fidelity）評価用の確率変換・損失関数。`onevsrest_predict_proba`（Ridge出力のクリップ＋正規化）、`fisher_predict_proba`（LDA確率モデルによる擬似確率、`LinearDiscriminantAnalysis.predict_proba`と同じ考え方）、`weighted_hellinger_loss`（SLISEMAP Eq.11と同じ二乗Hellinger距離）。
- `src/run_fidelity_experiment.py`: 次元数×クラス数グリッドでone-vs-rest / Fisher(hard) / Fisher(soft)の忠実性（Hellinger損失）を比較する実験ドライバ。`N_DATASET_SEEDS=20`で反復。結果は`results/fidelity_results.csv`、統計比較は`results/fidelity_stats.csv`。
- `src/run_contrastive_experiment.py`: one-vs-rest / Fisher(hard) / Contrastiveの3手法を、fidelity（ペア符号一致率）・stability（正規化）・feature overlapの3指標で同時比較するグリッド実験。`N_DATASET_SEEDS=20`で反復。結果は`results/contrastive_results.csv`、統計比較は`results/contrastive_stats.csv`。
- `src/run_extreme_regime_experiment.py`: 同じ局所近傍を「競合2クラスの確率が両方とも極端（一方が閾値未満）」と「穏やか」に分割し、fidelityを領域別に比較するグリッド実験。`N_DATASET_SEEDS=20`で反復。結果は`results/extreme_regime_results.csv`、統計比較は`results/extreme_regime_stats.csv`。
- `src/diagnose_fisher_direction.py`: Fisher方向$S_W^{-1}(\mu_c-\mu_d)$がなぜ回帰より真の係数を復元できないかを部品分解する診断（重心差のみ／pooled $S_W$／ペア限定$S_W$／対角$S_W$／連続応答版、hard/soft）。黒箱はロジスティック回帰。結果は`results/diagnose_fisher_direction_{results,stats}.csv`。
- `src/run_shared_support_experiment.py`: 提案（二段構成：Fisherで共有特徴集合を選び、Contrastive回帰で係数を出す）の評価。対照は同じ二段構成でStage 1をOVR Ridge係数集約にした版と、per-pair Contrastive Lasso。黒箱はロジスティック（真のtop-K再現率・Spearman）とRF（ペア符号fidelity・特徴集合安定性・方向安定性・ペア横断overlap）。結果は`results/shared_support_{logistic,rf}_{results,stats}.csv`。
- `src/run_groundtruth_experiment.py`: Rahnama et al. (2024)型のグラウンドトゥルース検証。黒箱を`LogisticRegression(multinomial)`に差し替え、真の対数オッズ係数$\theta_{c^*}-\theta_{c'}$と各手法（OVR / Fisher hard / Fisher soft / Contrastive）の推定係数のSpearman順位相関（`metrics.pairwise_coef_spearman`）を測る。`N_DATASET_SEEDS=20`で反復。結果は`results/groundtruth_results.csv`、統計比較は`results/groundtruth_stats.csv`。
- `docs/OVO_LIME_METHODS.md`: OVR、pairwise probability、pairwise log-odds、OVO Logistic-LIME、Contrastive LIME、OVO Fisher-LIME、共同学習の定式化と評価案。
- `docs/RELATED_WORK.md`: 多クラスLIMEと対比的局所説明に関する査読済み文献、各研究との重なり、安全な新規性の位置づけ、比較実験への示唆。
- `.venv/`: Python仮想環境（`.gitignore`で除外、コミット対象外）。

## セットアップと実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/run_experiment.py         # メイングリッド実験。結果は results/ にCSV出力
python3 src/investigate_reversal.py   # 高クラス数での逆転を調べる診断実験
```

必要パッケージ: numpy, scipy, scikit-learn, pandas, lime, matplotlib（`requirements.txt`参照）。

## 重要な設計判断

- **one-vs-rest LIMEとFisher LIMEは、同一の摂動サンプル・同一の近接度重みを使う**（`src/perturbation.py`で共有）。これは元の問題提起の前提条件（「摂動データは同じサンプリングとする」）を実験でも忠実に再現するため。
- one-vs-rest側の黒箱への問い合わせターゲットは各クラスの確率 `predict_proba` の各列。Fisher側は主にハードラベル（`argmax`）でクラスを割り当ててから重心・散布行列を計算するが、ソフトラベル版（`fit_fisher_soft`）も実装済み。どちらが良いかは条件依存（`docs/RECENT_WORK.md`参照）。
- Fisher側のS_Wは正則化のため `S_W + ε·(trace(S_W)/d)·I` のshrinkageを適用している（`shrinkage`パラメータ、デフォルト1e-3）。
- feature overlap実験（consistency検証）では、one-vs-rest側は真のLasso選択（`fit_onevsrest_lasso`）を使う。単純なRidge事後top-K切り詰めでは差が出なかったため（詳細は`docs/RECENT_WORK.md`）。
- 合成データは`n_redundant = n_features - max(3, n_classes)`で、相関の強い冗長特徴量を意図的に含める。Lassoの特徴量選択不安定性（相関特徴量群からの恣意的な選択）を再現するため。

## データと環境依存

- 合成データ（`sklearn.datasets.make_classification`）を使用。実データセットは今のところ使用していない。
- 黒箱モデルは`RandomForestClassifier`（非線形決定境界を作るため）。
- 秘密情報・環境依存の外部サービスなし。

## 既知の制約・リスク

- **手法間で出力ベクトルの生の大きさ・分散を直接比較しない**。Fisherの$v=S_W^{-1}(\mu_X-\mu_Y)$には自然な尺度がなく、one-vs-restの回帰係数と直接比較すると尺度差だけで数十〜数百倍の見かけの差が生まれる。比較する際は必ず正規化（`total_variance_normalized`など）を使うか、両手法とも適切な単位（実際の確率値など）に変換してから比較すること。
- `transitivity_violation_rate`（推移律違反率）という指標は理論的に常に0になる無意味な指標であることが判明。使用しない（docstringに理由明記済み）。
- sum-to-one（$\hat P(A)+\hat P(B)+\hat P(C)=1$）は、one-vs-rest側が全特徴量を使った同一設計の回帰であれば厳密に成立してしまう。崩れを見たい場合はtop-K切り詰め（`sum_to_one_deviation_topk`）で検証する。
- **Fisher（ハードラベル版）はクラス数が多い、または局所サンプル数が少ない場合、局所近傍にそのクラスのサンプルが1つも現れず、そのクラスの説明が丸ごと欠落することがある**（`min_class_count_avg`が0近くになる事例を確認済み）。ソフトラベル版はこの欠落を解消するが、feature overlap自体はむしろ悪化する条件が多い。単純にどちらかを常に推奨できる状態ではない。
- Fisher LDAのS_Wは次元が高くローカルサンプル数が少ない場合に特異・悪条件になりうる（shrinkage正則化で緩和しているが、パラメータの妥当性は未検証）。

## 今後の大きな課題

- consistencyの主張を、実測で裏付けられる正確な形（LIMEtreeの「共通構造の有無」の定義に基づく、条件付きの主張）に修論の記述を修正する。
- ハード版・ソフト版Fisherのトレードオフを理論的に説明する（重心間距離・S_Bの直接比較など）。ハイブリッド案（局所サンプルが少ないクラスだけソフトにフォールバック）の検討。
- ソフト版の正規化stabilityを、今回整備した統計的枠組み（`N_DATASET_SEEDS`反復＋`stats_utils.py`）でフルグリッド再検証し、恒久的なスクリプトとして組み込む（現状3セルのアドホック検証のみ）。
- `src/investigate_reversal.py`（n_classes=6〜7での「逆転」診断）を同じ統計的枠組みで再検証する。今回の`run_experiment.py`再検証（n_classes 3〜5では「逆転」ではなく「優位性の消失」）との整合性を確認する必要がある。
- feature overlapの「クラス間 vs ペア間」という比較単位の不一致を解消し、Contrastive・Fisher・one-vs-restを公平に再比較する。
- one-vs-rest, Fisher(hard/soft), Contrastiveの4手法×fidelity・stability・feature overlap・sum-to-one・極端領域fidelity・グラウンドトゥルース復元の、統計的に裏付けられた結果を統合し、修論の主張として文章化する。「Fisherが優れている」という単純な主張ではなく、条件付き・トレードオフとして誠実に書く必要がある。グラウンドトゥルース検証はContrastiveにとって最も明確に勝てる指標なので、修論の主張の中心に据えることを検討する。
- インスタンス選択（マージン最小の8点のみ）が結果を偏らせていないか検証する（今回のスコープ外、ユーザーの優先度確認により統計的厳密性のみ対応）。
- 実データセットでの再現性確認。
