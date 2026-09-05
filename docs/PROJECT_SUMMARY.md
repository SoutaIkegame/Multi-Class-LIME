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
- 全特徴でサロゲートを一度だけ学習し、標準化寄与の上位特徴を後から表示するdense→post-hoc方式を、合成データとDigitsで検証済み。表示特徴上での再学習は行わず、説明点を原点とする標準化座標で係数を隠すため、説明点での予測は保持される。
- **重要な理論修正あり**：当初の「推移律が崩れる」という問題提起は数学的に成立しないことが判明（任意の実数の大小比較は常に推移的なため）。
- **consistencyの正しい定義はLIMEtree（Sokol & Flach 2025）の一次資料に基づく**：「モデル同士が共通構造を共有しない・異なる特徴量部分集合を使う」ことが矛盾した説明の原因（p.5）。sum-to-oneや推移律そのものではない。
- **査読済み文献を中心に関連研究を再整理済み**（`docs/RELATED_WORK.md`）。pairwise条件付き確率、log-ratio、多クラス同時サロゲート、共有特徴選択、LDAの局所説明への導入にはそれぞれ先行研究がある。一方、「LIMEの局所摂動上でユーザー指定の任意クラス対の$\log(p_c/p_d)$を疎な線形モデルとして直接学習し、各代替手法と統一条件で比較する」という組合せに完全一致する査読済み手法は、2026-09-03時点の調査では未確認。
- 実測で確認済みの結論（詳細は`docs/RECENT_WORK.md`参照）：
  - **【重要な訂正あり】stability（ペア方向ベクトルの分散）の当初の結論（Fisherが15〜100倍安定）は測定上の誤りだった**。Fisherの方向ベクトル$v=S_W^{-1}(\mu_X-\mu_Y)$はone-vs-restの回帰係数差より常に6〜15倍小さい大きさで、これは尺度の違い（Fisherの出力に自然な単位がないこと）によるもの。分散を尺度不変な形（単位ベクトルに正規化してから比較）で測り直すと、**Fisher（ハード版）はone-vs-restより2〜4倍不安定**という逆の結果になった。ハードラベル版でのこの不安定性は、局所近傍でのクラスごとのサンプル飢餓が原因と考えられ、**ソフトラベル版に切り替えるとone-vs-restとほぼ同等の安定性に回復する**ことを確認済み（詳細な検証は3セルのみ、フルグリッドでの再現は未実施）。
  - feature overlap・fidelity実験は、この尺度の問題を抱えていないことを確認済み（feature overlapは順位のみ使うため尺度不変、fidelityは両手法とも適切な単位の確率値に変換してから比較しているため）。以下の結論はそのまま有効。
  - **feature overlap（LIMEtreeの主張の直接的な操作化）はFisherが優位になる条件と、逆転する条件が両方ある**。真のLasso選択＋相関の強い特徴量がある設定では概ねFisher優位だが、クラス数が多いとFisherのハードラベル設計がクラスを局所近傍から丸ごと欠落させる問題があり、これが逆転の主因（冗長特徴量の予算不足ではない）。
  - ソフトラベル版Fisher（`fit_fisher_soft`）はこの欠落は解消するが、feature overlap自体は多くの条件でむしろ悪化する（クラス間の分離が弱まるため）。単純な優劣ではなくトレードオフとして扱うべき。
  - **fidelity（忠実性）は測り方で結論が変わる**。Fisherを標準の多クラス確率分類器として評価すると one-vs-rest に大きく劣る（Hellinger損失で2〜6倍、argmax一致率でも76.5%対55%前後）。しかし提案アルゴリズムが実際に使う量（黒箱が決めた予測クラス$c^*$と競合クラス$c'$のペア比較の符号）で測り直すと、one-vs-rest 81.3% vs Fisher 80.4%とほぼ互角。**Fisherの忠実性の弱さは「絶対確率値としての解釈」に限定され、「2クラス比較の方向」としての忠実性はone-vs-restと同等**、という切り分けが重要。
  - **現時点の全体像**：当初考えていたほど明確な優位性はハード版Fisherにはない（stability・feature overlap(高クラス数)・fidelity(絶対値)のいずれも劣る）。ソフト版はstability・fidelityでone-vs-restとほぼ互角まで回復するが、feature overlapは悪化しやすい。「Fisherが勝つ」という単純な主張ではなく、**指標ごとに条件付きで一長一短がある**、という正直な立ち位置。
  - **4手法目「Contrastive LIME」を追加検証**：one-vs-restを2本フィットして引くのではなく、$\log(p_A/p_B)$を直接回帰する方式（ユーザー提案）。理論的には忠実性で明確に勝つと予想したが、実測ではone-vs-restとほぼ完全な同点だった（同一設計のRidge回帰なら「引き算」と「直接回帰」が線形性により数学的に一致するため）。stabilityはone-vs-restと同格（Fisherより良い）。feature overlapはFisherとone-vs-restの中間だが、比較単位（クラス間 vs ペア間）が異なるため直接の優劣比較ではない。
  - **極端確率領域での検証**：競合する2クラスの一方の確率が0に近い局所領域に絞ってfidelityを測ると、Contrastiveのone-vs-restに対する優位性は「弱いが本物」（平均+0.8ポイント、穏やかな領域では約0ポイント）。効果量は小さく、決定的な差別化ポイントとまでは言えない。一方、**Fisher(hard)はこの極端領域で一貫して3〜4ポイント劣化**しており、これはfeature overlap・stabilityでも見られた「ハードラベルによるサンプル飢餓」が3つ目の独立した文脈で再確認された形になる。
  - **held-out OVO評価（2026-09-03）**：独立な評価用摂動を使い、OVR確率、OVR logit、OVO条件付き確率$q_{c,d}=p_c/(p_c+p_d)$、OVO log-ratioを180説明点で比較。全特徴ではOVO条件付き確率がOVR確率よりpairwise確率MSEを8.46%低減し、$R^2$を0.036、勝敗一致率を0.34ポイント、方向安定性（cosine）を0.004改善。5特徴ではOVRが平均7.15個の特徴（2説明の和集合）を使うのに対し、OVOは5個でMSEを5.63%低減し、勝敗一致率はほぼ同等。ただし疎なOVOは方向安定性がOVRより低く、3特徴では勝敗一致率も約0.52ポイント低下した。**OVOは全面的に優位ではなく、簡潔性・pairwise確率fidelityとstabilityのトレードオフ**。log-ratioは係数解釈が明快だが、この実験では$q_{c,d}$直接回帰の方が確率fidelityに優れた。
  - **OVO計算時間スケーリング（2026-09-03）**：BB出力取得後の疎サロゲート学習時間を測定。$M=1000,D=50,K=10$で$n=3\to100$とすると、OVR全クラスは3.8 msから148.2 ms、OVO全ペアは5.9 msから7126.0 msへ増加し、$n=100$で全ペアOVOはOVRの48.1倍。一方、foilを先に選ぶOVO 1本は1.2--1.6 ms程度でクラス数にほぼ依存しない。同じ指定ペアのためにOVRを2本学習する条件は2.4--3.3 msで、OVO 1本のおよそ2倍。特徴次元スケーリングは$n=10,M=1000,K=10$で測定し、$D=10\to200$でOVO全45ペアは28.5 msから169.0 ms、選択1ペアは0.57 msから3.24 msへ増加した。全ペア個別疎化は$O(n^2)$で不利だが、on-demand方式ならこの問題を回避できる。時間の絶対値は環境依存であり、BB推論時間は両方式で共有可能なため除外した。
  - **dense→post-hoc表示（2026-09-03）**：合成データ18黒箱・180説明点では、検証MSEが最小となる表示数を選ぶと、平均14特徴からOVR 10.44、OVO 11.46、多クラスsoft Fisher 11.21、pairwise soft Fisher 11.46特徴へ削減し、独立testの$R^2$はdense比で同等または改善した。寄与エネルギー95%方式なら約7.4特徴まで削減できるが、$R^2$は0.008--0.018低下する。固定5特徴は平均$R^2$がdenseより約0.04--0.06低く、常に少数特徴へ切る設計は支持されない。
  - **Digits実データ（64特徴・10クラス、2黒箱・30説明点）**：検証MSE最小方式はOVR 21.6、OVO 28.0、多クラスsoft Fisher 28.5、pairwise soft Fisher 27.5特徴を表示し、denseより$R^2$が0.006--0.032改善した。小さいノイズ係数を隠すことがpost-hoc正則化として働く。95%寄与方式は全手法約22特徴で、OVO/Fisherはdenseとほぼ同等、OVRは$R^2$が0.027改善した。5特徴固定は$R^2=0.44$--$0.46$に留まり、20--30特徴の$0.56$--$0.61$より明確に悪い。
  - **pairwise soft FisherとOVO最小二乗の実質的同値性**：同じ局所重みと$q_{c,d}$ / $1-q_{c,d}$をsoft所属度に使う無正則化Fisher方向は、重み付きOLSで$q_{c,d}$を回帰した係数と厳密に同一直線になる。単体テストで相関を持つ任意入力でも確認した。実験でもOVO Ridgeとの方向cosineは合成・Digitsとも0.999999、多クラスsoft Fisherでも0.9967 / 0.9932だった。したがってpairwise Fisherのわずかな数値差は主に正則化と1次元確率校正によるもので、独自の特徴方向が得られた証拠ではない。

## 主要な構成

- `src/perturbation.py`: one-vs-rest LIMEとFisher LIMEに共通の摂動サンプリング（LIMEのデフォルト方式：Gaussianノイズ×特徴量標準偏差、指数カーネルで近接度重み付け）。両手法を公平に比較するため、この摂動生成ステップだけは完全に共有する設計。
- `src/surrogates.py`: サロゲートフィッティング関数群。
  - `fit_onevsrest`: クラスごとに独立な重み付きRidge回帰（全特徴量使用、intercept込み）。
  - `fit_onevsrest_lasso`: クラスごとに独立な重み付きLasso選択（`lasso_path`スタイル、二分探索でK個以上の非ゼロ係数を持つ最疎解を求めて上位K個を採用）。真の特徴量部分集合選択を再現するため、feature overlap実験で使用。
  - `fit_fisher`: 3クラス（以上）共通のプールされたクラス内散布行列S_Wを使うFisher LDAサロゲート（ハードラベル版）。shrinkage正則化あり。ペア方向`v(X,Y)=S_W^{-1}(μ_X-μ_Y)`と one-vs-rest形式の`onevsrest_direction(c)=S_W^{-1}(μ_c-μ_¬c)`を計算するヘルパーを返す。
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
- `src/run_experiment.py`: 次元数×クラス数×Kのグリッドで両手法を比較する実験ドライバ。完走済み、結果は`results/experiment_results.csv`等に出力。
- `src/investigate_reversal.py`: feature overlapでFisherの優位性が逆転する条件（高クラス数）の原因を切り分ける診断スクリプト。ハード版・ソフト版Fisherを同時比較する。
- `src/fidelity.py`: 忠実性（fidelity）評価用の確率変換・損失関数。`onevsrest_predict_proba`（Ridge出力のクリップ＋正規化）、`fisher_predict_proba`（LDA確率モデルによる擬似確率、`LinearDiscriminantAnalysis.predict_proba`と同じ考え方）、`weighted_hellinger_loss`（SLISEMAP Eq.11と同じ二乗Hellinger距離）。
- `src/run_fidelity_experiment.py`: 次元数×クラス数グリッドでone-vs-rest / Fisher(hard) / Fisher(soft)の忠実性を比較する実験ドライバ。結果は`results/fidelity_results.csv`。
- `src/run_contrastive_experiment.py`: one-vs-rest / Fisher(hard) / Contrastiveの3手法を、fidelity・stability（正規化）・feature overlapの3指標で同時比較するグリッド実験。結果は`results/contrastive_results.csv`。
- `src/run_extreme_regime_experiment.py`: 同じ局所近傍を「競合2クラスの確率が両方とも極端（一方が閾値未満）」と「穏やか」に分割し、fidelityを領域別に比較するグリッド実験。結果は`results/extreme_regime_results.csv`。
- `src/run_ovo_evaluation.py`: OVR確率2本、CLIMAX-style OVR logit 2本、OVO条件付き確率、OVO log-ratioを、独立な評価用局所摂動上のpairwise確率MSE・$R^2$・勝敗一致率と、再サンプリング時の方向cosine・top-K Jaccardで比較する実験。全特徴・3特徴・5特徴の予算を評価し、OVRは2つのクラス別説明の特徴和集合数も記録する。結果は`results/ovo_evaluation_*.csv`。
- `src/benchmark_ovo_runtime.py`: OVR全クラス、指定ペア用OVR 2本、OVO全ペア、foil選択済みOVO 1本について、クラス数$n$、摂動数$M$、特徴次元$D$に対する疎サロゲート学習時間を測定する。5反復の中央値・四分位範囲を保存し、短時間のon-demand条件は20回の内部反復で計時誤差を抑える。BB推論は方式間で共有されるため計測対象外。結果は`results/runtime_benchmark_*.csv`。
- `src/run_dense_posthoc_experiment.py`: OVR確率、OVO条件付き確率、多クラスsoft Fisher、pairwise soft Fisherを全特徴で一度だけ学習し、係数を再学習せずpost-hoc top-K表示する実験。合成データとDigitsに対応し、dense保持最小K・検証MSE最小K・95%寄与Kを独立hold-outで比較する。結果は`results/dense_posthoc_*.csv`。
- `tests/test_ovo_evaluation.py`: pairwise確率とlog-ratioの変換・クラス対反転、および主要評価指標の最小単体テスト。
- `tests/test_dense_posthoc_experiment.py`: post-hoc選択、説明点での予測保持、soft logistic校正、Fisher方向、適応的K選択、および無正則化pairwise soft Fisherと重み付きOLSの方向同値性を検証する単体テスト。
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
python3 src/run_dense_posthoc_experiment.py          # dense→post-hoc合成本実験
python3 src/run_dense_posthoc_experiment.py --digits --output-dir results/dense_posthoc_digits
```

必要パッケージ: numpy, scipy, scikit-learn, pandas, lime, matplotlib（`requirements.txt`参照）。

## 重要な設計判断

- **one-vs-rest LIMEとFisher LIMEは、同一の摂動サンプル・同一の近接度重みを使う**（`src/perturbation.py`で共有）。これは元の問題提起の前提条件（「摂動データは同じサンプリングとする」）を実験でも忠実に再現するため。
- one-vs-rest側の黒箱への問い合わせターゲットは各クラスの確率 `predict_proba` の各列。Fisher側は主にハードラベル（`argmax`）でクラスを割り当ててから重心・散布行列を計算するが、ソフトラベル版（`fit_fisher_soft`）も実装済み。どちらが良いかは条件依存（`docs/RECENT_WORK.md`参照）。
- Fisher側のS_Wは正則化のため `S_W + ε·(trace(S_W)/d)·I` のshrinkageを適用している（`shrinkage`パラメータ、デフォルト1e-3）。
- feature overlap実験（consistency検証）では、one-vs-rest側は真のLasso選択（`fit_onevsrest_lasso`）を使う。単純なRidge事後top-K切り詰めでは差が出なかったため（詳細は`docs/RECENT_WORK.md`）。
- 実用的な簡潔表示では、説明モデルの学習と表示特徴の選択を分離する。`run_dense_posthoc_experiment.py`は全特徴で一度学習した後、説明点を原点とする標準化座標上で係数を隠す。標準の推奨候補は、別の検証近傍でpairwise MSEが最小となるK、または単純で再現しやすい95%寄与エネルギーK。
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
- dense→post-hoc方式は「学習には全特徴を使う」ため、表示特徴だけから元のdense予測を完全に再構成できるとは限らない。固定5特徴では忠実性低下が確認され、Kはデータ・説明点ごとに選ぶ必要がある。検証選択は平均では良好でも個別testで変動するため、厳密な保証ではない。
- pairwise soft Fisherを$q_{c,d}$のsoft所属度から構成する場合、無正則化の方向は$q_{c,d}$への重み付きOLSと同一直線になる。これをFisher固有の安定性・特徴選択上の優位として主張しない。

## 今後の大きな課題

- consistencyの主張を、実測で裏付けられる正確な形（LIMEtreeの「共通構造の有無」の定義に基づく、条件付きの主張）に修論の記述を修正する。
- ハード版・ソフト版Fisherのトレードオフを理論的に説明する（重心間距離・S_Bの直接比較など）。ハイブリッド案（局所サンプルが少ないクラスだけソフトにフォールバック）の検討。
- ソフト版の正規化stabilityをフルグリッドで再検証し、恒久的なスクリプトとして組み込む（現状3セルのアドホック検証のみ）。
- feature overlapの「クラス間 vs ペア間」という比較単位の不一致を解消し、Contrastive・Fisher・one-vs-restを公平に再比較する。
- one-vs-rest, Fisher(hard/soft), Contrastiveの4手法×fidelity・stability・feature overlap・sum-to-one・極端領域fidelityの結果を統合し、修論の主張として文章化する。「Fisherが優れている」という単純な主張ではなく、条件付き・トレードオフとして誠実に書く必要がある。
- 実データセットでの再現性確認。
- dense→post-hoc方式を複数の実データ・黒箱・説明点確信度で再現し、表示Kの選択規則を事前固定できるか検証する。Digitsは実施済みだが独立黒箱は2個のみ。
