# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-03（Codex Desktop、ローカルMac）
- 作業環境: `/Users/triumph1118/Github/Multi-Class-LIME/multi-class-lime`
- ブランチ: `main`
- 基準コミット: `118d8f0`
- 注意: 引き継ぎに記載されていた`.venv/`は現在の作業ツリーに存在しない。検証にはpyenvのPython 3.10.14を使用した。

## 今回の目的

「全特徴量でサロゲートを学習し、比較方向を作った後に重要な特徴だけ提示すればよいのではないか」という案を検証する。学習時の特徴選択・再学習は行わず、dense係数をpost-hocに隠した説明が、簡潔性とpairwise fidelityを両立できるかを調べる。結果が弱い場合は、表示数不足、選択規則、手法間の方向同値性まで原因を切り分ける。

## 実施した変更と主要ファイル

1. **`src/run_dense_posthoc_experiment.py`（新規）**
   - 全手法を説明点中心の標準化座標$h(z)=(z-x)/\sigma$で全特徴学習。
   - OVR確率2本、OVO条件付き確率Ridge、多クラスsoft Fisher、pairwise soft Fisherを同一摂動で比較。
   - Fisher方向を1次元soft logisticで$q_{c,d}$へ校正。
   - dense方向の$|\beta_j|$で順位付けし、選択外係数を0にするだけで、選択特徴上の再学習はしない。$h(x)=0$なので説明点での予測は不変。
   - 学習・K選択用validation・最終testの局所摂動を分離。
   - K選択は、(a) dense性能を所定許容内で保つ最小K、(b) validation MSE最小K、(c)寄与エネルギー95%Kの3方針。
   - 合成グリッドと`sklearn` Digits実データの両方に対応。
2. **`tests/test_dense_posthoc_experiment.py`（新規）**
   - top-K、寄与エネルギー、説明点予測保持、soft logistic校正、Fisher方向、K選択をテスト。
   - 無正則化pairwise soft Fisher方向と重み付きOLS係数が、相関入力でも同一直線になることをテスト。
3. **`README.md`**
   - quick、合成本実験、Digits実験の実行コマンドを追加。
4. **`docs/PROJECT_SUMMARY.md`**
   - dense→post-hoc実験結果、Digits結果、Fisher/OLS方向同値性、設計上の注意を恒久情報として追加。

## 重要な結果と判断

### 1. 合成データ

- 条件: 3/4/5クラス、8/14/20特徴、2反復の18独立黒箱、各10説明点、計180説明点。学習300、validation 500、test 1000摂動、安定性8反復。
- dense全特徴の平均は14特徴。validation MSE最小Kでは、OVR 10.44、OVO 11.46、多クラスsoft Fisher 11.21、pairwise soft Fisher 11.46特徴を表示。
- 同方式のtest $R^2$は順に0.5146、0.5378、0.5407、0.5435。dense比の$R^2$変化は+0.0091、+0.0006、+0.0008、+0.0006で、平均では全手法が同等以上。
- 95%寄与方式は全手法約7.4特徴まで減るが、dense比$R^2$は0.008--0.018低下。簡潔性優先の実用候補だが、無損失ではない。
- 固定5特徴では$R^2=0.461$--$0.486$、denseでは0.506--0.543。常に5個だけ表示する方式は情報不足。
- OVRに対し、OVO/Fisher系はK選択後もpairwise MSE、$R^2$、勝敗一致率で優位。ただし表示数は平均0.3--1.0特徴多い。

### 2. Digits実データ

- 条件: 64特徴、10クラス、異なる分割・RF seedの2黒箱、各15説明点、計30説明点。学習500、validation 800、test 1500摂動、安定性8反復。
- validation MSE最小Kは、OVR 21.6、OVO 28.0、多クラスsoft Fisher 28.5、pairwise soft Fisher 27.5特徴。
- test $R^2$は順に0.5682、0.6135、0.6085、0.6156。dense比で0.0319、0.0064、0.0061、0.0061改善し、小係数を隠すことがpost-hoc正則化として働いた。
- 95%寄与Kは全手法約22特徴。OVO/Fisherはdenseとほぼ同等、OVRは$R^2$が0.0269改善。
- 固定5特徴の$R^2$は0.440--0.459、20特徴では0.560--0.607。Digitsで5特徴だけを見せるのは明確に不足。
- 実データの独立黒箱は2個だけなので、DigitsのCIは探索的であり一般化を主張しない。

### 3. うまく差が出なかった理由

- pairwise soft Fisherは$q=q_{c,d}$、$1-q$をsoft所属度として使う。このとき全分散の分解から、クラス内散布は全分散からrank-1のクラス間散布を引いた形になる。Sherman--Morrisonを適用すると、無正則化Fisher方向は$q$への重み付きOLS係数の定数倍となる。
- 単体テストで一般の相関入力について方向cosine=1を確認。実験でもpairwise Fisher対OVO Ridgeは合成・Digitsとも平均0.999999。
- 多クラスsoft Fisher対OVOも合成0.9967、Digits0.9932で非常に近い。pairwise Fisherの小さなMSE改善は、Fisher固有の方向ではなく、主にRidge/shrinkage差と1次元sigmoid校正による。
- したがって研究上は、pairwise soft FisherをOVO回帰とは異なる特徴方向を生む新手法として押し出さない。簡潔な実装ならOVO条件付き確率のdense Ridge＋post-hoc表示でほぼ同じ説明を得られる。

## 実行した検証

- `python3 -m unittest discover -s tests -v`: 14テスト成功。
- `python3 -m py_compile src/*.py`: 成功。
- `python3 src/run_dense_posthoc_experiment.py --quick`: 完走。
- `python3 src/run_dense_posthoc_experiment.py`: 複数回完走。最終コードで180説明点、欠損・校正失敗なし。
- `python3 src/run_dense_posthoc_experiment.py --r2-tolerance 0.01 --agreement-tolerance 0.005 --output-dir results/dense_posthoc_strict`: 完走。厳格化すると平均表示数は8.4--9.1へ増え、dense比平均$R^2$低下は概ね0--0.005へ縮小。
- `python3 src/run_dense_posthoc_experiment.py --digits --output-dir results/dense_posthoc_digits`: 最終コードで完走。30説明点、欠損・校正失敗なし。
- `git diff --check`: 成功。
- 結果CSVは`results/dense_posthoc_*.csv`、`results/dense_posthoc_strict/`、`results/dense_posthoc_digits/`へ出力。`results/`はGit管理外。

## 未完了・既知の問題

- 合成データとDigitsのみ。実データの独立黒箱は2個で、他の表形式データ・黒箱では未検証。
- 説明点は予測1位と2位が競合する点。確信度の高い点、ランダム点、ユーザー指定foilでは未検証。
- post-hoc表示は説明点での予測を保持するが、表示特徴だけでdenseモデル全体を再学習・再現するものではない。
- validationで選んだKは平均ではtestへ一般化したが、個別説明の性能保証ではない。安全側なら厳しい許容値または95%寄与方式を使う。
- 多クラスsoft FisherとOVO方向が近くなる条件の厳密な同値性は未証明。厳密に示したのはpairwise soft Fisherと無正則化OLS。
- `.DS_Store`は既存のユーザー変更として未変更・未削除。
- 今回の変更を含め、作業ツリーには以前からのOVO評価・runtime・関連研究・発表原稿の未コミット変更も残っている。commit/pushは実施していない。

## 次に行うこと

1. 修論の主方式候補を「OVO条件付き確率dense Ridge＋post-hoc top-K表示」とし、Fisherは理論比較・校正比較として位置づけ直す。
2. Kの規則を、簡潔性重視なら95%寄与、fidelity重視なら独立validation MSE最小として事前固定し、複数実データで再検証する。
3. Digits以外の中規模多クラス表形式データ、Gradient Boostingまたはニューラルネットで再現する。
4. 競合点以外の確信度層と、ユーザー指定foilで感度分析する。
5. push前に全差分、秘密情報、生成物除外を再確認する。
