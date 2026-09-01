# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-01（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/lime-multiclass-consistency-u9jgn9`（PR #1）
- 基準コミット: `e15a5d6`

## 今回の目的

前回コミット（`e15a5d6`）でconsistency・stability実験を完走させた後、3つ目の評価軸である**忠実性（fidelity）**を実験で検証する。理論的に予想されていた「Fisherは忠実性で劣るはず」という弱点を、実測でどこまで・どういう形で成り立つか確認する。

## 実施した変更と主要な変更ファイル

1. **`src/fidelity.py`**（新規）: 忠実性を測るための変換・損失関数。
   - `onevsrest_predict_proba`: Ridge回帰の生出力を非負にクリップして正規化（確率として解釈するための後処理）。
   - `fisher_predict_proba`: FisherのS_W・重心から、標準的なLDA確率モデル（マハラノビス距離＋事前確率のsoftmax、scikit-learnの`LinearDiscriminantAnalysis.predict_proba`と同じ考え方）で擬似確率を構成。ハード版・ソフト版どちらの`fit_fisher`結果にも使える。
   - `weighted_hellinger_loss`: SLISEMAP論文（Eq.11）と同じ二乗Hellinger距離を、LIMEの近接度重みで加重平均。
2. **`src/run_fidelity_experiment.py`**（新規）: 次元数×クラス数グリッドで one-vs-rest / Fisher(hard) / Fisher(soft) の忠実性を比較する実験ドライバ。`results/fidelity_results.csv`に出力。

## 重要な判断とその理由（★今回一番重要）

1. **第1回実験（Fisherを標準の多クラス確率分類器として評価）は非常に悪い結果だった**。全体平均でHellinger損失が one-vs-rest 0.016 に対し Fisher(hard) 0.065（約5.6倍）、Fisher(soft) 0.035（約2.2倍）。さらに「黒箱が一番あり得るとするクラスと、Fisherが一番あり得るとするクラスの一致率（argmax agreement）」も one-vs-rest 76.5% に対し Fisher(hard/soft) 55%前後と大きく劣った。9グリッドセル全てで一貫してこの順序（one-vs-rest < Fisher(soft) < Fisher(hard)、値が小さいほど良い）。
2. **【重要な訂正】上記は提案アルゴリズムの実際の使い方とズレたテストだった**。元の8ステップの設計（ステップ6-7）では、予測クラス$c^*$と競合クラス$c'$は**黒箱自身の`predict_proba`**から決め、Fisherはこの2クラス間の方向 $v(c^*,c')$ だけを担当する。Fisherに全クラスの中から1つを選ばせる（argmax）役割は元々与えられていない。
3. **本来の使い方（黒箱が決めた2クラスのペア比較の符号一致率）で測り直したところ、ほぼ互角だった**。9グリッドセル平均で one-vs-rest 81.3% vs Fisher 80.4%（差は約1ポイント、セルごとに見ても最大でも数ポイント差で、明確な優劣はない）。
4. **結論**：Fisherの弱点は「絶対確率値の推定（人に見せる数値としての信頼性）」に限定される。実際にこの手法が主張する「2クラスのどちらの証拠が強いか、その方向はどの特徴量か」という、LIMEの出力形式に相当する部分の忠実性は one-vs-rest とほぼ同等。修論では**「本手法が保証する範囲（ペア比較の方向）」と「保証しない範囲（絶対確率値としての解釈）」を明確に切り分けて書く**必要がある。

## 実行したテスト・確認結果

- `src/run_fidelity_experiment.py`をフルグリッド（n_features=[8,14,20]×n_classes=[3,4,5]）で完走。`results/fidelity_results.csv`に保存。
- argmax一致率の検証、および「黒箱が選んだペアでの符号一致率」の検証はアドホックなスクリプトで実施（`results/`には未保存、本文書に数値を記録）。

## 未完了・既知の問題・未検証事項

- 「黒箱が選んだペアでの符号一致率」の検証は`src/`に恒久的なスクリプトとして残していない（アドホック実行のみ）。再現性のため、`src/run_fidelity_experiment.py`に正式な指標として組み込むことを検討する。
- ソフトラベル版がなぜ`n_classes=7`でのみfeature overlapで有効なのかは未検証のまま（前回からの持ち越し）。
- 実データセットでの再現性確認は未着手。

## 次に行うこと

1. 「黒箱が選んだペアでの符号一致率」を正式な指標として`src/fidelity.py`/`src/run_fidelity_experiment.py`に組み込み、再現可能にする。
2. ソフト版が`n_classes=7`でのみfeature overlapに有効な理由を、重心間距離・S_Bの直接比較で検証する。
3. 実データセットでの再現性確認。
4. ここまでの4指標（stability, feature overlap, sum-to-one, fidelity）の結果を統合し、修論の主張として文章化する。
