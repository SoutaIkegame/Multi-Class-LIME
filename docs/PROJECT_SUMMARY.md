# Project Summary

<!--
この文書はプロジェクト全体の現在像を示す長期的なまとめです。
一時的な作業履歴ではなく、別端末・別セッションでも必要になる恒久情報を記載してください。
-->

## プロジェクトの目的

多クラス分類問題におけるLIMEの拡張を検討する修論研究。従来のLIMEの多クラス対応（one-vs-rest：クラスごとに独立した線形サロゲートをフィットする）が抱える問題点を整理し、代替手法としてFisher判別分析（LDA）を局所サロゲートとして使う方式を提案・検証する。

## 現在の状態

- 理論設計は一通り完了。摂動生成・黒箱への問い合わせ・近接度重み付けはLIMEと共通のまま、「サロゲートをフィットする」ステップだけをFisher LDAに置き換える設計。
- 実験コード（`src/`）を実装中。one-vs-rest LIME vs Fisher LIMEのconsistency・stability比較実験。
- **重要な理論修正あり**：当初の「推移律が崩れる」という問題提起は数学的に成立しないことが判明（任意の実数の大小比較は常に推移的なため）。詳細と正しい主張の立て方は`docs/RECENT_WORK.md`の該当セクション参照。stability（分散）は実測で有意な差を確認済みで、これが現時点で最も説得力のある差別化ポイント。

## 主要な構成

- `src/perturbation.py`: one-vs-rest LIMEとFisher LIMEに共通の摂動サンプリング（LIMEのデフォルト方式：Gaussianノイズ×特徴量標準偏差、指数カーネルで近接度重み付け）。両手法を公平に比較するため、この摂動生成ステップだけは完全に共有する設計。
- `src/surrogates.py`: 2つのサロゲートフィッティング関数。
  - `fit_onevsrest`: クラスごとに独立な重み付きRidge回帰（標準的なLIMEの実装方式）。
  - `fit_fisher`: 3クラス（以上）共通のプールされたクラス内散布行列S_Wを使うFisher LDAサロゲート。shrinkage正則化あり。ペア方向 `v(X,Y) = S_W^{-1}(μ_X-μ_Y)` を計算するヘルパーを返す。
- `src/metrics.py`: consistency・stability指標。
- `src/run_experiment.py`: 次元数×クラス数のグリッドで両手法を比較する実験ドライバ。まだ完走・結果確定はしていない（`docs/RECENT_WORK.md`参照）。
- `.venv/`: Python仮想環境（`.gitignore`で除外、コミット対象外）。

## セットアップと実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/run_experiment.py   # 結果は results/ にCSV出力
```

必要パッケージ: numpy, scipy, scikit-learn, pandas, lime, matplotlib（`requirements.txt`参照）。

## 重要な設計判断

- **one-vs-rest LIMEとFisher LIMEは、同一の摂動サンプル・同一の近接度重みを使う**（`src/perturbation.py`で共有）。これは元の問題提起の前提条件（「摂動データは同じサンプリングとする」）を実験でも忠実に再現するため。
- one-vs-rest側の黒箱への問い合わせターゲットは各クラスの確率 `predict_proba` の各列。Fisher側はハードラベル（`argmax`）でクラスを割り当ててから重心・散布行列を計算する（拡張案としてソフトラベル版も議論済みだが未実装）。
- Fisher側のS_Wは正則化のため `S_W + ε·(trace(S_W)/d)·I` のshrinkageを適用している（`shrinkage`パラメータ、デフォルト1e-3）。

## データと環境依存

- 合成データ（`sklearn.datasets.make_classification`）を使用。実データセットは今のところ使用していない。
- 黒箱モデルは`RandomForestClassifier`（非線形決定境界を作るため）。
- 秘密情報・環境依存の外部サービスなし。

## 既知の制約・リスク

- `transitivity_violation_rate`（推移律違反率）という指標は理論的に常に0になる無意味な指標であることが判明。使用する場合は必ず`docs/RECENT_WORK.md`の説明を踏まえること。
- sum-to-one（$\hat P(A)+\hat P(B)+\hat P(C)=1$）は、one-vs-rest側が全特徴量を使った同一設計の回帰であれば厳密に成立してしまう。崩れを見たい場合は`lime`公式パッケージの`feature_selection='auto'`＋`num_features < 全特徴量数`の設定（クラスごとに異なる特徴量部分集合が選ばれる状況）で検証する必要がある。
- Fisher LDAのS_Wは次元が高くローカルサンプル数が少ない場合に特異・悪条件になりうる（shrinkage正則化で緩和しているが、パラメータの妥当性は未検証）。

## 今後の大きな課題

- consistencyの主張を、実測で裏付けられる正確な形（「独立フィットそのものが原因」ではなく「特徴量選択の不一致が原因」）に修論の記述を修正する。
- stability比較を本実験として完走させ、次元数・クラス数依存性を確認する。
- ソフトラベル版Fisher LIMEの実装（拡張案、未着手）。
- 忠実性（fidelity）の評価（TODO、未着手）。
