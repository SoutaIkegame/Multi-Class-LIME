# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-01（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/lime-multiclass-consistency-u9jgn9`（PR #1）
- 基準コミット: `e253ad5`

## 今回の目的

ユーザー提案「参照クラス型の対数オッズ比を、行単位のgroup lasso（MultiTaskLasso）で複数クラスペア分同時推定する」手法（5案の中の3・4・5番）を実装し、これまでの3手法（one-vs-rest, Fisher(hard), Contrastive）と同じ指標で比較する。

## 実施した変更と主要な変更ファイル

1. **`src/grouplasso.py`**（新規）：
   - `fit_grouplasso` / `fit_grouplasso_sparse`：黒箱の予測クラス$r$を参照クラスとし、$\log((p_r+\varepsilon)/(p_k+\varepsilon))$（$k\neq r$）を全クラスペア分まとめて`sklearn.linear_model.MultiTaskLasso`（行ノルム$\|B_{j,:}\|_2$への罰則＝group lasso）で同時フィット。`fit_grouplasso_sparse`はK個以上の非ゼロ行を持つ最疎解を二分探索で求める版（feature overlap実験用）。
   - `recover_proba`：多項ロジット逆変換で、有界かつ合計1の確率ベクトルを復元するヘルパー。
2. **`src/run_grouplasso_experiment.py`**（新規）：one-vs-rest / Fisher(hard) / Group Lassoを、(a) Hellinger忠実性（復元確率 vs 黒箱の実際の確率）、(b) 符号一致率（黒箱が選んだペアでの方向一致）、(c) stability（正規化分散）、(d) feature overlap（Group Lassoは構造的に1.0が保証されるはずで、これを直接検証）で比較するグリッド実験。

## 重要な判断とその理由（★今回一番重要）

1. **feature overlapは構造的な保証通り、9セル全て厳密に1.000だった**（one-vs-rest 0.458, Fisher 0.518 に対して）。group lassoの行単位罰則により、全クラスペアの説明が同じ特徴量部分集合を共有することが、推定ではなく直接確認できた。LIMEtreeが名指しした失敗モード（「異なる特徴量部分集合を使う」）を、線形サロゲートのまま構造的に回避できている。
2. **符号一致率・stabilityはone-vs-restと同格かそれ以上**（符号一致率：gl 0.800 vs ovr 0.799、stability：gl 0.046 vs ovr 0.047、9セル中6セルでgl優位）。Fisherよりは明確に安定。
3. **【重要な弱点】Hellinger忠実性はone-vs-restの24倍、Fisherの7倍悪化した**（gl 0.402 vs ovr 0.017 vs fisher 0.059、9セル全てで一貫）。原因を調査：alphaを4桁変えても変化せず、線形フィット自体のR²も生の確率回帰と遜色ない（0.37〜0.68）ことを確認した。つまり「線形近似が下手」ではなく、**$\hat p_r=1/(1+\sum_k\exp(\hat\ell_k))$という復元式が、残った線形フィット誤差を指数的に増幅する**ことが原因。
4. **この結果はFisherと同じパターンを繰り返している**：「何らかの構造（Fisherはクラス分離、Group Lassoは行スパース性）を最適化する設計は、ペア比較の方向・符号は保つが、絶対確率値としての忠実性は犠牲になる」という一般化した傾向が、2つ目の独立した手法で確認できた。「pure probability regression以外の設計は、程度の差はあれこの代償を払う」という主張の材料になる。

## 実行したテスト・確認結果

- alphaを{0.1, 0.01, 0.001, 0.0001, 0.00001}で振ってHellinger損失への感度を確認（0.229〜0.242でほぼ不変、正則化強度が原因ではないことを確認）。
- epsを{1e-6, 1e-3, 1e-2}で振って確認（0.216→0.207→0.189、多少改善するが根本解決しない）。
- 生の確率回帰と対数比回帰それぞれのR²を直接比較し、線形フィット自体は同程度の精度であることを確認。
- `src/run_grouplasso_experiment.py`をフルグリッド（n_features=[8,14,20]×n_classes=[3,4,5]×K）で完走。`results/grouplasso_results.csv`に保存。

## 未完了・既知の問題・未検証事項

- Hellinger忠実性の悪化を緩和する方法（例：exp前にクリッピングする、ソフトラベル的な重み付けをする等）は未検証。
- Contrastiveとの直接比較（Contrastiveもfeature overlapで「ペア間」の重なりを見ていたが、Group Lassoは構造的に1.0保証という点で質的に異なる）はまだ整理していない。
- 実データセットでの再現性確認は未着手。

## 次に行うこと

1. Group LassoのHellinger忠実性を改善する方法を検討する（exp前のクリッピング、ソフトラベル重み付けなど）。
2. feature overlapの「クラス間 vs ペア間 vs 構造的保証」という3種類の比較単位を整理し、Contrastive・Fisher・one-vs-rest・Group Lassoを公平に位置づけ直す。
3. ソフト版Fisherの正規化安定性をフルグリッドで検証する（前回からの持ち越し）。
4. 4手法（one-vs-rest, Fisher(hard/soft), Contrastive, Group Lasso）の結果を統合し、修論の主張として文章化する。「構造の共有と絶対確率忠実性はトレードオフの関係にある」という一般化した主張が中心になりそう。
5. 実データセットでの再現性確認。
