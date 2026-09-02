# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-01（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/lime-multiclass-consistency-u9jgn9`（PR #1）
- 基準コミット: `afd0584`

## 今回の目的

ユーザー提案の「Contrastive LIME」（one-vs-restを2本フィットしてから引くのではなく、$\log\frac{p_A(z)+\varepsilon}{p_B(z)+\varepsilon}$を目的変数にして直接回帰する第3の手法）を実装し、one-vs-rest・Fisher(hard)と同じ3指標（fidelity, stability, feature overlap）で比較する。

## 実施した変更と主要な変更ファイル

1. **`src/surrogates.py`**: `fit_contrastive`（$\log((p_{c1}+\varepsilon)/(p_{c2}+\varepsilon))$を目的変数にした重み付きRidge回帰）、`fit_contrastive_lasso`（同じ目的変数でのLasso選択版、feature overlap実験用）を追加。
2. **`src/run_contrastive_experiment.py`**（新規）：one-vs-rest / Fisher(hard) / Contrastiveの3手法を、fidelity（黒箱が選んだペアでの符号一致率）・stability（正規化分散）・feature overlap（Lasso top-K重なり）で同時比較するグリッド実験。

## 重要な判断とその理由

1. **fidelityは理論的な予想（Contrastiveが明確に勝つ）が外れ、3手法ほぼ同点だった**（全体平均 one-vs-rest 0.799, Fisher 0.791, Contrastive 0.800）。理由：one-vs-restの「$p_A$の回帰」から「$p_B$の回帰」を引く操作は、同一設計（同じ特徴量・重み・Ridge）である限り、線形性により**$p_A-p_B$を直接回帰したものと数学的に完全に一致する**（以前発見したsum-to-oneの理屈と同型）。Contrastiveが変えたのは「差分ではなくlog比」という点だけで、符号一致率という指標にはこの違いがほとんど効かなかった。
2. **stabilityは予想通り、one-vs-restとほぼ同格**（正規化分散 one-vs-rest 0.047 vs Contrastive 0.046）で、Fisher(hard)の0.117よりずっと安定。Contrastiveは1回のRidge回帰という点でone-vs-restと同じ仕組みのため、妥当な結果。
3. **feature overlapは予想と半分違った**。「ペアごとに独立フィットするので共有構造がなく、Fisherより悪化するはず」と予想していたが、実際は one-vs-rest(0.458) < Contrastive(0.485) < Fisher(0.518) で、Fisherより低いがone-vs-restより高かった。**ただし比較の単位が異なる点に注意**：one-vs-rest/Fisherは「クラス間」（n_classes個の説明）の重なり、Contrastiveは「ペア間」（C(n_classes,2)個の説明）の重なりを見ており、直接の優劣比較ではない。合成データの構造上、全ペアの境界を分ける「核となる特徴量」が共通している可能性があり、明示的な共有構造がなくてもある程度自然に重なりが生まれたと考えられる。

## 実行したテスト・確認結果

- `src/run_contrastive_experiment.py`をフルグリッド（n_features=[8,14,20]×n_classes=[3,4,5]×K∈{0.3,0.6}×n_features）で完走。`results/contrastive_results.csv`に保存。

## 未完了・既知の問題・未検証事項

- fidelityの差が出なかったのは「符号一致率」という指標が粗すぎる可能性がある。log比と生の差分の違いは、確率が0/1に近い極端な領域でより顕著に出るかもしれない（未検証）。
- feature overlapの「クラス間 vs ペア間」という比較単位の違いを解消した、より公平な指標が必要（例：one-vs-rest/Fisherも「予測クラス vs 各対抗クラス」のペア単位で揃えて再比較する）。
- Contrastiveのfeature overlapが理論的懸念（共有構造なし）ほど悪化しなかった理由は、合成データの構造への依存を疑っており、実データまたはより特徴量間の相関構造を変えたデータでの再検証が必要。

## 次に行うこと

1. feature overlapの比較単位をクラス間・ペア間で統一し、Contrastive vs Fisher vs one-vs-restを公平に再比較する。
2. 確率が極端な領域（0または1に近い）でのContrastiveの優位性を、専用の検証で確認する。
3. ソフト版Fisherの正規化安定性をフルグリッドで検証する（前回からの持ち越し）。
4. 4手法（one-vs-rest, Fisher(hard/soft), Contrastive）×3指標の結果を統合し、修論の主張として文章化する。
5. 実データセットでの再現性確認。
