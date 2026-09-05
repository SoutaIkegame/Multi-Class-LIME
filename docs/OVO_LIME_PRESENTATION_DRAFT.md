# OVO-LIME 発表用構成案

更新日: 2026-09-03  
用途: 「XAI → LIME → 多クラスLIME（OVR）」まで説明した後に続けるスライド原稿  
想定枚数: 本編15枚＋補足3枚

---

## この構成で伝える結論

本発表で最終的に伝えたいことは、次の一文である。

> 従来のクラス別LIMEは対象クラスと残り全クラスの関係を説明するが、利用者が知りたい「なぜクラス$c$であり、特定のクラス$d$ではないのか」という問いには直接対応しない。そこで、元の多クラスBBを変更せず、指定した2クラスの条件付き確率またはlog-ratioを直接近似するOVO-LIMEを検討した。予備実験では、OVO条件付き確率回帰は、より少ない表示特徴数でOVRと同等以上のpairwise fidelityを得られる可能性を示した一方、強い疎性ではstabilityが低下した。

「OVOは常に優れる」ではなく、**問いへの適合・簡潔性・忠実度と安定性のトレードオフを明らかにした**、という着地にする。

---

# 本編

## Slide 1. 問題提起：知りたいのは「なぜ$c$であって$d$ではないのか」

### スライドに載せる内容

従来の多クラスLIME：

$$
p_c(z)
\quad\Rightarrow\quad
C_c\ \text{vs.}\ \mathcal C\setminus\{C_c\}
$$

利用者が知りたい問い：

$$
\text{Why }C_c\text{ rather than }C_d\text{?}
$$

例：犬・猫・自動車の3クラス分類

- OVR：「なぜ犬か」＝犬 vs 猫＋自動車
- OVO：「なぜ犬であって猫ではないか」＝犬 vs 猫
- 自動車との違いは、犬と猫を区別する理由には必ずしも必要ない

### このスライドの一言

> OVRの説明対象と、利用者が尋ねる対比的な問いが一致していない。

### 図の案

左側に「犬」、右側に「猫＋自動車」をまとめたOVR図を置く。その下に「犬 ↔ 猫」だけをつないだOVO図を置く。

### 発表時に話す内容

通常のLIMEでは、あるクラス$c$の確率を1本のサロゲートで説明する。したがって意味的には、クラス$c$とそれ以外を比較するone-vs-rest説明になる。しかし、実際の利用者が知りたいのは、単に「なぜ犬か」ではなく、「なぜ猫ではなく犬なのか」のような、比較対象を指定した問いである場合が多い。

### 出典

- Sokol and Flachは、LIMEがクラスごとに別のサロゲートを作り、単一クラスの確率しか説明できないため、暗黙的なone-vs-rest explainerになると整理している。[LIMEtree, 2025](https://doi.org/10.3390/electronics14050929)

---

## Slide 2. なぜOne-vs-Oneなのか：説明は対比的である

### スライドに載せる内容

人は一般に、

$$
\text{Why }P\text{ rather than }Q\text{?}
$$

という形で説明を求める。

- 比較対象（foil）を明示すると、説明すべき差が限定される
- 類似したクラスほど、共通特徴ではなく識別特徴が重要になる
- 予測1位 vs 2位、正解 vs 誤分類、危険な誤分類先などに利用できる

### このスライドの一言

> 説明対象を1クラスに固定するのではなく、「何と比べるか」を指定したい。

### 発表時に話す内容

Millerは、社会科学の説明研究を整理し、人間の説明が対比的であることを指摘している。またGANMEXは、多クラス画像分類において、似たクラス同士ではone-vs-one説明が重要だとしている。例えばリンゴとオレンジを区別するなら、両方に共通する丸い形より、色の違いが重要になる。

### 出典

- Miller, *Explanation in Artificial Intelligence: Insights from the Social Sciences*, Artificial Intelligence, 2019. [DOI](https://doi.org/10.1016/j.artint.2018.07.007)
- Shih, Tien, and Karnin, *GANMEX: One-vs-One Attributions using GAN-based Model Explainability*, ICML 2021. [PMLR](https://proceedings.mlr.press/v139/shih21a.html)

---

## Slide 3. 関連研究と本研究の位置づけ

### スライドに載せる表

| 手法 | 比較対象 | 説明方法 |
|---|---|---|
| LIME | $C_c$ vs rest（暗黙的） | $p_c$の局所線形回帰 |
| CLIMAX | $C_c$ vs rest | $\log\frac{p_c}{1-p_c}$の局所説明 |
| Local Foil Trees | 指定したfact vs foilを出力 | foil vs restの局所決定木からルール差を抽出 |
| GANMEX | 指定した$C_c$ vs $C_d$ | GANでfoilクラスの基準画像を生成し特徴帰属 |
| 提案するOVO-LIME | 指定した$C_c$ vs $C_d$ | pairwise出力の局所疎線形回帰 |

### 研究ギャップ

> OVO・foil指定・局所サロゲートの各アイデアには先行研究がある。一方、元の多クラスBBのsoft pairwise確率またはlog-ratioを教師信号として、LIME型の局所疎線形サロゲートで直接近似し、その忠実度・簡潔性・安定性をOVRと統一条件で比較する方法は、調査した範囲では確認できなかった。

### 発表時に話す内容

LIMEとCLIMAXは、対象クラスと残り全部を比較する。Local Foil TreesはLIMEに近い局所サロゲートでfactとfoilの差を出力するが、学習する木自体はfoil対restであり、softな$p_c$対$p_d$を直接近似しない。GANMEXは明確なOVO説明だが、GANで反実仮想的な基準画像を作る画像向け特徴帰属手法であり、LIME型線形サロゲートではない。本研究では、既存の多クラスBBを再学習せず、説明時に任意のクラス対を選び、その相対出力を局所線形モデルで説明する。

### 注意

- 「OVO説明を初めて提案した」とは言わない
- 「OVO-LIMEが世界初」と断定せず、「soft pairwise出力の直接回帰と統一比較の組合せは、調査範囲で完全一致を確認できなかった」とする
- CLIMAXについては、完全なアルゴリズム全体ではなく、基本目的変数がtarget vs restである点を比較する

### 出典

- Ribeiro, Singh, and Guestrin, *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier*, KDD 2016. [DOI](https://doi.org/10.1145/2939672.2939778)
- Nanavati and Prasad, *CLIMAX: An Exploration of Classifier-Based Contrastive Explanations*, IEEE CogMI 2023. [DOI](https://doi.org/10.1109/CogMI58952.2023.00017)
- van der Waa et al., *Contrastive Explanations with Local Foil Trees*, ICML WHI Workshop 2018. [Utrecht University](https://research-portal.uu.nl/en/publications/contrastive-explanations-with-local-foil-trees-2/)
- Shih et al., GANMEX, ICML 2021. [PMLR](https://proceedings.mlr.press/v139/shih21a.html)

---

## Slide 4. OVOで学習する量：二者間条件付き確率

### スライドに載せる内容

多クラスBBの出力：

$$
f(z)=\left(p_1(z),\ldots,p_n(z)\right)
$$

クラス$c,d$の二者間条件付き確率：

$$
q_{c,d}(z)
=
\frac{p_c(z)}{p_c(z)+p_d(z)}
$$

性質：

$$
q_{d,c}(z)=1-q_{c,d}(z)
$$

$$
q_{c,d}(z)>0.5
\iff
p_c(z)>p_d(z)
$$

### 重要な説明

- BBを2クラスで再学習するわけではない
- 摂動サンプルも削除しない
- 元の多クラスBBが返した$p_c,p_d$だけを説明時に再正規化する

### 数値例

$$
(p_c,p_d,p_e)=(0.45,0.30,0.25)
$$

$$
q_{c,d}=\frac{0.45}{0.45+0.30}=0.60
$$

「$c,d$の二択として見れば、$c$側が60%」と解釈する。

### 出典

このpairwise条件付き確率自体は新しいものではなく、多クラス確率推定のpairwise couplingで利用されている。

- Wu, Lin, and Weng, *Probability Estimates for Multi-class Classification by Pairwise Coupling*, JMLR 5, 2004. [JMLR](https://www.jmlr.org/papers/v5/wu04a.html)

---

## Slide 5. もう一つの候補：pairwise log-ratio

### スライドに載せる内容

$$
\ell_{c,d}(z)
=
\log\frac{p_c(z)}{p_d(z)}
$$

二者間確率との関係：

$$
\ell_{c,d}
=
\log\frac{q_{c,d}}{1-q_{c,d}}
$$

$$
q_{c,d}=\sigma(\ell_{c,d})
$$

線形サロゲート：

$$
\hat\ell_{c,d}(z)
=
a_{c,d}+\boldsymbol\beta_{c,d}^{\top}z
$$

### 係数の解釈

$$
\beta_{c,d,j}>0
$$

なら、特徴$j$が増えるほど、局所的に$d$より$c$が相対的に有利になる。

$\beta_{c,d,j}=0.2$なら、特徴$j$が1単位増えたとき、$c$対$d$のオッズが、他の条件を固定した局所線形近似上で、

$$
e^{0.2}\approx1.22
$$

倍になる。

### 発表時に強調する内容

- 因果効果ではなく、BBの局所的な入出力関係
- 「$p_c$自体が増える」とは限らない
- 「$d$に比べて$c$が有利になる」という相対的解釈

---

## Slide 6. 提案手法：OVO-LIME

### スライドに載せる目的関数

説明点$x$の周辺に摂動点$Z=\{z_m\}_{m=1}^{M}$を生成し、局所重み$\pi_x(z_m)$を与える。

log-ratio版：

$$
\hat g_{c,d}
=
\underset{g\in G}{\arg\min}
\left\{
\sum_{m=1}^{M}
\pi_x(z_m)
\left[
\log\frac{p_c(z_m)+\varepsilon}{p_d(z_m)+\varepsilon}
-g(z_m)
\right]^2
+\Omega(g)
\right\}
$$

条件付き確率版では、目的変数を

$$
q_{c,d}(z_m)=\frac{p_c(z_m)+\varepsilon}{p_c(z_m)+p_d(z_m)+2\varepsilon}
$$

へ置き換える。

### 処理フロー

```text
説明点 x
   ↓ 周辺を摂動
摂動点 z_1,...,z_M
   ↓ 元の多クラスBBへ入力
(p_1(z_m),...,p_n(z_m))
   ↓ 比較する c,d を選択
q_cd(z_m) または log[p_c(z_m)/p_d(z_m)]
   ↓ 局所重み付き疎回帰
クラス対(c,d)専用の特徴係数 β_cd
```

### 重要な比較

単に$p_c-p_d$を同じRidgeで直接回帰するだけでは、同一設計のOVR 2本の差と数学的に一致する。したがって、本手法の差は、**pairwise条件付き確率またはlog-ratioを説明対象にすること**にある。

---

## Slide 7. $\binom{n}{2}$本を全部作り、全部見せるのか

### 最初に示す答え

> 通常は作らない。比較したいfoilだけを選び、1本をon-demandで学習・表示する。

### 表示するクラス対の選択

予測クラスを$c^*$とすると、優先順位は次のように決められる。

1. **既定値**：予測1位$c^*$ vs 2位クラス
2. **ユーザー指定**：$c^*$ vs ユーザーが疑問に思ったクラス
3. **誤分類分析**：誤予測クラス vs 正解クラス
4. **業務上の指定**：混同すると危険なクラス対
5. **上位$k$クラス**：必要な場合だけ$c^*$と上位候補を比較

したがって、通常の説明で必要なのは、

$$
1\text{本}
\quad\text{または多くても}\quad
n-1\text{本}
$$

であり、全$\binom{n}{2}$本ではない。

### 計算上のポイント

- BBへの問い合わせは、$M$個の摂動について**1回**行えば全クラス確率が得られる
- ペアごとにBBを再実行する必要はない
- 追加コストは、取得済み確率から教師信号を作り、小さな線形サロゲートをフィットする部分
- 同じ$Z$、局所重み、Ridgeを使う全特徴モデルなら、全ペアをmulti-output回帰として一括計算できる
- ペアごとにLasso特徴選択を行う場合は、計算量がほぼ$O(n^2)$で増える

クラス数と全ペア数：

| クラス数$n$ | $\binom{n}{2}$ | 予測クラスとのみ比較$n-1$ |
|---:|---:|---:|
| 5 | 10 | 4 |
| 10 | 45 | 9 |
| 20 | 190 | 19 |
| 50 | 1,225 | 49 |
| 100 | 4,950 | 99 |

### 全ペアが必要な場合の削減案

基準クラス$r$に対する$n-1$本のlog-ratio、

$$
s_c(z)=\log\frac{p_c(z)}{p_r(z)}
$$

を学習すれば、

$$
\ell_{c,d}(z)=s_c(z)-s_d(z)
$$

として任意ペアを復元できる。さらに、

$$
q_{c,d}(z)=\sigma\left(s_c(z)-s_d(z)\right)
$$

へ変換できる。ただし、ペアごとに異なる疎な特徴を直接選ぶ方式とは説明結果が一致しない可能性がある。

### 計算時間のスケーリング実験

現在の実装（全特徴Ridgeで順位付けし、上位$K$特徴でRidgeを再フィット）について、クラス数$n$、摂動数$M$、特徴次元$D$をそれぞれ変化させた。各条件を5回測定した中央値で、BLASは1スレッドに固定した。

まず、$M=1000,D=50,K=10$に固定してクラス数を増やした結果を示す。

| $n$ | ペア数 | OVR全$n$本 | OVO全ペア | 指定ペア用OVR 2本 | OVO選択1本 | 全OVO / OVR | OVR 2本 / OVO 1本 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 3.8 ms | 5.9 ms | 2.92 ms | 1.64 ms | 1.6倍 | 1.8倍 |
| 5 | 10 | 6.2 ms | 12.4 ms | 2.41 ms | 1.17 ms | 2.0倍 | 2.1倍 |
| 10 | 45 | 13.1 ms | 79.5 ms | 3.31 ms | 1.52 ms | 6.1倍 | 2.2倍 |
| 20 | 190 | 27.8 ms | 278.6 ms | 2.85 ms | 1.38 ms | 10.0倍 | 2.1倍 |
| 50 | 1,225 | 67.1 ms | 1,756.9 ms | 3.04 ms | 1.42 ms | 26.2倍 | 2.1倍 |
| 100 | 4,950 | 148.2 ms | 7,126.0 ms | 2.82 ms | 1.55 ms | 48.1倍 | 1.8倍 |

全ペアOVOは$\binom{n}{2}$本を個別に疎化するため、$n$に対してほぼ二次的に増加する。$n=100$では約7.1秒で、OVR全クラスの約48倍だった。一方、foilを先に選んでOVOを1本だけ学習する時間は、$M,D$が一定なら約1.2--1.6 msであり、クラス数にはほぼ依存しない。同じ指定ペアを説明するためにOVRを2本作る条件と比べても、OVO 1本はおよそ半分の時間だった。

入力規模に対する推移も確認した。摂動数の表は$n=20,D=50,K=10$、特徴次元の表は$n=10,M=1000,K=10$に固定した結果である。

| 摂動数$M$ | OVR全クラス | OVO全ペア | 指定ペア用OVR 2本 | OVO選択1本 |
|---:|---:|---:|---:|---:|
| 100 | 16.6 ms | 150.6 ms | 1.57 ms | 0.80 ms |
| 300 | 22.8 ms | 209.9 ms | 2.65 ms | 1.06 ms |
| 1,000 | 25.9 ms | 261.7 ms | 2.56 ms | 1.18 ms |
| 3,000 | 55.6 ms | 442.1 ms | 5.69 ms | 2.65 ms |
| 10,000 | 121.0 ms | 1,240.8 ms | 10.89 ms | 6.99 ms |

| 特徴次元$D$ | OVR全クラス | OVO全ペア | 指定ペア用OVR 2本 | OVO選択1本 |
|---:|---:|---:|---:|---:|
| 10 | 5.5 ms | 28.5 ms | 1.05 ms | 0.57 ms |
| 20 | 12.4 ms | 57.6 ms | 2.34 ms | 1.27 ms |
| 50 | 11.5 ms | 51.6 ms | 2.29 ms | 1.15 ms |
| 100 | 17.8 ms | 94.3 ms | 4.09 ms | 1.66 ms |
| 200 | 37.4 ms | 169.0 ms | 7.17 ms | 3.24 ms |

特徴次元の小さい領域では実行時ノイズや線形代数ライブラリの処理効率により完全な単調増加ではないが、$D=10\to200$の全体では各方式の時間が増加した。それでも最大条件で、選択済み1ペアは摂動数実験で約7.0 ms、特徴次元実験で約3.2 msである。したがって、**全ペア生成を標準動作にすると計算面で負けるが、foilを先に決めるon-demand方式なら追加コストは小さい**、という設計上の結論になる。

#### 計測条件に関する注意

- BBへの問い合わせ時間は含めない。$M$個の摂動に対する全クラス確率はOVR・OVOで共通して一度取得できるため、ここでは方式間で異なるサロゲート学習部分だけを比較した。
- 合成したsoftmax確率をBB出力の代わりに用いた計算量実験であり、fidelity実験とは別である。
- 時間の絶対値は機器・実装・線形ソルバに依存するため、主張は絶対時間ではなく増加傾向に置く。
- OVO選択1本は短時間で測定誤差が大きいため、各反復内で20回実行した平均時間を1測定値とした。
- スライドでは、クラス数を横軸、時間を対数軸とした3系列の折れ線グラフにすると、全ペアOVOの増加とon-demand OVOの横ばいを同時に示しやすい。

また、「全ペアを作り、fidelityが最も高い説明を後から選ぶ」方式は採用しない。これは計算量が増えるだけでなく、説明しやすいペアを事後的に選ぶことになり、利用者が本来尋ねた比較とずれる可能性がある。foilはBBの2位確率、ユーザー指定、正解ラベル、業務上の危険クラスなど、**説明生成前の基準**で選ぶ。

### このスライドの一言

> 「生成可能な説明数」と「ユーザーに提示する説明数」は分けて設計する。

### 出典

- pairwise couplingでは、全クラス対の比較を結合して多クラス確率を扱う。[Wu et al., JMLR 2004](https://www.jmlr.org/papers/v5/wu04a.html)
- 一般的なOVO分類でも$\binom{n}{2}$個の二値問題を作ることが知られている。[Galar et al., Pattern Recognition 2011](https://doi.org/10.1016/j.patcog.2011.01.017)

---

## Slide 8. 何をもって良い説明とするか

### スライドに載せる内容

BBの分類精度ではなく、サロゲート説明の品質を評価する。

| 評価軸 | 本研究での意味 | 指標 |
|---|---|---|
| Fidelity | BBの$c$対$d$判断を再現できるか | weighted MSE、$R^2$、勝敗一致率 |
| Stability | 摂動を変えても同じ説明になるか | 係数方向cosine、top-$K$ Jaccard |
| Compactness | 少数特徴で説明できるか | 実効特徴数 |
| Contrastivity | 指定したfoilに答えているか | 目的変数を$q_{c,d}$または$\ell_{c,d}$に固定 |

### 出典

Nauta et al.は、XAI評価を単一指標ではなく、correctness、consistency、contrastivity、compactnessなど複数の性質として整理している。またS-LIMEは、摂動サンプリングによるLIMEの説明不安定性を直接扱っている。

- Nauta et al., *From Anecdotal Evidence to Quantitative Evaluation Methods*, ACM Computing Surveys, 2023. [DOI](https://doi.org/10.1145/3583558)
- Zhou, Hooker, and Wang, *S-LIME: Stabilized-LIME for Model Explanation*, KDD 2021. [DOI](https://doi.org/10.1145/3447548.3467274)

---

## Slide 9. 評価指標の定義

### 1. Pairwise probability fidelity

学習に使っていない評価用摂動$Z_{\mathrm{test}}$で測定する。

$$
\mathrm{WMSE}
=
\frac{
\sum_{z\in Z_{\mathrm{test}}}
\pi_x(z)
\left(q_{c,d}(z)-\hat q_{c,d}(z)\right)^2
}{
\sum_{z\in Z_{\mathrm{test}}}\pi_x(z)
}
$$

### 2. Pairwise decision agreement

$$
\mathrm{Agreement}
=
\frac{
\sum_z\pi_x(z)
\mathbf 1
\left[
(q_{c,d}(z)\ge0.5)
=
(\hat q_{c,d}(z)\ge0.5)
\right]
}{
\sum_z\pi_x(z)
}
$$

### 3. Stability

同じ説明点について摂動生成を10回繰り返す。係数を特徴量の標準偏差でスケールし、単位ベクトル化してから平均cosine similarityを計算する。

$$
\tilde\beta^{(r)}
=
\frac{\beta^{(r)}\odot\sigma_X}
{\|\beta^{(r)}\odot\sigma_X\|_2}
$$

さらに、各試行で選択された特徴集合のJaccard係数も測る。

### 補足

- WMSEは小さいほど良い
- $R^2$、Agreement、cosine、Jaccardは大きいほど良い
- in-sampleではなくheld-out摂動でfidelityを測る

---

## Slide 10. 実験設定

### スライドに載せる表

| 項目 | 設定 |
|---|---|
| データ | `make_classification`による合成表形式データ |
| データ数 | 2,000件／データセット |
| クラス数 | 3、4、5 |
| 特徴数 | 8、14、20 |
| 有効特徴数 | $\max(3,\text{クラス数})$ |
| 冗長特徴数 | 残りすべて。相関特徴を意図的に含める |
| クラス分離 | `class_sep=1.2` |
| BB | RandomForestClassifier |
| RFパラメータ | 200 trees、`random_state`固定 |
| BB数 | $3\times3\times2=18$個 |
| 説明点 | 各BBで予測1位と2位の確率差が小さい10点 |
| 合計説明点 | 180点 |
| 比較クラス | 各説明点の予測1位$c$と2位$d$ |
| 学習用摂動 | 300点 |
| 評価用摂動 | 独立に生成した1,000点 |
| 近傍生成 | $z=x+\mathcal N(0,I)\odot\sigma_X$ |
| LIME kernel width | $0.75\sqrt{D}$ |
| サロゲート | 重み付きRidge、$\alpha=1.0$ |
| 特徴予算 | 全特徴、3特徴、5特徴 |
| Stability反復 | 10回 |
| seed | 20260903 |

### 発表時に話す内容

Random Forestを用いた理由は、サロゲートと同じ線形モデルをBBにせず、非線形な確率面を局所近似する条件にするためである。説明点は、OVO説明が特に必要になる1位と2位が競合した点を選択した。

### 信頼区間

同じBBを共有する10説明点を独立標本として扱わず、18個のデータセット／BBごとに平均した値から探索的95%信頼区間を計算した。

---

## Slide 11. 比較手法

### スライドに載せる表

| 表示名 | 学習する量 | $\hat q_{c,d}$への変換 |
|---|---|---|
| OVR probability | $p_c,p_d$を別々に回帰 | $\hat p_c/(\hat p_c+\hat p_d)$ |
| OVR logit | $\mathrm{logit}(p_c),\mathrm{logit}(p_d)$を別々に回帰 | $\sigma(\hat h_c-\hat h_d)$ |
| OVO probability | $q_{c,d}$を直接回帰 | 回帰出力を$[0,1]$へclip |
| OVO log-ratio | $\log(p_c/p_d)$を直接回帰 | $\sigma(\hat\ell_{c,d})$ |

### 特徴選択

- 全特徴Ridgeの$|\beta_j\sigma_j|$で特徴を順位付け
- 上位$K$特徴だけで再フィット
- OVRは$c,d$で別々に選ぶため、表示時の特徴和集合が$K$を超える場合がある
- OVOはクラス対に対して1本なので、必ず$K$特徴

### 重要な注意

OVR logitは、CLIMAXのtarget-vs-restという考え方を模した補助ベースラインであり、サンプリング改善やinfluence functionを含むCLIMAX完全実装ではない。したがって「CLIMAXより優れている」とは結論しない。

---

## Slide 12. 結果：全特徴ではOVO probabilityが最良

### スライドに載せる表

| 手法 | pairwise MSE↓ | $R^2$↑ | 勝敗一致率↑ | Stability cosine↑ |
|---|---:|---:|---:|---:|
| OVR probability | 0.01478 | 0.5079 | 0.8069 | 0.9568 |
| OVR logit | 0.02011 | 0.2974 | 0.8067 | 0.9577 |
| **OVO probability** | **0.01353** | **0.5438** | **0.8103** | **0.9610** |
| OVO log-ratio | 0.01398 | 0.5320 | 0.8057 | 0.9580 |

OVO probability − OVR probabilityの対応あり差：

- MSE：$-0.001251\pm0.000599$（8.46%低減）
- $R^2$：$+0.03595\pm0.01409$
- 勝敗一致率：$+0.00338\pm0.00199$
- Stability cosine：$+0.00413\pm0.00173$

$\pm$は18個の独立BB平均に対する探索的95% CI。

### 図の案

横軸を手法、縦軸をMSEにした棒グラフを主図にする。AgreementとStabilityは右側に小さな数値カードとして置く。縦軸は差が小さいため、棒グラフを0始点にするか、差分グラフを使う。

### 発表時に話す内容

全特徴条件では、$q_{c,d}$を直接回帰するOVO probabilityがすべての主指標で最良だった。OVO log-ratioも確率MSEと$R^2$ではOVRより良いが、勝敗一致率では明確な改善がなかった。

ただし、$q_{c,d}$を直接学習し、$q_{c,d}$のMSEで評価しているため、OVO probabilityに有利な評価である。この結果は「OVO probabilityがすべての意味で優れる」ことではなく、**pairwise確率を説明したい場合には目的変数を直接合わせることが有効**だと解釈する。

---

## Slide 13. 結果と考察：少ない特徴で忠実度を維持できるか

### 5特徴条件

| 手法 | 実効特徴数↓ | MSE↓ | 勝敗一致率↑ | Stability cosine↑ |
|---|---:|---:|---:|---:|
| OVR probability | 7.15 | 0.01572 | 0.7971 | **0.9383** |
| **OVO probability** | **5.00** | **0.01484** | **0.7977** | 0.9286 |
| OVO log-ratio | **5.00** | 0.01527 | 0.7926 | 0.9237 |

OVO probabilityは、OVRより約2.15個少ない特徴で、MSEを5.63%低減し、勝敗一致率はほぼ同等だった。一方、Stability cosineは0.0097低下した。

### 3特徴条件

- OVRの実効特徴数：平均4.56
- OVOの実効特徴数：3.00
- OVO probabilityのMSE差：$-0.000164\pm0.000664$で明確な差なし
- 勝敗一致率：OVRより0.52ポイント低下
- Stability cosine：OVRより0.0175低下

### このスライドの結論

> OVOは、5特徴程度なら、より少ない表示特徴でpairwise fidelityを改善できる可能性がある。しかし、3特徴まで強く疎化すると、判断再現と安定性が悪化する。

### 考察

1. OVRでは$c,d$の特徴を別々に選ぶため、利用者は2つの説明を統合する必要がある。
2. OVOではペアに必要な特徴を1本のモデルで選べるため、説明を短くできる。
3. ただし、ペア専用の少数特徴を毎回選ぶことで、摂動サンプルに対する特徴選択の揺れが大きくなる。
4. log-ratioは係数の意味が明快だが、今回の確率fidelityでは$q_{c,d}$直接回帰が上回った。

---

## Slide 14. 限界

時間が短い場合はSlide 13下部かまとめへ統合する。

### スライドに載せる内容

- 合成表形式データ＋Random Forestのみ
- 予測1位と2位が競合する説明点に限定
- 特徴選択は公式LIMEの`lasso_path`完全再現ではない
- $q$-MSEは$q$直接回帰に有利であり、log-ratioの意味的利点とは分ける必要がある
- 18個のBBによる予備実験であり、正式な仮説検定・多重比較補正は未実施
- 人にとって本当に理解しやすいかは、今回の代理指標だけでは結論できない

### 言い方

> 今回定量化したのは、解釈性そのもののすべてではなく、局所忠実度、安定性、簡潔性という計算可能な側面である。

---

## Slide 15. まとめ

### スライドに載せる内容

1. 従来LIMEとCLIMAXの説明対象は、基本的にクラス$c$対restである
2. 利用者の「なぜ$c$であって$d$ではないか」という問いには、クラス対を直接説明するOVOが自然である
3. 元の多クラスBBを変更せず、

   $$
   q_{c,d}=\frac{p_c}{p_c+p_d}
   \quad\text{または}\quad
   \log\frac{p_c}{p_d}
   $$

   を局所疎線形回帰するOVO-LIMEを検討した
4. 予備実験では、OVO probabilityは5特徴で、OVRの平均7.15特徴より簡潔かつMSEが5.63%低かった
5. 一方、強い疎性ではstabilityが低下し、OVOが常に優れるわけではなかった

### 最後の一文

> OVO-LIMEの価値は、分類精度を変えることではなく、特定のfoilに対する問いと説明対象を一致させ、少数特徴で直接比較できる点にある。

### 今後

- 実データと複数BBで再検証
- 公式LIMEと同じ特徴選択条件で比較
- 境界付近・通常点・高確信点で条件分け
- fidelity–compactness–stabilityのPareto比較
- 必要に応じてユーザー指定foilや業務上重要なfoilへ拡張

---

# 補足スライド

## Appendix A. なぜ$p_c-p_d$を直接回帰するだけでは不十分か

同じ説明行列$Z$、重み$W$、Ridge正則化を使うとする。

$$
\hat\beta_c=(Z^\top WZ+\lambda I)^{-1}Z^\top Wy_c
$$

$$
\hat\beta_d=(Z^\top WZ+\lambda I)^{-1}Z^\top Wy_d
$$

したがって、

$$
\hat\beta_c-\hat\beta_d
=
(Z^\top WZ+\lambda I)^{-1}Z^\top W(y_c-y_d)
$$

となり、$p_c-p_d$の直接回帰とOVR 2本の差は一致する。

本研究でOVO固有の意味が生じるのは、目的変数を$q_{c,d}$または$\log(p_c/p_d)$へ変えるためである。

---

## Appendix B. OVR logitとpairwise log-ratioは異なる

CLIMAX型OVR logit：

$$
\log\frac{p_c}{1-p_c}
$$

2クラス分の差：

$$
\log\frac{p_c}{1-p_c}
-
\log\frac{p_d}{1-p_d}
=
\log\frac{p_c(1-p_d)}{p_d(1-p_c)}
$$

提案するpairwise log-ratio：

$$
\log\frac{p_c}{p_d}
$$

一般には両者は一致しない。OVR logit差は$c,d$の勝敗順序を保存するが、$q_{c,d}$としては較正されていない。

---

## Appendix C. 想定質問

### Q1. なぜBBを$c,d$だけで再学習しないのか

説明したい対象は、すでに運用・学習された元の多クラスBBである。2クラスで再学習すると別のモデルになり、元のBBの判断説明ではなくなる。本研究はpost-hocかつmodel-agnosticな説明を目的とする。

### Q2. なぜ摂動点から他クラス予測のサンプルを削除しないのか

削除するとペアごとに近傍分布が変わり、比較が不公平になる。また、foilクラスが局所的に1位にならない場合にサンプル不足が起きる。そのため摂動点はすべて残し、BB出力の$c,d$列だけからsoftな目的変数を作る。

### Q3. OVOなら$\binom{n}{2}$本必要ではないか

全ペアを作れば$\binom{n}{2}$本だが、通常は予測クラス$c^*$と、2位、ユーザー指定、危険な誤分類先などのfoilだけを説明すればよい。表示する説明数は1〜3本に絞れる。

### Q4. OVOの係数は因果効果か

因果効果ではない。説明点周辺で観測された、BBの入力とpairwise出力の局所的な関係である。

### Q5. log-ratioと$q$のどちらを採用するのか

- $q$：確率として直感的で、有界。今回のpairwise確率fidelityでは最良
- log-ratio：線形係数を相対オッズとして解釈でき、加法構造が明確

現段階では$q$を主手法、log-ratioを解釈重視の比較候補として残す。最終判断には実データと公式LIME条件での再検証が必要である。

### Q6. OVOは誰でも思いつくのに、本当に先行研究がないのか

OVOやfoil指定そのものには先行研究があるため、「誰も思いつかなかった」とは主張しない。

- OVO分類：古典的に存在する
- pairwise条件付き確率：pairwise couplingで既出
- OVO特徴帰属：GANMEXで既出
- 局所サロゲートによるfact–foil説明：Local Foil Treesで近縁例がある
- log-ratioによる確率分布の相対表現：統計学・多項ロジスティック回帰で既出

本研究で検証する差分は、これらの組合せである。

> 元の多クラスBBを変更せず、全摂動を保持したまま、任意の$c,d$についてsoftな$q_{c,d}$または$\log(p_c/p_d)$をLIME型局所疎線形モデルで直接近似し、OVRとの差をheld-out fidelity・stability・compactnessで評価する。

したがって研究上の貢献は、「単純なOVOの発明」ではなく、**説明対象の定式化、LIMEへの実装、表示戦略、評価による有効条件と限界の明確化**に置く。

### Q7. $\binom{n}{2}$本では計算量も表示量も大きすぎないか

表示時には全ペアを作らず、既定では1位対2位の1本だけをon-demand生成する。ユーザー指定foilでも1本である。全ペアが必要なのは、研究目的の網羅分析や監査の場合に限る。

全ペアを計算する場合も、BBへの問い合わせ結果は全ペアで共有できる。全特徴Ridgeならmulti-outputで一括計算でき、log-ratioでは基準クラスに対する$n-1$本から全ペアを復元できる。ただし、ペアごとのLasso特徴選択は$O(n^2)$で増えるため、クラス数が大きい場合はon-demand生成、上位候補への限定、キャッシュを使う。

---

# 参考文献

1. Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. *Proceedings of KDD 2016*, 1135–1144. <https://doi.org/10.1145/2939672.2939778>
2. Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the Social Sciences. *Artificial Intelligence*, 267, 1–38. <https://doi.org/10.1016/j.artint.2018.07.007>
3. Wu, T.-F., Lin, C.-J., and Weng, R. C. (2004). Probability Estimates for Multi-class Classification by Pairwise Coupling. *Journal of Machine Learning Research*, 5, 975–1005. <https://www.jmlr.org/papers/v5/wu04a.html>
4. Shih, S.-M., Tien, P.-J., and Karnin, Z. (2021). GANMEX: One-vs-One Attributions using GAN-based Model Explainability. *Proceedings of ICML 2021*, PMLR 139, 9592–9602. <https://proceedings.mlr.press/v139/shih21a.html>
5. van der Waa, J., Robeer, M., van Diggelen, J., Brinkhuis, M., and Neerincx, M. (2018). Contrastive Explanations with Local Foil Trees. *ICML Workshop on Human Interpretability in Machine Learning*. <https://research-portal.uu.nl/en/publications/contrastive-explanations-with-local-foil-trees-2/>
6. Galar, M., Fernández, A., Barrenechea, E., Bustince, H., and Herrera, F. (2011). An Overview of Ensemble Methods for Binary Classifiers in Multi-class Problems: Experimental Study on One-vs-One and One-vs-All Schemes. *Pattern Recognition*, 44(8), 1761–1776. <https://doi.org/10.1016/j.patcog.2011.01.017>
7. Zhou, Z., Hooker, G., and Wang, F. (2021). S-LIME: Stabilized-LIME for Model Explanation. *Proceedings of KDD 2021*, 2429–2438. <https://doi.org/10.1145/3447548.3467274>
8. Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt, Y., Schlötterer, J., van Keulen, M., and Seifert, C. (2023). From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI. *ACM Computing Surveys*, 55(13s), Article 295. <https://doi.org/10.1145/3583558>
9. Nanavati, S. and Prasad, R. (2023). CLIMAX: An Exploration of Classifier-Based Contrastive Explanations. *IEEE International Conference on Cognitive Machine Intelligence*, 49–58. <https://doi.org/10.1109/CogMI58952.2023.00017>
10. Sokol, K. and Flach, P. (2025). LIMEtree: Consistent and Faithful Surrogate Explanations of Multiple Classes. *Electronics*, 14(5), 929. <https://doi.org/10.3390/electronics14050929>

---

# 表現上の注意

発表では、次の表現を避ける。

- ×「LIMEはBBをone-vs-restで学習している」
  - ○「LIMEはBBのクラス確率をクラス別に説明するため、説明上は暗黙的なone-vs-restになる」
- ×「OVO説明は既存研究にない」
  - ○「GANMEXやLocal Foil Treesなど近縁研究は存在するが、soft pairwise出力を直接学ぶLIME型局所線形サロゲートと統一評価の組合せは、調査範囲で完全一致を確認できなかった」
- ×「OVOにより精度が上がった」
  - ○「BBの分類精度は不変で、pairwiseサロゲートのheld-out fidelityが改善した」
- ×「OVOはOVRより優れている」
  - ○「5特徴条件では、少ない実効特徴数でpairwise MSEを改善したが、stabilityは低下した」
- ×「特徴を増やすとクラス$c$になる確率が上がる」
  - ○「局所的に、クラス$d$に対するクラス$c$の相対的な支持が強まる」
- ×「OVOの係数は因果効果である」
  - ○「OVOの係数はBBの局所的な入出力関係を表す」
