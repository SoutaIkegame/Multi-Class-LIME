# Multi-Class-LIME

OVR/OVOサロゲートのheld-out fidelity・pairwise agreement・stability比較：

```bash
python3 src/run_ovo_evaluation.py --quick  # スモークテスト
python3 src/run_ovo_evaluation.py          # 本実験
```

結果は`results/ovo_evaluation_*.csv`に出力される（`results/`はGit管理外）。

OVR全クラス・指定ペア用OVR 2本・OVO全ペア・OVO選択1ペアの計算時間スケーリング：

```bash
python3 src/benchmark_ovo_runtime.py --quick  # スモークテスト
python3 src/benchmark_ovo_runtime.py          # n、摂動数M、特徴次元Dを変えた本計測
```

結果は`results/runtime_benchmark_*.csv`に出力される。BB推論時間は含まず、共通のBB出力を得た後のサロゲート学習時間を測定する。

全特徴で一度だけ学習し、上位特徴を後から隠すdense→post-hoc表示実験：

```bash
python3 src/run_dense_posthoc_experiment.py --quick  # スモークテスト
python3 src/run_dense_posthoc_experiment.py          # 本実験
python3 src/run_dense_posthoc_experiment.py --digits --output-dir results/dense_posthoc_digits
```

表示特徴数は、dense性能を保つ最小K、検証MSE最小K、寄与エネルギー95%の3方針で選び、別のhold-out摂動で最終評価する。結果は`results/dense_posthoc_*.csv`に出力される。
