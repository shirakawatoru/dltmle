# 設計メモ: カテゴリ治療・カテゴリ打ち切り / stremr wrapper 向け API

対象ブランチ: `dltmle-categorical-api`（ベース: `dev-seperateGQ` @ 9675d21）

## 0. 目的とスコープ

- PCORI Aim 1: 治療 A(t) を J クラス（例: SU / DPP4 / SGLT2i / GLP1）、打ち切り C(t) を K 区分
  （0 = 打ち切りなし、1..K-1 = 理由: non-CVD death / unknown death / administrative / disenrollment）に拡張する。
  アウトカムは生存（吸収的な二値イベント）のみ。
- PCORI Aim 3: R の stremr から reticulate / CLI で呼ぶための安定 API（long 形式入力、表形式の出力）。
- 既存の二値の使い方（`fit(rng_seed, hparams, W, L, A, C, Y, a)` など）は J=2/K=2 の特殊ケースとして残す。

## 1. データ規約

| 配列 | 形状 | 値 |
|---|---|---|
| `W` | (n, p_W) | ベースライン共変量 |
| `L` | (n, τ, p_L) | 時間依存共変量 |
| `A` | (n, τ, 1) | 治療クラス `0..J-1`（二値なら従来通り 0/1） |
| `C` | (n, τ, 1) | 打ち切り区分 `0..K-1`、0 = 打ち切りなし（吸収的: 以降は同じ値を持ち越し） |
| `Y` | (n, τ, 1) | 二値イベント（吸収的） |

推定対象は各評価時点 t の `psi(t) = P(Y_t^{g*} = 1)`（介入 g* の下でのリスク、全区分の打ち切りを除去）。
結果には `survival = 1 - psi` も併記する。

### 1.1 区間内の順序（stremr との違い）

- dltmle（既定、`censoring_timing="after_treatment"`）: `L_t → A_t → C_t → Y_t`
- stremr（`censoring_timing="before_treatment"`）: `L_t → C_t → A_t → Y_t`

どちらでもアウトカム列の行 t は「区間 t の治療の後に観測されるイベント」（stremr の `Y.tplus1`）で、
`C_t > 0` は「区間 t で打ち切り、Y_t は未観測」を意味する。

実装上の違いは次の3点だけで、引数 `censoring_timing` で切り替える:

1. g^C の条件付け: after は A_t を条件に含む（HTT の A_t トークンから読む）、before は含まない（L_t トークンから読む）。
   線形ベースラインも同様に特徴量を切り替える。
2. g^A の学習対象: before では打ち切り行（C_t > 0）の A_t は未観測なので尤度から外す。
3. long 形式変換: before では打ち切り行の治療・アウトカムが欠測でもよい（治療は直前の値で埋めるが、尤度にも clever covariate にも入らない）。

clever covariate は `1(C_t=0)/P(C_t=0|H) × g*(A_t|H)/g(A_t|H,C_t=0)`（before）と
`g*(A_t|H)/g(A_t|H) × 1(C_t=0)/P(C_t=0|H,A_t)`（after）で、ベイズの定理により同じ量
（どちらも `P(A_t, C_t=0 | H)` で割っている）。よって、打ち切り行で A_t が観測されているデータなら
どちらの順序でも同じパラメータを推定する（テスト `test_before_treatment_timing_agrees` で確認）。
stremr から渡すデータは `censoring_timing="before_treatment"` を指定すればよい。

## 2. モデル（`dltmle/model/`）

| ファイル | 変更 |
|---|---|
| `categorical.py`（新規） | 参照クラス符号化の softmax（`reference_softmax`）、多クラス NLL、`g*` の混合 `mix_logits`、`censoring_timing` の検証 |
| `g_model.py` | `G_a: Linear(d, J-1)`、`G_c: Linear(d, K-1)`、出力は確率 `p_A (B,T,J)`, `p_C (B,T,K)` |
| `q_model.py` | `forward(W, L, A) -> (logit_Q, logit_V_cls)`。target ネットワークの全クラス logit `(B,T,J)` を返し、介入での混合は外で行う |
| `linear.py` | 多項ロジスティック（治療は参照クラスを除いた one-hot）。Q は各クラスの logit を返す |
| `dltmle.py` | `batch["gstar"]` を使う forward、カテゴリ clever covariate、`from_models()`、targeting 診断の返却 |

- 参照クラス 0 の logit を 0 に固定する多項ロジット。J=2 では `sigmoid(Linear(z))` と同じパラメータ形状・初期化・数値になる。
  二値の損失は従来どおり BCE を使う。結果として、二値の推定値は旧実装と 1e-8 以内で一致する（§7）。
- g^C は K 区分の多クラス softmax（原因別の打ち切りハザード）。clever covariate に入るのは `P(C_t=0|·)` だけで、
  区分ごとの予測は診断に出す（観測頻度と比べる）。
- `init_bias`（hparam）: g ヘッドのバイアスを周辺対数オッズで初期化する。稀な打ち切り理由の学習が速くなる。
  新 API では既定 True、旧 `fit()` では既定 False（数値を変えないため）。

## 3. 介入（`dltmle/intervention.py`）

すべての介入を `gstar[i, t, j] = g*(A_t = j | 観測履歴)` の (n, τ, J) 配列に解決する。

- `Static(level)` / `Static([level_t0, level_t1, ...])`: 各時点のクラスを指定する。
- `Dynamic(fn)`: `fn(W, L, A) -> (n, τ)` のクラス。観測履歴で評価する（時点 t では `L[:, :t+1]`, `A[:, :t]` だけを使うこと）。
- `Stochastic(probs)`: `(J,)`、`(τ, J)`、`(n, τ, J)`、または関数。
- 旧形式もそのまま受け付ける: `a = np.ones_like(A)`（クラス指定）と、二値での `a ∈ (0,1)`（`P(A=1)`、旧来の確率的介入）。

ICE / TMLE で観測履歴を使って g* を評価するのは正しい。過去が介入と食い違う人は clever covariate の重みが 0 になるので、
寄与しない。

## 4. Targeting と影響関数

- Q-learning の擬似アウトカム: `V_t = Σ_a g*(a|H_t) Q_t(H_t, a)`。確率スケールで混合し、one-hot のときは旧実装と同じく選ばれた logit をそのまま使う。
- 揺らぎ: 全クラス・全時点で共通の ε を logit に足す（従来どおり）。確率的介入では反復のたびに `V_t` を全クラス logit + ε から作り直す
  （旧実装は fractional な `a` に対して `V` を `a.long()` で作っており、確率的介入の V が正しくなかった）。
- IC: `Σ_{t≥1} g_t (V*_t − Q*_t) + V*_0 − psi` を全標本の psi で中心化する。旧実装はバッチ平均で中心化していたので、
  SE がわずかに違う（golden テストでは推定値を 1e-6、SE を相対 2% で比較）。
- 診断: ε の総和と反復回数、収束理由、`|PnIC|/se`、累積重みの最大値・99% 点・クリップ率・ESS、g の較正（観測頻度と平均予測）、
  正値性の分位点。問題があれば `diagnostics["warnings"]` に文字列で出す。

## 5. 複数介入・複数評価時点

- 1回の呼び出しで `intervention={"SU": "SU", "SGLT2i": "SGLT2i", ...}` と `eval_times="all"` を指定できる。
- G（g^A, g^C）は1回だけ学習して全ジョブで共有する（causal mask のため、時点 t の予測は t までの履歴だけに依存し、
  打ち切った系列でも同じになる）。Q は（介入, 評価時点）ごとに別に学習する。
- 最初のジョブ（最初の介入, 最後の評価時点）は G と Q を交互に学習する従来のスケジュールなので、単一推定と完全に一致する。
  残りのジョブは `seed + k` で再シードし、Q だけを学習する。
- パネルは1度だけテンソル化し、ジョブごとにゼロコピーの view で切り出す。
- 対比（リスク差・リスク比、デルタ法、比は対数スケールの CI）は `DLTMLEResult.contrasts(reference)` で出す。
  ID 単位の IC を共有しているので共分散も正しく入る。

## 6. 公開 API

### 6.1 Python / reticulate

```python
res = dltmle.estimate_long(
    df, id_col="ID", time_col="t", outcome_col="Y.tplus1", treatment_col="drug",
    censoring_cols=["C_death", "C_unknown", "C_admin", "C_disenroll"],   # 1列のカテゴリでも可
    static_cols=[...], dynamic_cols=[...],
    treatment_levels=["SU", "DPP4", "SGLT2i", "GLP1"],
    censoring_timing="before_treatment",          # stremr の順序
    intervention={"SU": "SU", "SGLT2i": "SGLT2i",
                  "dyn": {"type": "column", "column": "d_dyn"},             # stremr の intervened_TRT 相当
                  "sto": {"type": "stochastic", "columns": ["p_SU", "p_DPP4", "p_SGLT2i", "p_GLP1"]}},
    estimator="dltmle", hparams={...}, seed=1, device="cuda", deterministic=True,
    eval_times="all",
)
res.estimates_frame()     # intervention, time, time_index, psi, se, ci_lower, ci_upper, survival, ..., n
res.ic_frame()            # id, intervention, time, time_index, ic（long 形式）
res.contrasts("SU")       # 差・比
res.to_dict()             # JSON 互換・列指向（R の data.frame にそのまま変換できる）
res.metadata()            # version, seed, device, deterministic, hparams（実際に使った全値）, levels, ...
```

- `estimate(W, L, A, C, Y, intervention, ...)`: 配列版（同じ結果型）。long 版とは完全一致する（テスト済み）。
- `estimate_file(path, ...)`: Parquet / CSV。Parquet を読むには `pip install dltmle[parquet]`（pyarrow）が要る。
- long 形式の規則: 各 ID の行は時間グリッドの先頭から連続していること。最初のイベント行または打ち切り行で追跡終了とし、
  それ以降の行は警告付きで捨てる。コードなしで途中終了した ID はエラーにする（`implicit_censoring=<level>` を指定すれば、
  次の時点に打ち切りを置く）。共変量は数値のみで、欠測は不可（R 側で前処理する）。`standardize=True` で z-score 化する。
- reticulate からの使い方: `dltmle <- import("dltmle"); res <- dltmle$estimate_long(df, ...); res$estimates_frame()` で
  pandas の DataFrame が R の data.frame に変換される。

### 6.2 CLI

```
python -m dltmle estimate --input data.parquet --id-col ID --time-col t --outcome-col Y \
    --treatment-col drug --censoring-cols C1,C2 --static-cols age,sex --dynamic-cols hba1c \
    --censoring-timing before_treatment \
    --intervention SU=static:SU --intervention dyn=column:d_dyn --intervention sto=stochastic:p0,p1,p2,p3 \
    --estimator dltmle --hparams hp.json --seed 1 --device cuda --deterministic --eval-times all \
    --output result.json --estimates-csv est.csv --ic-csv ic.csv --contrasts-csv con.csv --reference SU
python -m dltmle estimate --config run.json      # キーは estimate_long の引数と同じ。CLI フラグが優先
python -m dltmle version
```

- 出力 JSON: `{metadata, estimates, diagnostics, contrasts?, ic?}`。表はすべて列指向なので、
  `jsonlite::fromJSON` → `as.data.frame` でそのまま R の data.frame になる。
- 終了コード: 0 = 成功、2 = 入力エラー（stderr にメッセージ）。診断の警告は stderr に出す。

### 6.3 再現性・実行環境

- `seed`: torch / numpy / cuda を全部シードする。ジョブ k は `seed + k`。
- `device`: `"auto" | "cpu" | "cuda" | "cuda:N"`（CUDA が無いのに cuda を指定するとエラー）。
- `deterministic=True`: `torch.use_deterministic_algorithms(True, warn_only=True)`、cuDNN deterministic、
  `CUBLAS_WORKSPACE_CONFIG`（CUDA 初期化前に設定する必要がある）。勾配のある gather は one-hot 積に置き換えてある。
- `num_threads`: torch の CPU スレッド数（小さなモデルではスレッドのオーバーヘッドが大きい）。
- `DataLoader` を `BatchLoader`（バッチ単位のスライス）に置き換えた。RNG の消費は `DataLoader(shuffle=True)` と同じなので、
  旧実装と同じ乱数列になる。

## 7. 後方互換

- `fit` / `fit_linear` / `tune` のシグネチャはそのままで、キーワード引数（`n_treatment`, `n_censoring`, `device`,
  `deterministic`, `verbose`）を足しただけ。`fit()` は従来どおり `RALEstimate` を返し、`diagnostics` 属性を新たに持つ。
- Golden テスト: `dev-seperateGQ` で出した推定値（deep / linear × a=0/1）と 1e-6 以内で一致する（実測では差は約 1e-8）。
- 二値の float 入力と、明示的なカテゴリ（int, J=K=2, `Static(1)` / one-hot `gstar` / dict 指定）が完全に一致すること、
  旧形式の fractional `a` が `Stochastic([1-p, p])` と一致することをテストで確認している。
- 内部 API の変更: `QModel.forward` が `a` を取らなくなった。`GModel` は確率 `(B,T,J)`, `(B,T,K)` を返す。
  `DatasetWLACYa` は `a` の代わりに `gstar` を受け取る。

## 8. 死亡を競合事象として扱う拡張（未決 → 今回は打ち切り扱い）

今回は死亡（non-CVD death / unknown cause）も打ち切り区分として扱い、介入下で除去する
（「死亡が起きない仮想世界でのリスク」）。total effect（累積発生率）にしたくなった場合の拡張点は次のとおり:

1. `data.long_to_panel` に `competing_levels=[...]` を足し、該当する打ち切り区分を吸収的な競合事象配列 `D` に移す。
2. `DeepLTMLE._set_deterministic_Q` と `_q_loss`: D が起きた後は `V = Q = 0`（Y はもう起こりえない）、at-risk 行列は `(1−Y)(1−D)`。
3. clever covariate から D を外す（介入しない）。ターゲティングの solver の境界条件も同じように直す。
4. 区間内での D と Y の順序を決める（同時点の扱い）。

打ち切り区分は `censoring_levels` に名前付きで持っていて、変換の入口は `long_to_panel` 1か所なので、上の変更は局所的に済む。

## 9. ON TARGET DM 規模（約30万人、4クラス、400超の共変量、最大約1,000万 person-time）での注意

- パネルは密な `(n, τ, p)` の float32 テンソルとして CPU メモリに置き、バッチ単位で GPU に送る。
  例: 30万 × 48 × 400 × 4B ≈ 23 GB。入力検証で float64 の一時コピーも作るので、ピークは約2倍になる。
  必要になったら、memmap / float16 / ストリーミングの Dataset に置き換える（`BatchLoader` のインターフェースはそのまま使える）。
- Q はジョブ（介入 × 評価時点）ごとに学習するので、4介入 × 全時点だとコストが大きい。`eval_times` は必要な時点に絞るのを推奨する。
  将来の改善案として、全評価時点を同時に出す multi-horizon Q head を検討する。
- 確率的介入の妥当性チェック、正値性（重みのクリップ率・ESS）、ε の暴走（イベントが疎）は `diagnostics["warnings"]` に出る。

## 10. 未決事項・既知の制約

- 死亡を競合事象にするかどうか（§8）。
- stremr 側の列の対応（`Y.tplus1` = dltmle の Y_t、`CENS` の各列 = 打ち切り理由、`intervened_TRT` = `{"type":"column"}`）を、
  Aim 3 のプロトコルで確定すること。区間内順序は `censoring_timing="before_treatment"` で対応済み。
- 小標本で Q が誤特定のとき、IC ベースの SE の被覆がやや過小になる傾向がある（二値の `example_dgp` でも |z|≈3.5 の回が出る、
  既存の挙動）。シミュレーションでの被覆評価は別途必要。
- 稀なクラスの static 介入で、追従者にイベントがほぼ無いと ε が発散して psi≈0（SE≈0）になる。警告は出すが、
  Q の下限を設けるなどの対処は未実装。
- `tune()` はカテゴリデータでも動くが、long 形式 API とはまだ統合していない。
