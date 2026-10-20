# bias_check — 拡散モデルのサンプラ(MyYarnBSDF)の明るさの偏りの原因調査

既存のファイルは変更していません。補正はすべて `yarn_patches.py` で、読み込み時に差し替える形にしています。
対象は fleece(idx=26、チェックポイント `bsdf_26_yarn_curve`)、繊維の配置は seed 0 です。

## 結論
中間発表の「糸の部分が 3% 明るい」は、2つの誤差がたまたま打ち消し合った結果でした。

- **参照側**:参照画像は、バグのある繊維 BSDF(`neuralyarn/bsdf/khungurn.py`)で描かれていて、約 11% 暗くなっていた。
- **拡散モデル側**:下の3つの誤差があった。

すべてを直すと、NN の画像は修正版の参照と一致しました。糸の部分の輝度比は **1.0006 ± 0.0008** です。

| # | 原因 | 場所 | 影響 |
|---|---|---|---|
| 1 | 基底分布の `log_prob` の正規化の誤り。サンプリングでは σ = exp(log_scale) + 0.001 を使うのに、正規化には exp(log_scale) を使っている | `utils/model.py` `NN_cond_pretrain_spherical_one.log_prob` | 繊維1回の反射ごとに **−0.6%**。多重散乱で積み重なり、透過率の高い青ほど大きい |
| 2 | ODE で θ が [−π/2, π/2] の外に出る。範囲外の角度のまま BSDF を評価し(負の重みが出る)、PDF も片側の対応しか数えていない | `bsdf_myresult.py` `MyYarnBSDF.sample` | すれすれの入射で −2.5〜4%。画面全体では小さい |
| 3 | PyTorch 版の BSDF の値に、元の `gaussian_cdf` のバグ(N_TT の一様な裾の重み)が残っている。∫N_TT dφ = 1.0017 | `utils/mitsuba_brdf_yarn.py` `_uct_gaussian_pdf` | 透過のたびに **+0.17%**。多重散乱で画像全体が **+1.4%** |

MIS で使う逆 ODE の PDF と順方向の PDF の不一致(1サンプルごとに約 ±2%)は、明るさの偏りには影響しませんでした。ただし、MIS の PDF を統一すると、明るい点(ファイアフライ)は減りました。

なお、`diffusion_bsdf_myresult/scene_bsdf_fleece_fibers_nn.exr`(10/9、輝度比 0.927)は、同じコードで今描き直した画像(0.990)とも一致しません。今の環境では cuda_ad_rgb の初期化に失敗し、llvm_ad_rgb で描かれます。その画像はドライバ更新(595 → 610)の前に、cuda_ad_rgb で描かれた可能性があります(未確認)。

## 段階ごとの結果
**1. 参照画像の描き直し**(`render_yarn26_fibers_fixed.py`。512 / 4096 spp で、元の版と修正版を描画)
- 修正版 ÷ 元の版の輝度比は 1.114(B は 1.172)。
- 512 spp と 4096 spp の差は 0.1% 未満。

**2–3. 1回の反射の試験**(`s2_single_bounce.py`)
- E[eval/pdf] を、R と透過 TT に分けて ∫eval dω と比べた。
- 補正前:0.994(z = −7〜−13)。R と TT で差はない。
- 補正1を入れると 1.0001。補正1+2を入れると、すべての入射方向で 0.998〜1.001。

**4–5. 補正の組み合わせごとに描画**(`render_yarn26_fibers_nn_patched.py`、128 spp、`s4_compare.py`)
- 修正版の参照(4096 spp)に対する糸の部分の輝度比(標準誤差は約 0.0008)は、次の表のとおり。

| 描画 | 輝度比 |
|---|---|
| 補正なし | 0.990 |
| 補正1 | 1.014 |
| 補正1+2 | 1.013 |
| 補正1+2 + MIS の統一 | 1.014 |
| **補正1+2+3 + MIS の統一** | **1.0006** |
| 解析的なサンプラ(NN と同じ描画環境) | 0.9997 |
| 解析的なサンプラ + 元の BSDF の値 | 1.0146 |

- 最後の2行から分かること:描画の環境(pip 版 Mitsuba、llvm、`attach_curve_frames`)自体に偏りはない。残っていた +1.4% は、補正3の BSDF の値の違いで再現できる。

**6. 入射方向の網羅的な調査**(`s6_sweep_wi.py`、`figures/s6_sweep_wi.png`)
- θi 18 通り × φi 12 通りで、補正1+2 のサンプラを調べた。
- 1回の反射の比の平均は 0.9999(最小 0.994、最大 1.006、それぞれ ±0.2%)。

## 使い方(conda env:描画と試験は bsdf35、参照画像は neuralyarn、比較は bsdf-diff)
```
bash run_nn_variants.sh 128 final            # NN、補正1+2+3 と MIS の統一
bash run_nn_variants.sh 128 none analytic    # 補正なし / 解析的なサンプラ
bash run_s2.sh 524288 --fix_base --fix_chart # 1回の反射の試験
bash run_s6.sh 131072                        # 入射方向の網羅的な調査
python s4_compare.py 128                     # 輝度比の比較
```

## 本体に反映する場合の修正点(未反映)
1. `NN_cond_pretrain_spherical_one.log_prob` の正規化を `torch.log(torch.exp(log_scale) + self.eps)` にする。
   - 学習時の尤度にも同じ式が使われているので、再学習するかどうかは別に判断する。
2. `MyYarnBSDF.sample` / `pdf` を、`yarn_patches.PatchedYarnBSDF` と同じ扱いにする。具体的には、正規化した角度で評価し、2つの対応の PDF を足し、MIS の PDF を統一する。
3. `utils/mitsuba_brdf_yarn.py` の `_uct_gaussian_pdf` で、μ と σ を使う。
   - 学習側の `learning_repo_cleanup/utils/mitsuba_brdf_yarn.py` も同様。目標分布の差は 0.17% 程度。
