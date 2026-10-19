"""
Khungurn ヤーンBSDF の 2D PDF-slice heatmap 作成スクリプト
論文 (Fu et al. SIGGRAPH Asia 2024) の Fig.5/10 スタイルに対応。

【表示内容】
  行0: Ground Truth  (khungurn_bsdf.eval による解析的 BSDF × cos θ_o)
  行1: Model PDF     (拡散モデルの学習済み sampling PDF)  ← --model 指定時

【軸の定義】
  横軸: phi_o   ∈ [-π, π]     方位角 [度]
  縦軸: theta_o ∈ [-π/2, π/2] 縦方向角 [度]
  各列: 異なる theta_i（入射縦方向角）

【出力ファイル】
  PNG : ヒートマップ全体の可視化画像（matplotlib figure）
  EXR : 各スライスの生データ（theta_i × row ごとに個別ファイル）
        ファイル名例: heatmap_fleece_gt_theta+0.exr
                      heatmap_fleece_model_theta-45.exr

Usage:
    # Ground truth のみ（どの素材でも可）
    python bsdf_heatmap_yarn.py --material cotton

    # Ground truth + Model PDF（idx=26 fleece のみ、checkpoint が揃っている）
    python bsdf_heatmap_yarn.py --material fleece --model 26

    # 出力先ディレクトリを指定
    python bsdf_heatmap_yarn.py --material fleece --model 26 --out_dir ./results
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import OpenEXR
import Imath

# rendering/ から直接実行できるようパス追加
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from utils.mitsuba_brdf_yarn import khungurn_bsdf

# デフォルト出力ディレクトリ（rendering/diffusion_bsdf_myresult/）
_DEFAULT_OUT_DIR = os.path.join(_SCRIPT_DIR, "diffusion_bsdf_myresult")


def _save_exr(path: str, data: np.ndarray):
    """float 配列を RGB EXR として保存する。
    data: (H, W, 3) → RGB 3ch、(H, W) → グレースケール (R=G=B)
    """
    h, w = data.shape[:2]
    header = OpenEXR.Header(w, h)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pt),
        "G": Imath.Channel(pt),
        "B": Imath.Channel(pt),
    }
    d = data.astype(np.float32)
    f = OpenEXR.OutputFile(path, header)
    if d.ndim == 3:
        f.writePixels({"R": d[:, :, 0].tobytes(),
                       "G": d[:, :, 1].tobytes(),
                       "B": d[:, :, 2].tobytes()})
    else:
        raw = d.tobytes()
        f.writePixels({"R": raw, "G": raw, "B": raw})
    f.close()
    print(f"  EXR: {path}")

# ---- 素材プリセット（bsdf_dict.py の idx=26〜30 と同じパラメータ） ----
MATERIALS = {
    "fleece": dict(
        idx=26,
        reflectance=[0.040, 0.087, 0.087], transmittance=[0.452, 0.725, 0.948],
        r_longwidth_deg=7.238, tt_longwidth_deg=10.000, tt_aziwidth_deg=25.989,
    ),
    "gabardine": dict(
        idx=27,
        reflectance=[0.185, 0.047, 0.069], transmittance=[0.999, 0.330, 0.354],
        r_longwidth_deg=2.141, tt_longwidth_deg=10.000, tt_aziwidth_deg=23.548,
    ),
    "silk": dict(
        idx=28,
        reflectance=[0.745, 0.008, 0.070], transmittance=[0.620, 0.553, 0.562],
        r_longwidth_deg=1.000, tt_longwidth_deg=10.000, tt_aziwidth_deg=19.823,
    ),
    "cotton": dict(
        idx=29,
        reflectance=[0.989, 0.959, 0.874], transmittance=[0.999, 0.999, 0.999],
        r_longwidth_deg=1.000, tt_longwidth_deg=27.197, tt_aziwidth_deg=38.269,
    ),
    "polyester": dict(
        idx=30,
        reflectance=[0.700, 0.700, 0.700], transmittance=[0.600, 0.000, 0.800],
        r_longwidth_deg=5.238, tt_longwidth_deg=10.000, tt_aziwidth_deg=25.000,
    ),
}


def _load_model(idx: int):
    """idx 番の yarn 拡散モデル (D_base, D_sample) を返す。"""
    from utils.model import NN_cond_pos, NN_cond_pretrain_spherical_one
    from utils.mlp_brdf_sampling import network_pdf_spherical

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(
        _SCRIPT_DIR, "..", "learning_repo_cleanup",
        "checkpoints_new", f"bsdf_{idx}_yarn",
    )

    rectify_path = os.path.join(ckpt_dir, f"brdf_rectify_network{idx}.pth")
    pretrain_path = os.path.join(ckpt_dir, f"brdf_pretrain_network{idx}.pth")

    for p in (rectify_path, pretrain_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                "（現在 checkpoint が揃っているのは idx=26 のみです）"
            )

    D_sample = NN_cond_pos(
        input_dim=6, output_dim=2, N_NEURONS=32,
        POSITIONAL_ENCODING_BASIS_NUM=5
    ).to(device)
    D_sample.load_state_dict(torch.load(rectify_path, map_location=device))
    D_sample.eval()

    D_base = NN_cond_pretrain_spherical_one(input_dim=2, N_NEURONS=16).to(device)
    D_base.load_state_dict(torch.load(pretrain_path, map_location=device))
    D_base.eval()

    return D_base, D_sample, network_pdf_spherical, device


def _eval_gt(bsdf, theta_i_rad, phi_i_rad, wo_flat):
    """グラウンドトゥルース BSDF × cos θ_o を grid 上で RGB 評価する。(N, 3) を返す。"""
    N = wo_flat.shape[0]
    wi_flat = torch.tensor([[theta_i_rad, phi_i_rad]],
                           dtype=torch.float32).expand(N, -1)
    with torch.no_grad():
        vals = bsdf.eval_rgb(wi_flat, wo_flat).numpy()  # (N, 3)
    return np.clip(vals, 0.0, None)


def _eval_model_pdf(D_base, D_sample, network_pdf_spherical, device,
                    theta_i_rad, phi_i_rad, wo_flat):
    """拡散モデルの sampling PDF を grid 上で評価する。

    network_pdf_spherical(D_base, D_sample, omega_o, omega_i, T=8) は
    omega_o（評価点）× omega_i（条件）の PDF を返す。
    """
    N = wo_flat.shape[0]
    wo_dev = wo_flat.to(device, dtype=torch.float32)
    wi_dev = torch.tensor([[theta_i_rad, phi_i_rad]],
                          dtype=torch.float32, device=device).expand(N, -1)
    # network_pdf_spherical は内部で Jacobian を backward() で計算するため
    # torch.no_grad() コンテキストは使用しない
    pdf = network_pdf_spherical(D_base, D_sample, wo_dev, wi_dev, T=8)
    return np.clip(pdf.detach().cpu().numpy(), 0.0, None)


def make_heatmap(
    material_name: str,
    theta_i_degs,
    phi_i_deg: float = 0.0,
    phi_center_deg: float = 0.0,
    res: int = 200,
    log_scale: bool = False,
    model_idx: int = None,
    out_dir: str = None,
):
    """論文 Fig.5/10 スタイルの PDF-slice heatmap を描画し、PNG と EXR を出力する。

    Parameters
    ----------
    material_name : str   素材名
    theta_i_degs  : list  入射縦方向角 [度] のリスト。各値が列1つ。
    phi_i_deg     : float 入射方位角 [度]（default 0）
    res           : int   theta_o/phi_o の解像度
    log_scale     : bool  True で対数スケール
    model_idx     : int   モデルを表示する場合の bsdf idx。None で GT のみ。
    out_dir       : str   出力先ディレクトリ（default: diffusion_bsdf_myresult/）
    """
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    basename = f"heatmap_{material_name}"
    if model_idx is not None:
        basename += f"_model{model_idx}"
    params = {k: v for k, v in MATERIALS[material_name].items() if k != "idx"}
    bsdf = khungurn_bsdf(**params)

    # モデルのロード
    if model_idx is not None:
        D_base, D_sample, net_pdf, device = _load_model(model_idx)
        n_rows = 2
        row_labels = ["Ground Truth", "Model PDF"]
    else:
        n_rows = 1
        row_labels = ["Ground Truth"]

    n_cols = len(theta_i_degs)

    # 出射方向グリッド
    # phi_center_deg: φ軸の表示中心 [度]
    #   0   → [-180°, 180°] Rローブ中心（デフォルト）
    #   180 → [  0°, 360°] TTローブ中心（φ_i=0のとき左右分割を解消）
    phi_c = math.radians(phi_center_deg)
    res_phi = res * 2  # φ の角度範囲 (360°) は θ (180°) の 2 倍
    theta_o_vals = torch.linspace(-math.pi / 2, math.pi / 2, res)
    phi_o_vals   = torch.linspace(phi_c - math.pi, phi_c + math.pi, res_phi)
    grid_theta, grid_phi = torch.meshgrid(theta_o_vals, phi_o_vals, indexing="ij")
    wo_flat = torch.stack([grid_theta.reshape(-1), grid_phi.reshape(-1)], dim=1)

    phi_i_rad = math.radians(phi_i_deg)

    # --- 全データを事前計算してスケールを統一 ---
    gt_maps, model_maps = [], []
    for theta_i_deg in theta_i_degs:
        theta_i_rad = math.radians(theta_i_deg)
        gt_rgb = _eval_gt(bsdf, theta_i_rad, phi_i_rad, wo_flat).reshape(res, res_phi, 3)
        gt_maps.append(gt_rgb)
        if model_idx is not None:
            mp = _eval_model_pdf(
                D_base, D_sample, net_pdf, device,
                theta_i_rad, phi_i_rad, wo_flat
            ).reshape(res, res_phi)
            model_maps.append(mp)

    # 輝度（単純平均 R+G+B/3）は表示・スケール計算に使用、RGB は EXR 保存に使用
    gt_lum_maps = [gt.mean(axis=-1) for gt in gt_maps]

    def _make_norm(maps, log_scale, fixed_vmax=None):
        all_vals = np.concatenate([m.ravel() for m in maps])
        positive = all_vals[all_vals > 0]
        if log_scale and len(positive) > 0:
            vmin = float(np.percentile(positive, 1))
            vmax = fixed_vmax if fixed_vmax is not None else float(np.max(positive))
            vmin = min(vmin, vmax)
            return mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=vmax)
        else:
            vmax = fixed_vmax if fixed_vmax is not None else float(np.max(all_vals))
            return mcolors.Normalize(vmin=0.0, vmax=vmax)

    gt_norm    = _make_norm(gt_lum_maps, log_scale)
    model_norm = _make_norm(model_maps, log_scale) if model_maps else None

    # --- 描画 ---
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    title = f"PDF slices — {material_name}"
    if phi_i_deg != 0.0:
        title += f"  (φ_i = {phi_i_deg:.0f}°)"
    fig.suptitle(title, fontsize=13)

    phi_min_deg = math.degrees(phi_c - math.pi)
    phi_max_deg = math.degrees(phi_c + math.pi)
    extent = [
        phi_min_deg,                  # phi_o min
        phi_max_deg,                  # phi_o max
        math.degrees(-math.pi / 2),   # theta_o min
        math.degrees(math.pi / 2),    # theta_o max
    ]
    # φ軸のtick: 中心と±90°, ±180° 相当の位置
    phi_ticks = sorted(set(int(round(phi_center_deg + d))
                           for d in [-180, -90, 0, 90, 180]))

    for col, (theta_i_deg, gt_lum) in enumerate(zip(theta_i_degs, gt_lum_maps)):
        ax = axes[0][col]
        im = ax.imshow(
            gt_lum, origin="lower", extent=extent, aspect="auto",
            cmap="inferno", norm=gt_norm,
        )
        ax.set_title(f"θ_i = {theta_i_deg:+.0f}°")
        if col == 0:
            ax.set_ylabel(f"{row_labels[0]}\nθ_o [°]")
        else:
            ax.set_ylabel("θ_o [°]")
        ax.set_xlabel("φ_o [°]")
        ax.set_xticks(phi_ticks)
        ax.set_yticks([-90, -45, 0, 45, 90])
        plt.colorbar(im, ax=ax, label="BSDF × cos θ_o")

    if model_idx is not None:
        for col, (theta_i_deg, mp) in enumerate(zip(theta_i_degs, model_maps)):
            ax = axes[1][col]
            im = ax.imshow(
                mp, origin="lower", extent=extent, aspect="auto",
                cmap="inferno", norm=model_norm,
            )
            ax.set_title(f"θ_i = {theta_i_deg:+.0f}°")
            if col == 0:
                ax.set_ylabel(f"{row_labels[1]}\nθ_o [°]")
            else:
                ax.set_ylabel("θ_o [°]")
            ax.set_xlabel("φ_o [°]")
            ax.set_xticks(phi_ticks)
            ax.set_yticks([-90, -45, 0, 45, 90])
            plt.colorbar(im, ax=ax, label="PDF")

    plt.tight_layout()

    # --- PNG 保存 ---
    png_path = os.path.join(out_dir, f"{basename}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  PNG: {png_path}")
    plt.close()

    # --- EXR 保存（各スライスを個別ファイルに） ---
    for theta_i_deg, gt in zip(theta_i_degs, gt_maps):
        ti_tag = f"theta{int(round(theta_i_deg)):+d}"
        exr_path = os.path.join(out_dir, f"{basename}_gt_{ti_tag}.exr")
        _save_exr(exr_path, gt)

    for theta_i_deg, mp in zip(theta_i_degs, model_maps):
        ti_tag = f"theta{int(round(theta_i_deg)):+d}"
        exr_path = os.path.join(out_dir, f"{basename}_model_{ti_tag}.exr")
        _save_exr(exr_path, mp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Khungurn yarn BSDF — PDF slice heatmap (論文 Fig.5/10 スタイル)"
    )
    parser.add_argument(
        "--material", choices=list(MATERIALS.keys()), default="cotton",
        help="素材名 (default: cotton)",
    )
    parser.add_argument(
        "--theta_i", nargs="+", type=float, default=[-45.0, 0.0, 45.0],
        help="入射縦方向角 theta_i [度] (default: -45 0 45)",
    )
    parser.add_argument(
        "--phi_i", type=float, default=0.0,
        help="入射方位角 phi_i [度] (default: 0)",
    )
    parser.add_argument(
        "--phi_center", type=float, default=0.0,
        help="φ_o 軸の表示中心 [度] (default: 0 → [-180,180]; 180 → [0,360] でTTローブ分割解消)",
    )
    parser.add_argument(
        "--res", type=int, default=200,
        help="解像度 (default: 200)",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="対数スケールで表示",
    )
    parser.add_argument(
        "--model", type=int, default=None, metavar="IDX",
        help="モデルPDFも表示する場合の bsdf idx (例: 26)。現在は 26 のみ対応。",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help=f"出力先ディレクトリ (default: diffusion_bsdf_myresult/)",
    )
    args = parser.parse_args()

    make_heatmap(
        material_name=args.material,
        theta_i_degs=args.theta_i,
        phi_i_deg=args.phi_i,
        phi_center_deg=args.phi_center,
        res=args.res,
        log_scale=args.log,
        model_idx=args.model,
        out_dir=args.out_dir,
    )
