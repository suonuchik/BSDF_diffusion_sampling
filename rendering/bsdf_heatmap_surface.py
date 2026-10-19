"""
表面BSDF（金属・誘電体・Disney Principled など）の 2D PDF-slice heatmap
論文 (Fu et al. SIGGRAPH Asia 2024) の Fig.5/10 スタイル。

【可視化ドメイン】
  disk domain     : 単位円盤。中心 = 法線方向 (θ=0)、境界 = グレージング (θ=π/2)
  spherical domain: θ_o × φ_o の矩形マップ

【軸の定義 (spherical)】
  横軸: phi_o   ∈ [-π, π]  [度]
  縦軸: theta_o ∈ [0, π/2] [度]  ※ 反射半球のみ (transmission は除く)
  各列: 異なる theta_i

Usage:
    python bsdf_heatmap_surface.py --type roughconductor --theta_i 0 30 60
    python bsdf_heatmap_surface.py --type principled --roughness 0.2 --metallic 0.9
    python bsdf_heatmap_surface.py --type roughconductor --domain disk --log
    python bsdf_heatmap_surface.py --type roughconductor --save metal_heatmap.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# rendering/ から直接実行できるようパス追加
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# ---- Mitsuba の初期化（llvm / cuda 自動選択） ----
import mitsuba as mi
import drjit as dr

_mi_variant = None
for _v in ("cuda_ad_rgb", "llvm_ad_rgb"):
    try:
        mi.set_variant(_v)
        mi.load_dict({
            "type": "scene",
            "integrator": {"type": "path"},
            "sensor": {
                "type": "perspective",
                "film": {"type": "hdrfilm", "width": 1, "height": 1},
                "sampler": {"type": "independent"},
            },
        })
        _mi_variant = _v
        break
    except Exception:
        pass

if _mi_variant is None:
    raise RuntimeError("Mitsuba variant (cuda_ad_rgb / llvm_ad_rgb) が見つかりません。")

if _mi_variant != "cuda_ad_rgb":
    dr.set_thread_count(1)

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# BSDF 作成ヘルパー
# ---------------------------------------------------------------------------

def _make_bsdf(args):
    """コマンドライン引数から Mitsuba BSDF を生成する。"""
    t = args.type
    if t == "measured":
        bsdf_path = args.bsdf_file
        if not os.path.isabs(bsdf_path):
            bsdf_path = os.path.join(_SCRIPT_DIR, bsdf_path)
        if not os.path.exists(bsdf_path):
            raise FileNotFoundError(f"BSDF file not found: {bsdf_path}")
        return mi.load_dict({"type": "measured", "filename": bsdf_path})
    elif t == "roughconductor":
        return mi.load_dict({
            "type": "roughconductor",
            "material": args.conductor_material,
            "alpha_u": args.alpha_u,
            "alpha_v": args.alpha_v,
            "distribution": args.distribution,
        })
    elif t == "roughdielectric":
        return mi.load_dict({
            "type": "roughdielectric",
            "distribution": args.distribution,
            "alpha": args.alpha_u,
            "int_ior": args.int_ior,
            "ext_ior": "air",
        })
    elif t == "principled":
        return mi.load_dict({
            "type": "principled",
            "base_color": {"type": "rgb", "value": args.base_color},
            "metallic":        args.metallic,
            "specular":        args.specular,
            "roughness":       args.roughness,
            "anisotropic":     args.anisotropic,
            "spec_tint":       0.5,
            "sheen":           0.0,
            "sheen_tint":      0.0,
            "clearcoat":       0.0,
            "clearcoat_gloss": 0.0,
            "spec_trans":      0.0,
        })
    else:
        raise ValueError(f"Unknown BSDF type: {t}")


# ---------------------------------------------------------------------------
# 座標変換
# ---------------------------------------------------------------------------

def _sph_to_vec(theta, phi):
    """球面座標 (theta, phi) → Mitsuba Vector3f。局所座標 (z=法線方向)。"""
    st = np.sin(theta);  ct = np.cos(theta)
    sp = np.sin(phi);    cp = np.cos(phi)
    return mi.Vector3f(
        mi.Float(cp * st),
        mi.Float(sp * st),
        mi.Float(ct),
    )


def _grid_spherical(res, phi_center=0.0):
    """theta_o × phi_o の矩形グリッドを返す。

    Returns
    -------
    theta_o : (res, res) ndarray  ∈ [0, π/2]
    phi_o   : (res, res) ndarray  ∈ [phi_center-π, phi_center+π]
    wo_mi   : mi.Vector3f  (res*res 本) の出射方向ベクトル
    mask    : (res*res,) ndarray of bool  有効点（常に True、形式のため）
    """
    th = np.linspace(0.0, math.pi / 2, res)
    ph = np.linspace(phi_center - math.pi, phi_center + math.pi, res)
    TH, PH = np.meshgrid(th, ph, indexing="ij")  # (res, res)

    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)

    wo_mi = mi.Vector3f(
        mi.Float(x.ravel().astype(np.float32)),
        mi.Float(y.ravel().astype(np.float32)),
        mi.Float(z.ravel().astype(np.float32)),
    )
    mask = np.ones(res * res, dtype=bool)
    return TH, PH, wo_mi, mask


def _grid_disk(res):
    """単位円盤グリッド (paper Fig.5 の左側スタイル) を返す。

    Returns
    -------
    u, v    : (res, res) ndarray  ∈ [-1, 1]
    wo_mi   : mi.Vector3f  (res*res 本)  ※ 円盤外は z=1（無効点）
    mask    : (res*res,) ndarray of bool  円盤内の有効点
    """
    uv = np.linspace(-1.0, 1.0, res)
    U, V = np.meshgrid(uv, uv, indexing="ij")  # (res, res)
    r2 = U ** 2 + V ** 2
    mask = (r2 <= 1.0).ravel()

    # 円盤内: z = sqrt(1 - r²)、円盤外はダミーで z=1 (後でマスク)
    Z = np.where(r2 <= 1.0, np.sqrt(np.clip(1.0 - r2, 0.0, None)), 1.0)

    wo_mi = mi.Vector3f(
        mi.Float(U.ravel().astype(np.float32)),
        mi.Float(V.ravel().astype(np.float32)),
        mi.Float(Z.ravel().astype(np.float32)),
    )
    return U, V, wo_mi, mask


# ---------------------------------------------------------------------------
# BSDF 評価
# ---------------------------------------------------------------------------

def _eval_bsdf(bsdf, wi_mi, wo_mi, N):
    """bsdf.eval(wi, wo) を N 点で評価し輝度 (N,) を返す。"""
    si = dr.zeros(mi.SurfaceInteraction3f, N)
    si.wi = wi_mi
    vals = bsdf.eval(mi.BSDFContext(), si, wo_mi)
    # RGB → 輝度
    lum = 0.2126 * vals.x + 0.7152 * vals.y + 0.0722 * vals.z
    return np.array(lum.numpy(), dtype=np.float32)


# ---------------------------------------------------------------------------
# heatmap 描画
# ---------------------------------------------------------------------------

def make_heatmap(bsdf, bsdf_label, theta_i_degs, domain, res, log_scale,
                 phi_i_deg=0.0, phi_center_deg=0.0, save_path=None):
    """
    Parameters
    ----------
    bsdf           : Mitsuba BSDF オブジェクト
    bsdf_label     : タイトル用文字列
    theta_i_degs   : list[float]  入射極角 [度]（各値が列1つ）
    domain         : "spherical" | "disk"
    res            : int  グリッド解像度
    log_scale      : bool  対数スケール
    phi_i_deg      : float 入射方位角 [度]
    phi_center_deg : float φ_o 軸の表示中心 [度] (spherical のみ有効)
    save_path      : str | None
    """
    phi_i_rad = math.radians(phi_i_deg)
    n_cols = len(theta_i_degs)

    # グリッド生成
    if domain == "disk":
        coord1, coord2, wo_mi, mask = _grid_disk(res)
        extent = [-1, 1, -1, 1]
        xlabel, ylabel = "u", "v"
        xticks = yticks = [-1, -0.5, 0, 0.5, 1]
    else:
        phi_c = math.radians(phi_center_deg)
        coord1, coord2, wo_mi, mask = _grid_spherical(res, phi_center=phi_c)
        phi_min_deg = math.degrees(phi_c - math.pi)
        phi_max_deg = math.degrees(phi_c + math.pi)
        extent = [phi_min_deg, phi_max_deg, 0, math.degrees(math.pi / 2)]
        xlabel, ylabel = "φ_o [°]", "θ_o [°]"
        xticks = sorted(set(int(round(phi_center_deg + d))
                            for d in [-180, -90, 0, 90, 180]))
        yticks = [0, 15, 30, 45, 60, 75, 90]

    N = res * res

    # --- 全列を計算してスケール統一 ---
    maps = []
    for theta_i_deg in theta_i_degs:
        theta_i_rad = math.radians(theta_i_deg)
        wi_mi = _sph_to_vec(theta_i_rad, phi_i_rad)
        wi_mi = mi.Vector3f(
            dr.repeat(wi_mi.x, N),
            dr.repeat(wi_mi.y, N),
            dr.repeat(wi_mi.z, N),
        )
        vals = _eval_bsdf(bsdf, wi_mi, wo_mi, N)
        vals[~mask] = np.nan  # 円盤外は NaN（diskモード）
        maps.append(np.clip(vals, 0.0, None).reshape(res, res))

    # カラーノルム
    all_vals = np.concatenate([m[~np.isnan(m)] for m in maps])
    positive = all_vals[all_vals > 0]
    if log_scale and len(positive) > 0:
        vmin = float(np.percentile(positive, 2))
        vmax = float(np.max(positive))
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=vmax)
    else:
        vmax = float(np.percentile(all_vals[~np.isnan(all_vals)], 99))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    # --- 描画 ---
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.2 * n_cols, 4.0),
        squeeze=False,
    )
    title = f"PDF slices — {bsdf_label}  [{domain} domain]"
    if phi_i_deg != 0.0:
        title += f"  φ_i={phi_i_deg:.0f}°"
    fig.suptitle(title, fontsize=12)

    for col, (theta_i_deg, mp) in enumerate(zip(theta_i_degs, maps)):
        ax = axes[0][col]

        if domain == "disk":
            # 円盤外を白で塗る（NaN → facecolor）
            cmap = plt.cm.inferno.copy()
            cmap.set_bad("white")
            im = ax.imshow(
                mp, origin="lower", extent=extent, aspect="equal",
                cmap=cmap, norm=norm,
            )
            # 円の輪郭
            circle = plt.Circle((0, 0), 1.0, fill=False,
                                 edgecolor="gray", linewidth=0.8)
            ax.add_patch(circle)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
        else:
            im = ax.imshow(
                mp, origin="lower", extent=extent, aspect="auto",
                cmap="inferno", norm=norm,
            )

        ax.set_title(f"θ_i = {theta_i_deg:+.0f}°")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        plt.colorbar(im, ax=ax, label="BSDF luminance")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="表面BSDF PDF-slice heatmap (論文 Fig.5/10 スタイル)"
    )
    parser.add_argument(
        "--type",
        choices=["roughconductor", "roughdielectric", "principled", "measured"],
        default="roughconductor",
        help="BSDF タイプ (default: roughconductor)",
    )
    parser.add_argument(
        "--bsdf_file", type=str,
        default="measuredbsdfs/aniso_miro_7_rgb.bsdf",
        help="measured タイプ時の .bsdf ファイルパス "
             "(rendering/ からの相対パス、default: measuredbsdfs/aniso_miro_7_rgb.bsdf)",
    )
    # roughconductor
    parser.add_argument("--conductor_material", default="Cu",
                        help="金属素材名 (Cu, Al, Au, Ag, ... default: Cu)")
    parser.add_argument("--alpha_u", type=float, default=0.3,
                        help="表面粗さ alpha_u (default: 0.3)")
    parser.add_argument("--alpha_v", type=float, default=0.3,
                        help="表面粗さ alpha_v (default: 0.3)")
    parser.add_argument("--distribution", default="ggx",
                        choices=["ggx", "beckmann"],
                        help="マイクロファセット分布 (default: ggx)")
    # roughdielectric
    parser.add_argument("--int_ior", default="bk7",
                        help="内部屈折率素材名 (default: bk7)")
    # principled
    parser.add_argument("--base_color", nargs=3, type=float,
                        default=[1.0, 1.0, 1.0], metavar=("R", "G", "B"),
                        help="ベースカラー RGB (default: 1 1 1)")
    parser.add_argument("--metallic", type=float, default=0.9)
    parser.add_argument("--specular", type=float, default=0.8)
    parser.add_argument("--roughness", type=float, default=0.2)
    parser.add_argument("--anisotropic", type=float, default=0.0)
    # 共通
    parser.add_argument(
        "--theta_i", nargs="+", type=float, default=[0.0, 30.0, 60.0],
        help="入射極角 theta_i [度] (default: 0 30 60)",
    )
    parser.add_argument("--phi_i", type=float, default=0.0,
                        help="入射方位角 phi_i [度] (default: 0)")
    parser.add_argument("--phi_center", type=float, default=0.0,
                        help="φ_o 軸の表示中心 [度] (spherical のみ有効, default: 0 → [-180,180])")
    parser.add_argument(
        "--domain", choices=["spherical", "disk"], default="spherical",
        help="可視化ドメイン: spherical (矩形) または disk (円形) (default: spherical)",
    )
    parser.add_argument("--res", type=int, default=256,
                        help="解像度 (default: 256)")
    parser.add_argument("--log", action="store_true",
                        help="対数スケールで表示")
    parser.add_argument("--save", type=str, default=None,
                        help="保存先パス (省略時は画面表示)")

    args = parser.parse_args()

    bsdf = _make_bsdf(args)
    label = {
        "roughconductor": f"RoughConductor ({args.conductor_material}, α={args.alpha_u})",
        "roughdielectric": f"RoughDielectric ({args.int_ior}, α={args.alpha_u})",
        "principled":     f"Principled (metallic={args.metallic}, rough={args.roughness})",
        "measured":       os.path.splitext(os.path.basename(args.bsdf_file))[0],
    }[args.type]

    make_heatmap(
        bsdf=bsdf,
        bsdf_label=label,
        theta_i_degs=args.theta_i,
        domain=args.domain,
        res=args.res,
        log_scale=args.log,
        phi_i_deg=args.phi_i,
        phi_center_deg=args.phi_center,
        save_path=args.save,
    )
