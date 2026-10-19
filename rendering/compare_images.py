"""
画像間の誤差指標を計算するスクリプト。
EXR（線形HDR）と PNG の両方に対応。

使い方:
    python compare_images.py <画像A> <画像B>
    python compare_images.py <画像A> <画像B> --clip 10   # 外れ値をクリップ
    python compare_images.py <画像A> <画像B> --channel   # チャンネル別表示

例:
    python compare_images.py diffusion_bsdf_myresult/scene_bsdf_fleece_fibers.exr \
                             diffusion_bsdf_myresult/scene_bsdf_fleece_fibers_nn.exr
"""

import argparse
import numpy as np
import os


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        return _load_exr(path)
    else:
        from PIL import Image
        img = np.array(Image.open(path)).astype(np.float32)
        # PNG は [0,255] → [0,1] に正規化
        if img.max() > 1.0:
            img = img / 255.0
        # RGBA なら RGB のみ使う
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img


def _load_exr(path):
    import OpenEXR, Imath
    f = OpenEXR.InputFile(path)
    dw = f.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = f.header()["channels"].keys()
    if "R" in channels and "G" in channels and "B" in channels:
        def ch(c):
            return np.frombuffer(f.channel(c, pt), np.float32).reshape(h, w)
        return np.stack([ch("R"), ch("G"), ch("B")], axis=-1)
    elif "Y" in channels:
        # グレースケール EXR
        raw = np.frombuffer(f.channel("Y", pt), np.float32).reshape(h, w)
        return raw[:, :, np.newaxis]
    else:
        raise ValueError(f"未対応チャンネル構成: {list(channels)}")


def compare(a, b, clip=None, per_channel=False):
    if a.shape != b.shape:
        raise ValueError(f"画像サイズが一致しません: {a.shape} vs {b.shape}")

    if clip is not None:
        a = np.clip(a, 0, clip)
        b = np.clip(b, 0, clip)

    diff = a - b
    diff2 = diff ** 2

    results = {}

    # --- 全チャンネル合計 ---
    mse  = diff2.mean()
    rmse = np.sqrt(mse)
    mae  = np.abs(diff).mean()

    # rMSE: 基準画像の平均輝度で正規化
    ref_mean = a.mean()
    rel_mse  = mse / (ref_mean ** 2 + 1e-8)

    results["all"] = {
        "MSE":    mse,
        "RMSE":   rmse,
        "MAE":    mae,
        "relMSE": rel_mse,
    }

    # --- チャンネル別 ---
    if per_channel and a.ndim == 3 and a.shape[2] == 3:
        names = ["R", "G", "B"]
        for i, name in enumerate(names):
            c_mse  = diff2[:, :, i].mean()
            c_rmse = np.sqrt(c_mse)
            c_mae  = np.abs(diff[:, :, i]).mean()
            c_ref  = a[:, :, i].mean()
            results[name] = {
                "MSE":    c_mse,
                "RMSE":   c_rmse,
                "MAE":    c_mae,
                "relMSE": c_mse / (c_ref ** 2 + 1e-8),
            }

    return results


def print_results(results, img_a, img_b):
    print(f"\n比較: {os.path.basename(img_a)}")
    print(f"  vs: {os.path.basename(img_b)}")
    print("-" * 48)

    order = ["all", "R", "G", "B"]
    labels = {"all": "全チャンネル", "R": "R チャンネル", "G": "G チャンネル", "B": "B チャンネル"}

    for key in order:
        if key not in results:
            continue
        r = results[key]
        print(f"\n【{labels[key]}】")
        print(f"  MSE    = {r['MSE']:.6f}")
        print(f"  RMSE   = {r['RMSE']:.6f}")
        print(f"  MAE    = {r['MAE']:.6f}")
        print(f"  relMSE = {r['relMSE']:.6f}  （基準画像の平均輝度で正規化）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像間の誤差指標を計算する")
    parser.add_argument("img_a", help="基準画像（ground truth）")
    parser.add_argument("img_b", help="比較画像（rendered / predicted）")
    parser.add_argument("--clip", type=float, default=None,
                        help="外れ値クリップ上限値（EXR の firefly 除去に有効）")
    parser.add_argument("--channel", action="store_true",
                        help="RGB チャンネル別の結果も表示する")
    args = parser.parse_args()

    a = load_image(args.img_a)
    b = load_image(args.img_b)

    results = compare(a, b, clip=args.clip, per_channel=args.channel)
    print_results(results, args.img_a, args.img_b)
