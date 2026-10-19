"""
Measure KL divergence between two HDR images (EXR or PNG).

Usage:
    python kl_divergence.py <img_a> <img_b> [--bins 256] [--clip 10.0] [--channel]

KL(P || Q): P = img_a (reference/physics-based), Q = img_b (NN-based)
Histograms are estimated per channel in linear HDR space.
"""

import argparse
import numpy as np
import os
import sys


# -----------------------------------------------------------------------
def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.exr':
        return _load_exr(path)
    else:
        from PIL import Image
        img = np.array(Image.open(path)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img[:, :, :3] / 255.0


def _load_exr(path):
    import OpenEXR, Imath
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {c: np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
                for c in ['R', 'G', 'B']}
    return np.stack([channels['R'], channels['G'], channels['B']], axis=-1)


# -----------------------------------------------------------------------
def kl_divergence(p, q, bins, vmin, vmax, eps=1e-10):
    """KL(P || Q) estimated via histogram."""
    p_hist, edges = np.histogram(p, bins=bins, range=(vmin, vmax), density=False)
    q_hist, _     = np.histogram(q, bins=bins, range=(vmin, vmax), density=False)

    p_hist = p_hist.astype(np.float64) + eps
    q_hist = q_hist.astype(np.float64) + eps

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='KL divergence between two images')
    parser.add_argument('img_a', help='Reference image (physics-based)')
    parser.add_argument('img_b', help='Comparison image (NN-based)')
    parser.add_argument('--bins',    type=int,   default=256,
                        help='Number of histogram bins (default: 256)')
    parser.add_argument('--clip',    type=float, default=None,
                        help='Clip pixel values to [0, clip] before histogramming '
                             '(useful for HDR images with fireflies)')
    parser.add_argument('--channel', action='store_true',
                        help='Also show per-channel KL divergence')
    args = parser.parse_args()

    a = load_image(args.img_a)
    b = load_image(args.img_b)

    if a.shape != b.shape:
        print(f"WARNING: shapes differ  a={a.shape}  b={b.shape}", file=sys.stderr)

    clip = args.clip
    if clip is not None:
        a = np.clip(a, 0.0, clip)
        b = np.clip(b, 0.0, clip)

    vmin = 0.0
    vmax = float(max(a.max(), b.max())) if clip is None else clip

    ch_names = ['R', 'G', 'B']
    ch_kl = []
    for i, ch in enumerate(ch_names):
        kl = kl_divergence(a[:, :, i].ravel(), b[:, :, i].ravel(),
                           bins=args.bins, vmin=vmin, vmax=vmax)
        ch_kl.append(kl)

    # combined (flatten all channels together)
    kl_all = kl_divergence(a.ravel(), b.ravel(),
                           bins=args.bins, vmin=vmin, vmax=vmax)

    name_a = os.path.basename(args.img_a)
    name_b = os.path.basename(args.img_b)
    print(f"比較: {name_a}")
    print(f"  vs: {name_b}")
    print(f"  bins={args.bins}  clip={clip}  range=[{vmin:.3f}, {vmax:.3f}]")
    print("-" * 50)

    if args.channel:
        print(f"{'チャンネル':<8}  {'KL(P||Q)':>12}  {'KL(Q||P)':>12}  {'対称KL':>12}")
        print("-" * 50)

        kl_sym_all_r = kl_divergence(b[:, :, 0].ravel(), a[:, :, 0].ravel(),
                                      bins=args.bins, vmin=vmin, vmax=vmax)
        kl_sym_all_g = kl_divergence(b[:, :, 1].ravel(), a[:, :, 1].ravel(),
                                      bins=args.bins, vmin=vmin, vmax=vmax)
        kl_sym_all_b = kl_divergence(b[:, :, 2].ravel(), a[:, :, 2].ravel(),
                                      bins=args.bins, vmin=vmin, vmax=vmax)
        kl_rev = [kl_sym_all_r, kl_sym_all_g, kl_sym_all_b]

        for i, ch in enumerate(ch_names):
            sym = ch_kl[i] + kl_rev[i]
            print(f"{ch:<8}  {ch_kl[i]:>12.6f}  {kl_rev[i]:>12.6f}  {sym:>12.6f}")

        kl_all_rev = kl_divergence(b.ravel(), a.ravel(),
                                    bins=args.bins, vmin=vmin, vmax=vmax)
        sym_all = kl_all + kl_all_rev
        print("-" * 50)
        print(f"{'全体':<8}  {kl_all:>12.6f}  {kl_all_rev:>12.6f}  {sym_all:>12.6f}")
    else:
        kl_all_rev = kl_divergence(b.ravel(), a.ravel(),
                                    bins=args.bins, vmin=vmin, vmax=vmax)
        sym_all = kl_all + kl_all_rev
        print(f"KL(P||Q) [全体] = {kl_all:.6f}")
        print(f"KL(Q||P) [全体] = {kl_all_rev:.6f}")
        print(f"対称KL   [全体] = {sym_all:.6f}")


if __name__ == '__main__':
    main()
