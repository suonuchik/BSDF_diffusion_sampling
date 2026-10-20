import torch
import math


# ---------------------------------------------------------------------------
# 座標変換
# ---------------------------------------------------------------------------

def yarn_to_cart(theta_phi):
    """ヤーン繊維座標 (theta, phi) を 3D 直交座標に変換する。

    【表面BSDFとの違い】
    表面BSDF（mitsuba_brdf_scalar.py の spher_to_cart）は法線を z 軸に取る:
      x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)   θ ∈ [0, π]

    ヤーンBSDF（Khungurn et al. 2015）は繊維軸を x 軸に取る:
      x = sin(θ)                  ← 繊維軸方向の成分
      y = cos(θ)cos(φ)
      z = cos(θ)sin(φ)            θ ∈ [-π/2, π/2]

    θ の意味も変わる:
      表面BSDF : z 軸（法線）からの極角
      ヤーンBSDF: 繊維軸に垂直な赤道面からの縦方向角（longitudinal angle）
    """
    theta = theta_phi[..., 0]
    phi   = theta_phi[..., 1]
    return torch.stack([
        torch.sin(theta),
        torch.cos(theta) * torch.cos(phi),
        torch.cos(theta) * torch.sin(phi),
    ], dim=-1)


def _regularize(angle):
    """角度を [-π, π] の範囲に収める。"""
    angle = angle % (2.0 * math.pi)
    return torch.where(angle > math.pi, angle - 2.0 * math.pi, angle)


# ---------------------------------------------------------------------------
# 確率分布のヘルパー
# ---------------------------------------------------------------------------

def _gaussian_pdf(x, mu, sigma):
    """ガウス分布の確率密度 N(x; mu, sigma)"""
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) \
           * torch.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def _std_gaussian_cdf(x):
    """標準正規分布の累積分布関数 Φ(x)"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _gaussian_integral(a, b, mu, sigma):
    """平均 mu・標準偏差 sigma のガウス分布の [a, b] 上の積分 Φ((b-mu)/σ) - Φ((a-mu)/σ)

    移植元の neuralyarn の bsdf/khungurn.py は gaussian_cdf で mu, sigma を無視しており、
    以前はここでもその挙動を再現していた。その場合 Φ(π)-Φ(-π) = 0.9983 となり、
    N_TT の積分が 1.0017 になって、透過のたびにエネルギーが 0.17% 増えていた
    （rendering/bias_check/README.md）。修正版 khungurn_fixed.py に合わせて mu, sigma を使う。
    """
    a_t = torch.as_tensor((a - mu) / sigma, dtype=torch.float32)
    b_t = torch.as_tensor((b - mu) / sigma, dtype=torch.float32)
    return _std_gaussian_cdf(b_t) - _std_gaussian_cdf(a_t)


def _uct_gaussian_pdf(x, mu, sigma, a, b):
    """範囲外を一様分布で補完したガウスPDF（uct = uniform-compensated truncated）

    通常の切断ガウス分布では範囲 [a,b] 外に落ちたサンプルを棄却するが、
    この uct バージョンでは代わりに一様分布 U(a, b) を混合することで
    必ずサポート内にサンプルが収まるようにする（重要サンプリングの安定化）。
    """
    uniform_weight = (1.0 - _gaussian_integral(a, b, mu, sigma)) / (b - a)
    return _gaussian_pdf(x, mu, sigma) + uniform_weight


# ---------------------------------------------------------------------------
# cos² 正規化ガウス（縦方向散乱ローブの核となる関数）
# ---------------------------------------------------------------------------

def _G_integral(theta, mean, std):
    """cos²(θ) × N(θ; mean, std) の不定積分（cos² を多項式近似したもの）を解析的に計算する。

    【なぜ解析積分が必要か】
    縦方向ローブ M(θ) は以下で定義される:
        M(θ; mean, std) = N(θ; mean, std) / G(mean, std)
    分母 G は cos²(θ) で重み付けした正規化定数であり、数値積分を避けるため
    cos²(θ) を 8 次多項式で近似して解析的に積分する。

    多項式近似: cos²(θ) ≈ Σ p[k] θ^k
    係数 p[0..8] は neuralyarn の bsdf/khungurn.py と同じ値。テイラー展開ではなく
    [-π/2, π/2] 全体に合わせた係数で、θ = ±π/2 でもほぼ 0 になる。

    【積分の求め方】
    P(θ) = A + (θ - mean)·B(θ) - std²·B'(θ) を満たす定数 A と 7 次多項式 B を
    求めると、N'(θ) = -(θ - mean)/std² · N(θ) より
        ∫ P(θ) N(θ) dθ = (A/2)·erf((θ - mean)/(√2·std)) - std²·B(θ)·N(θ)
    となる。
    """
    std2 = std ** 2
    p = [1.0001, 0.0, -0.999745, 0.0, 0.3322, 0.0, -0.04301, 0.0, 0.002439]

    # 両辺の θ^(k+1) の係数を比べて、B の係数 b[k] を高次から順に求める
    #   b[k] = p[k+1] + mean·b[k+1] + (k+2)·std²·b[k+2]
    b = [0.0] * 9
    b[7] = p[8]
    b[6] = p[7] + mean * b[7]
    for j in range(5, -1, -1):
        b[j] = p[j + 1] + mean * b[j + 1] + (j + 2) * std2 * b[j + 2]

    B = sum(b[k] * theta ** k for k in range(8))
    A = p[0] + mean * b[0] + std2 * b[1]   # 定数項の比較: p[0] = A - mean·b[0] - std²·b[1]

    return (A / 2.0) * torch.erf((theta - mean) / (math.sqrt(2.0) * std)) \
           - std2 * B * _gaussian_pdf(theta, mean, std)


def _G(mean, std):
    """cos² 正規化定数 G(mean, std) = ∫_{-π/2}^{π/2} cos²(θ) N(θ; mean, std) dθ"""
    half_pi = torch.full_like(mean, math.pi / 2.0)
    return _G_integral(half_pi, mean, std) - _G_integral(-half_pi, mean, std)


def _cos2_normalized_gaussian(theta, mean, std):
    """縦方向散乱ローブ M(θ; mean, std) = N(θ; mean, std) / G(mean, std)

    分子は通常のガウスで、cos² は掛かっていない。G で割ることで
        ∫_{-π/2}^{π/2} M(θ) cos²(θ) dθ = 1
    となるように正規化されている。eval が掛ける cos(θ_o) と立体角要素
    dω = cos(θ)dθdφ の cos(θ) を合わせると cos² になるため、この正規化で
    ∫ eval dω = F_R + (1 - F_R)·c_tt ≤ 1（エネルギー保存）が成り立つ。
    """
    g = _G(mean, std).clamp(min=1e-8)
    return _gaussian_pdf(theta, mean, std) / g


# ---------------------------------------------------------------------------
# Khungurn BSDF クラス
# ---------------------------------------------------------------------------

class khungurn_bsdf:
    """ヤーン繊維BSDF（Khungurn et al. 2015）の純 PyTorch 実装。

    【物理的なモデル】
    繊維への光の散乱は主に 2 つのローブで構成される:
      R  ローブ: 繊維表面での鏡面反射（Specular Reflection）
                  → 縦方向: cos² 正規化ガウス（幅 r_longwidth）
                  → 方位角: 一様分布 1/(2π)
      TT ローブ: 繊維を貫通した前方透過（Transmitted-Transmitted）
                  → 縦方向: cos² 正規化ガウス（幅 tt_longwidth）
                  → 方位角: uct ガウス（幅 tt_aziwidth、中心はφ方向の対面）

    【座標系】
    入出力の (theta, phi) はヤーン繊維座標:
      theta ∈ [-π/2, π/2] : 縦方向角（繊維軸 x に垂直な面からの角度）
      phi   ∈ [-π,   π]   : 方位角（繊維軸 x まわり）

    【Mitsuba を使わない理由】
    Khungurn.eval() は解析式のみで構成されるため、Mitsuba / drjit なしで
    PyTorch に移植できる。これにより scalar_rgb との variant 競合を回避できる。
    """

    def __init__(self, reflectance, transmittance,
                 r_longwidth_deg, tt_longwidth_deg, tt_aziwidth_deg):
        """
        Parameters
        ----------
        reflectance      : list [r, g, b]  繊維の反射色
        transmittance    : list [r, g, b]  繊維の透過色
        r_longwidth_deg  : float  R  ローブの縦方向広がり [度]
        tt_longwidth_deg : float  TT ローブの縦方向広がり [度]
        tt_aziwidth_deg  : float  TT ローブの方位角広がり [度]
        """
        def lum(rgb):
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        # RGB カラーを輝度スカラーに変換
        # 理由: 既存の BSDF ラッパー（mitsuba_brdf_scalar.py）と
        #       インターフェースを揃えるため、スカラー輝度を返す
        self.c_r  = lum(reflectance)
        self.c_tt = lum(transmittance)

        # レンダリング用に RGB も保持する（eval_rgb で使用）
        # 学習（MCMC の目標密度）は輝度 eval() のままでよいが、
        # レンダリングで輝度を使うと色が失われ、物理ベースの結果と一致しない
        self.c_r_rgb  = list(reflectance)
        self.c_tt_rgb = list(transmittance)

        # 度数 → ラジアン変換
        self.r_longwidth  = math.radians(r_longwidth_deg)
        self.tt_longwidth = math.radians(tt_longwidth_deg)
        self.tt_aziwidth  = math.radians(tt_aziwidth_deg)

    def eval(self, wi, wo):
        """BSDF × cos(theta_o) を返す。

        【なぜ cos(theta_o) が掛かっているか】
        移植元の neuralyarn の Khungurn.eval() が value = (S_R + S_TT) * cos(theta_o)
        を返しており、それをそのまま再現している。移植元では Mitsuba の BSDF 規約
        （eval は f × cos を返す）の cos 項として扱われている。M の cos² 正規化は
        この cos(θ_o) と立体角要素 dω = cos(θ)dθdφ の cos(θ) を合わせた形になっている。

        なお、ヤーン繊維座標の立体角要素 dω = cos(θ)dθdφ（表面座標の sin(θ)dθdφ
        とは異なる）の Jacobian も cos(θ) なので、本リポジトリでは eval を
        (θ, φ) 空間での MCMC の目標密度として直接使用している。

        Parameters
        ----------
        wi : (N, 2) Tensor  入射方向 (theta_i, phi_i)
        wo : (N, 2) Tensor  出射方向 (theta_o, phi_o)

        Returns
        -------
        (N,) Tensor  BSDF 輝度値 × cos(theta_o)
        """
        return self._eval(wi, wo, self.c_r, self.c_tt)

    def eval_rgb(self, wi, wo):
        """eval() の RGB 版。BSDF × cos(theta_o) を (N, 3) で返す。

        eval() は反射色・透過色を輝度に潰しているため、そのまま
        レンダリングに使うと灰色の BSDF になる。レンダリングではこちらを使う。
        """
        c_r  = torch.tensor(self.c_r_rgb,  dtype=wi.dtype, device=wi.device)
        c_tt = torch.tensor(self.c_tt_rgb, dtype=wi.dtype, device=wi.device)
        return self._eval(wi.unsqueeze(1), wo.unsqueeze(1), c_r, c_tt)

    def _eval(self, wi, wo, c_r, c_tt):
        """eval / eval_rgb の共通実装。
        輝度版: wi, wo は (N, 2)、c_r, c_tt はスカラー → (N,)
        RGB 版: wi, wo は (N, 1, 2)、c_r, c_tt は (3,) → (N, 3)
        """
        theta_i = wi[..., 0]
        phi_i   = wi[..., 1]
        theta_o = wo[..., 0]
        phi_o   = wo[..., 1]

        # ---- Fresnel 反射率（Schlick 近似） ----
        # F_R が大きいほど R ローブが強く、TT ローブが弱くなる
        F_R = c_r + (1.0 - c_r) \
              * (1.0 - torch.abs(torch.cos(theta_i))) ** 5

        # ---- R ローブ（鏡面反射成分） ----
        # 縦方向: -theta_i を中心としたガウス（入射と対称な角度に散乱）
        # 方位角: 一様分布（Khungurn モデルの簡略化。実際の円柱での鏡面反射は
        #         方位角に依存するが、全方位に広く散らばるため一様で近似している）
        M_R = _cos2_normalized_gaussian(theta_o, -theta_i, self.r_longwidth)
        N_R = 1.0 / (2.0 * math.pi)
        S_R = F_R * M_R * N_R

        # ---- TT ローブ（前方透過成分） ----
        # 縦方向: -theta_i を中心としたガウス（入射と対称な縦角に透過）
        # 方位角: phi_i + π（真正面）を中心としたガウス（前方透過）
        M_TT  = _cos2_normalized_gaussian(theta_o, -theta_i, self.tt_longwidth)
        phi_tt = _regularize(phi_i + math.pi)   # TT の基準方位角（真正面）
        phi_d  = _regularize(phi_o - phi_tt)    # 基準からのずれ
        N_TT  = _uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -math.pi, math.pi)
        S_TT  = (1.0 - F_R) * c_tt * M_TT * N_TT

        return (S_R + S_TT) * torch.cos(theta_o)
