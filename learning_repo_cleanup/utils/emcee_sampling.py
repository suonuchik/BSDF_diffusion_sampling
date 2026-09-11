import emcee
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.analytical_brdf_torch import *
from utils.utils_sampling_torch_disk import stratified_sample_wo,stratified_sampling_2d
import torch


def lnprob_brdf_disk(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_omegai = x0 ** 2 + y0 ** 2 > rmax ** 2 or x0 ** 2 + y0 ** 2 < rmin ** 2
        
        if x1 ** 2 + y1 ** 2 > 1 or mask_omegai  == 0:
            return -np.inf
        
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_brdf_hemispheri(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_phi = y0 < np.pi and y0 > -np.pi and y1 < np.pi and y1 > -np.pi
        mask_theta = x1 < np.pi/2 and x1 > 0 and x0 < rmax and x0 > rmin
        if ~mask_phi or ~mask_theta:
            return -np.inf
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_brdf_allspheri(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_phi = y0 < np.pi and y0 > -np.pi and y1 < np.pi and y1 > -np.pi
        mask_theta = x1 < np.pi and x1 > 0 and x0 < rmax and x0 > rmin
        if ~mask_phi or ~mask_theta:
            return -np.inf
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_bsdf(p,pdf_func):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask1 = (x0 -1) ** 2 + y0 ** 2 > 1 and (x0 + 1) ** 2 + y1 ** 2 > 1
        mask2 = (x1 -1) ** 2 + y1 ** 2 > 1 and (x1 + 1) ** 2 + y1 ** 2 > 1
        pdf_value = pdf_func(p)
        if mask1 or mask2 or pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def find_omegao(omegai, pdf_func,is_spherical = False):
    while True:
        if is_spherical:
            omegao = stratified_sampling_2d(1)
            omegao[:,0] = omegao[:,0] * np.pi / 2
            omegao[:,1] = omegao[:,1] * 2 * np.pi - np.pi
        else:
            omegao = stratified_sample_wo(1)
        p = np.concatenate([omegai,omegao],axis=1).reshape(1,4)
        pdf_value = pdf_func(p)
        if pdf_value != 0:
            break
    return omegao

def find_omegao_bsdf(omegai, pdf_func):
    while True:
        
        omegao = stratified_sampling_2d(1)
        omegao[:,0] = omegao[:,0] * np.pi 
        omegao[:,1] = omegao[:,1] * 2 * np.pi - np.pi
        p = np.concatenate([omegai,omegao],axis=1).reshape(1,4)
        pdf_value = pdf_func(p)
        if pdf_value != 0:
            break
    return omegao

def emcee_mcmc_brdf_disk(pdf_func, nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegao = stratified_sample_wo(nwalkers)
    omegao = omegao[np.random.choice(nwalkers, nwalkers, replace=False)]
    omegai_base = stratified_sample_wo(2**22)
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise
        radius_current_min = i / piecewise
        mask = torch.logical_and((omegai_base[:,0] ** 2 + omegai_base[:,1] ** 2) < radius_current_max ** 2,  
                                 (omegai_base[:,0] ** 2 + omegai_base[:,1] ** 2) > radius_current_min ** 2)
        omegai = omegai_base[mask]
        omegai = omegai[np.random.choice(omegai.shape[0], nwalkers, replace=False)]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao(omegai[i].reshape(1,2),pdf_func)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_disk,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

def emcee_mcmc_brdf_spherical(pdf_func,nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegai_base = stratified_sampling_2d(2**22)
    omegai_base[:,0] = omegai_base[:,0] * np.pi / 2
    omegai_base[:,1] = omegai_base[:,1] * 2 * np.pi - np.pi
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise * np.pi / 2
        radius_current_min = i / piecewise * np.pi / 2
        mask = torch.logical_and(omegai_base[:,0] < radius_current_max, omegai_base[:,0] > radius_current_min)
        omegai = omegai_base[mask]
        selected_indices = np.random.choice(omegai.shape[0], nwalkers, replace=False)
        omegai = omegai[selected_indices]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao(omegai[i].reshape(1,2),pdf_func,is_spherical=True)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim) 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_hemispheri,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

def emcee_mcmc_bsdf(pdf_func, nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegai_base = stratified_sampling_2d(2**22)
    omegai_base[:,0] = omegai_base[:,0] * np.pi 
    omegai_base[:,1] = omegai_base[:,1] * 2 * np.pi - np.pi
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise * np.pi 
        radius_current_min = i / piecewise * np.pi 
        mask = torch.logical_and(omegai_base[:,0] < radius_current_max, omegai_base[:,0] > radius_current_min)
        omegai = omegai_base[mask]
        
        selected_indices = np.random.choice(omegai.shape[0], nwalkers, replace=False)
        omegai = omegai[selected_indices]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao_bsdf(omegai[i].reshape(1,2),pdf_func)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim) 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_allspheri,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

# ---------------------------------------------------------------------------
# ヤーン繊維BSDF用のMCMCサンプリング関数
# ---------------------------------------------------------------------------
# 【既存の emcee_mcmc_bsdf との違い】
# 既存の emcee_mcmc_bsdf は表面BSDFを想定しており:
#   theta_i ∈ [0, π]    （z軸からの極角、上半球＋下半球）
#   theta_o ∈ [0, π]
#   phi     ∈ [-π, π]
#
# ヤーンBSDF（Khungurn座標系）では繊維軸がx軸であり:
#   theta_i ∈ [-π/2, π/2]  （繊維軸に垂直な面からの縦方向角）
#   theta_o ∈ [-π/2, π/2]
#   phi     ∈ [-π, π]       （繊維軸まわりの方位角、変わらず）
#
# このため、theta の範囲チェックと初期サンプル生成を変更する必要がある。
# ---------------------------------------------------------------------------

def find_omegao_yarn(omegai, pdf_func):
    """ヤーン座標系で BSDF 値が非ゼロになる wo を探す初期サンプラー。

    find_omegao_bsdf との違い:
      theta_o を [0, π] ではなく [-π/2, π/2] の範囲で生成する。
    """
    while True:
        omegao = stratified_sampling_2d(1)
        omegao[:,0] = omegao[:,0] * np.pi - np.pi / 2  # theta_o ∈ [-π/2, π/2]
        omegao[:,1] = omegao[:,1] * 2 * np.pi - np.pi   # phi_o   ∈ [-π,   π]
        p = np.concatenate([omegai, omegao], axis=1).reshape(1, 4)
        pdf_value = pdf_func(p)
        if pdf_value != 0:
            break
    return omegao


def lnprob_yarn(p, pdf_func, theta_max, theta_min):
    """ヤーン座標系の対数確率（emcee が内部で呼ぶ関数）。

    lnprob_brdf_allspheri との違い:
      theta の許容範囲を [-π/2, π/2] に変更。
      theta_i は piecewise 区間 [theta_min, theta_max] に制限。
      theta_o は [-π/2, π/2] 全体を許容。
    """
    p = p.reshape(-1, 4)
    x0, y0, x1, y1 = p[:,0], p[:,1], p[:,2], p[:,3]
    mask_phi   = y0 < np.pi and y0 > -np.pi and y1 < np.pi and y1 > -np.pi
    mask_theta = (x1 < np.pi/2 and x1 > -np.pi/2
                  and x0 < theta_max and x0 > theta_min)
    if ~mask_phi or ~mask_theta:
        return -np.inf
    pdf_value = pdf_func(p)
    if pdf_value == 0:
        return -np.inf
    return np.log(np.clip(pdf_value, 0, None))


def emcee_mcmc_yarn(pdf_func, nsteps, ndim=4, nwalkers=49, piecewise=10, burn_in=10000):
    """ヤーン繊維BSDF向けの MCMC サンプラー。

    emcee_mcmc_bsdf との対応関係:
      emcee_mcmc_bsdf : theta_i ∈ [0, π]      を piecewise 分割
      emcee_mcmc_yarn : theta_i ∈ [-π/2, π/2] を piecewise 分割
    分割の意味: theta_i の全空間を piecewise 個の区間に分けてから
    各区間で独立に MCMC を走らせることで、theta_i 空間を均等にカバーする。
    """
    omegai_base = stratified_sampling_2d(2**22)
    omegai_base[:,0] = omegai_base[:,0] * np.pi - np.pi / 2  # [-π/2, π/2]
    omegai_base[:,1] = omegai_base[:,1] * 2 * np.pi - np.pi   # [-π,   π]

    all_samples = []
    for i in range(piecewise):
        theta_current_max = (i + 1) / piecewise * np.pi - np.pi / 2
        theta_current_min =  i      / piecewise * np.pi - np.pi / 2

        mask = torch.logical_and(
            omegai_base[:,0] < theta_current_max,
            omegai_base[:,0] > theta_current_min,
        )
        omegai = omegai_base[mask]
        selected_indices = np.random.choice(omegai.shape[0], nwalkers, replace=False)
        omegai = omegai[selected_indices]

        omegao = []
        for j in range(omegai.shape[0]):
            omegao_j = find_omegao_yarn(omegai[j].reshape(1, 2), pdf_func)
            omegao.append(omegao_j)
        omegao = np.concatenate(omegao)

        p0 = np.concatenate([omegai, omegao], axis=1).reshape(nwalkers, ndim)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob_yarn,
                args=(pdf_func, theta_current_max, theta_current_min),
                pool=pool,
            )
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)

        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)

    return np.concatenate(all_samples)


if __name__ == "__main__":
    roughness = 0.2
    nwalkers = 49
    nsteps = 200000
    ndim = 4
    samples = emcee_mcmc_brdf(pdf_func_np, nwalkers, nsteps, ndim)
    np.save("brdf_samples.npy", samples)
    print(samples.shape)
