# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from tqdm import tqdm
import torch
from utils.model import *

from pathlib import Path
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p = str(SCRIPT_DIR)
sys.path.insert(1, p)
from utils.mitsuba_brdf_draw import *
from utils.mlp_brdf_sampling import *
from utils.analytical_brdf_torch import *

torch.set_default_dtype(torch.float32)
mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, default="disney_bsdf_array0_envmap.xml")
parser.add_argument("--passes", type=int, default=128)

parser = parser.parse_args()

def to_mi_float(tensor):
    return mi.Float(tensor.detach().cpu().numpy())

def as_points(tensor):
    if tensor.ndim == 2 and tensor.shape[0] == 3 and tensor.shape[1] != 3:
        return tensor.transpose(0, 1)
    return tensor

class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        material = props["filename"]
        measured_path = SCRIPT_DIR / "measuredbsdfs" / f"{material}.bsdf"
        if not measured_path.exists():
            raise FileNotFoundError(f"Measured BSDF file does not exist: {measured_path}")

        self.bsdf = meaturedbsdf(str(measured_path))
        self.bsdf = mi.load_dict(
            {
                "type":"measured",
                "filename": str(measured_path)
            }
        )
        self.D_sample = NN_cond_pos_simpler(input_dim=5,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to("cuda")
        self.D_sample_path = SCRIPT_DIR / "checkpoints_new" / f"{material}_disk" / f"brdf_rectify_network{material}.pth"
        if not self.D_sample_path.exists():
            raise FileNotFoundError(f"Sample checkpoint does not exist: {self.D_sample_path}")
        self.D_sample.load_state_dict(torch.load(str(self.D_sample_path)))

        self.D_sample.eval()
        
        self.D_base = NN_cond_pretrain_disk_one(input_dim=2,N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3).to("cuda")
        self.D_base_path = SCRIPT_DIR / "checkpoints_new" / f"{material}_disk" / f"brdf_pretrain_network{material}.pth"
        if not self.D_base_path.exists():
            raise FileNotFoundError(f"Base checkpoint does not exist: {self.D_base_path}")
        self.D_base.load_state_dict(torch.load(str(self.D_base_path)))
        
        
        self.albedo = mi.Color3f([1, 1,1]) 
        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        wi = as_points(si.wi.torch())
        wi_input = wi[...,:2]
        wo,pdf = network_sampling_disk(self.D_base,self.D_sample,wi_input)
        valid = torch.square(wo[...,0]) + torch.square(wo[...,1]) < 0.995
        wo[~valid] = torch.tensor([0.0, 0.0], device=device)
        pdf[~valid] = 0.0
        
        wo_dr = disk_to_cart(wo)
        
        wo = mi.Vector3f(to_mi_float(wo_dr[...,0]), to_mi_float(wo_dr[...,1]), to_mi_float(wo_dr[...,2]))
        bs = mi.BSDFSample3f()
        
        
        
        bs.wo = wo           
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        bs.pdf = to_mi_float(pdf) * cos_theta_o
        # bs.wo = mi.warp.square_to_cosine_hebrdf_onlyphere(sample2)
        # bs.pdf = mi.warp.square_to_cosine_hebrdf_onlyphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0
        
        wi_input = as_points(si.wi.torch())[...,:2]
        wo_tmp = as_points(bs.wo.torch())[...,:2]
        #brdf = self.bsdf.eval(wi_input, wo_tmp)
        brdf = self.bsdf.eval(ctx, si, bs.wo)
        value = brdf / bs.pdf * self.albedo   

        # if dr.any_nested(dr.isnan(bs.wo)):
        #     print("nan pdf"
        value_torch = as_points(rgb2lum(value).torch())
        pdf = torch.where(value_torch<30, pdf, torch.zeros_like(pdf))

        bs.pdf = to_mi_float(pdf) * cos_theta_o
        return (bs, dr.select(active & (bs.pdf > 0.0) & (cos_theta_o > 0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        brdf = self.bsdf.eval(ctx, si, wo)
        value =  brdf * self.albedo 
        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0)
        )

    def pdf(self, ctx, si, wo, active=True):

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        wi = as_points(si.wi.torch())
        wi_input = wi[...,:2]
        wo = as_points(wo.torch())
        wo_input = wo[...,:2]
        pdf = network_pdf_disk(self.D_base,self.D_sample,wo_input,wi_input) 
        
        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), to_mi_float(pdf)* cos_theta_o, mi.Float(0)
        )

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    
    import time

    start_time = time.time()

    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

    scene_file = parser.scene_file
    if not scene_file.lower().endswith(".xml"):
        scene_file += ".xml"

    scene_path = SCRIPT_DIR / "matpreview" / scene_file
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_path}")

    scene = mi.load_file(str(scene_path))
    # print(params)
    SPP = 4
    spp = SPP * parser.passes

    seed = 0
    
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1

    output_dir = SCRIPT_DIR / "diffusion_brdf_measured_disk"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{parser.scene_file}.png"
    mi.util.write_bitmap(str(filepath), image)
    
    filepath_exr = output_dir / f"{parser.scene_file}.exr"
    mi.util.write_bitmap(str(filepath_exr), image)
    end_time = time.time()
    print("Render time: " + str(end_time - start_time) + " seconds")
    
    
