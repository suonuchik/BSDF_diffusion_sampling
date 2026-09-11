import torch
from tqdm import tqdm
from utils.utils import *
from utils.model import *
from utils.distribution import *
from utils.mitsuba_brdf_scalar import meaturedbsdf as meaturedbsdf_scalar
from utils.emcee_sampling import *
from utils.mitsuba_brdf_yarn import khungurn_bsdf
import numpy as np
import argparse
from utils.bsdf_dict import *



torch.set_default_dtype(torch.float32)

def pretrain_stage(args,brdf_samples,save_dir):
    Ndata = brdf_samples.shape[0] 
    pretrain_network = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3).to("cuda")
    optimizer_pretrain = torch.optim.Adam(pretrain_network.parameters(), lr=0.0003)
    pbar = tqdm(total=args.num_epochs_pretrain)
    print("Start training: pretrain")
    for iteration in (range(args.num_epochs_pretrain)):
        x_1 = brdf_samples[np.random.randint(0,Ndata,args.batchsize_pretrain),:]
        omega_o = x_1[:,2:4]
        omega_i = x_1[:,0:2]
        logp = pretrain_network.log_prob(omega_o,omega_i)    
        loss = -torch.mean(logp)
        loss.backward()
        optimizer_pretrain.step()
        if iteration % args.show_iter == 0:
            pbar.set_description(f"Loss {loss.item():.10f}")
            pbar.update(args.show_iter)   
        if iteration % args.save_iter == 0:
            save_model(pretrain_network,save_dir,"brdf_pretrain_network" + str(args.idx))
        pretrain_network.zero_grad()
    pbar.close()
    print("Finish training: pretrain")

def diffusion_stage(args,brdf_samples,save_dir):
    Ndata = brdf_samples.shape[0]
    pretrain_network = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3).to("cuda")
    pretrain_network.load_state_dict(torch.load(os.path.join(save_dir,"brdf_pretrain_network" + str(args.idx) + ".pth")))
    twopi = np.pi * 2

    simpler_ckpt_path = os.path.join(save_dir, "brdf_diffusion_network_simpler" + str(args.idx) + ".pth")
    complex_ckpt_path = os.path.join(save_dir, "brdf_diffusion_network_complex" + str(args.idx) + ".pth")

    # ---- simpler ----
    if os.path.exists(simpler_ckpt_path):
        print(f"Skipping diffusion simpler: checkpoint found ({simpler_ckpt_path})")
    else:
        diffusion_network_simpler = NN_cond_pos(input_dim=6,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to("cuda")
        optimizer_diffusion_simpler = torch.optim.Adam(diffusion_network_simpler.parameters(), 0.001)
        pbar = tqdm(total=args.num_epochs_diffusion)
        print("Start training: diffusion simpler")
        for iteration in (range(args.num_epochs_diffusion)):
            x_1 = brdf_samples[np.random.randint(0,Ndata,args.batchsize_diffusion),:]
            omega_o = x_1[:,2:4]
            omega_i = x_1[:,0:2]
            with torch.no_grad():
                x_0 = pretrain_network.sample(omega_i,args.batchsize_diffusion)
            alpha = torch.linspace(0,1,args.batchsize_diffusion).to("cuda")
            alpha = alpha.reshape(-1,1)
            tmp_dirc = omega_o[:,1] - x_0[:,1]
            omega_o[:,1] = torch.where(tmp_dirc < - np.pi,omega_o[:,1] + twopi , torch.where(tmp_dirc > np.pi,omega_o[:,1] - twopi,omega_o[:,1]))
            x_alpha = (1 - alpha) * x_0 + alpha * omega_o
            x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
            x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
            pred = diffusion_network_simpler(x_alpha_2d, alpha, omega_i)
            thetadirc = omega_o[:,0] - x_0[:,0]
            phidirc = torch.where(tmp_dirc < - np.pi,tmp_dirc + twopi , torch.where(tmp_dirc > np.pi,tmp_dirc - twopi,tmp_dirc))
            twod_dirc = torch.cat([thetadirc.reshape(-1,1),phidirc.reshape(-1,1)],dim=1)
            loss = torch.mean((pred -  (twod_dirc)) ** 2)
            loss.backward()
            optimizer_diffusion_simpler.step()
            if iteration % args.show_iter == 0:
                pbar.set_description(f"Loss {loss.item():.10f}")
                pbar.update(args.show_iter)
            if iteration % args.save_iter == 0:
                save_model(diffusion_network_simpler,save_dir,"brdf_diffusion_network_simpler" + str(args.idx))
            optimizer_diffusion_simpler.zero_grad()
        pbar.close()
        print("Finish training: diffusion simpler")

    # ---- complex ----
    # complex ネットワーク（6層×64ニューロン）は batchsize_diffusion が大きすぎると
    # GPU メモリ（backward の中間テンソルが ~8GB）を超えて極端に遅くなる。
    # simpler の 1/10 程度（args.batchsize_diffusion_complex）を推奨。
    if os.path.exists(complex_ckpt_path):
        print(f"Skipping diffusion complex: checkpoint found ({complex_ckpt_path})")
    else:
        batchsize_complex = args.batchsize_diffusion_complex
        diffusion_network_complex = NN_cond_pos_spherical_complicate(input_dim=6,output_dim=2,N_NEURONS=64,POSITIONAL_ENCODING_BASIS_NUM=5).to("cuda")
        optimizer_diffusion_complex = torch.optim.Adam(diffusion_network_complex.parameters(), 0.001)
        pbar = tqdm(total=args.num_epochs_diffusion)
        print(f"Start training: diffusion complex (batchsize={batchsize_complex})")
        for iteration in (range(args.num_epochs_diffusion)):
            x_1 = brdf_samples[np.random.randint(0,Ndata,batchsize_complex),:]
            omega_o = x_1[:,2:4]
            omega_i = x_1[:,0:2]
            with torch.no_grad():
                x_0 = pretrain_network.sample(omega_i,batchsize_complex)
            alpha = torch.linspace(0,1,batchsize_complex).to("cuda")
            alpha = alpha.reshape(-1,1)
            tmp_dirc = omega_o[:,1] - x_0[:,1]
            omega_o[:,1] = torch.where(tmp_dirc < - np.pi,omega_o[:,1] + twopi , torch.where(tmp_dirc > np.pi,omega_o[:,1] - twopi,omega_o[:,1]))
            x_alpha = (1 - alpha) * x_0 + alpha * omega_o
            x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
            x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
            pred = diffusion_network_complex(x_alpha_2d, alpha, omega_i)
            thetadirc = omega_o[:,0] - x_0[:,0]
            phidirc = torch.where(tmp_dirc < - np.pi,tmp_dirc + twopi , torch.where(tmp_dirc > np.pi,tmp_dirc - twopi,tmp_dirc))
            twod_dirc = torch.cat([thetadirc.reshape(-1,1),phidirc.reshape(-1,1)],dim=1)
            loss = torch.mean((pred -  (twod_dirc)) ** 2)
            loss.backward()
            optimizer_diffusion_complex.step()
            if iteration % args.show_iter == 0:
                pbar.set_description(f"Loss {loss.item():.10f}")
                pbar.update(args.show_iter)
            if iteration % args.save_iter == 0:
                save_model(diffusion_network_complex,save_dir,"brdf_diffusion_network_complex" + str(args.idx))
            optimizer_diffusion_complex.zero_grad()
        pbar.close()
        print("Finish training: diffusion complex")
    
def rectify_stage(args,save_dir,is_yarn=False):
    # tinycudann は高速推論のためのオプション依存。なければ同等の PyTorch モデルで代替する。
    try:
        import tinycudann
        _has_tinycudann = True
    except ImportError:
        _has_tinycudann = False
        print("tinycudann not found — falling back to PyTorch model (NN_cond_pos_spherical_complicate)")

    print("Start training: rectify")
    pretrain_network = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3).to("cuda")
    pretrain_network.load_state_dict(torch.load(os.path.join(save_dir,"brdf_pretrain_network" + str(args.idx) + ".pth")))
    diffusion_network = NN_cond_pos(input_dim=6,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to("cuda")
    diffusion_pytorch_weights = torch.load(os.path.join(save_dir,"brdf_diffusion_network_simpler" + str(args.idx) + ".pth"))
    diffusion_network.load_state_dict(diffusion_pytorch_weights)

    diffusion_pytorch_weights = torch.load(os.path.join(save_dir,"brdf_diffusion_network_complex" + str(args.idx) + ".pth"))
    if _has_tinycudann:
        # tinycudann の FullyFusedMLP は PyTorch 版 NN_cond_pos_spherical_complicate の
        # 重みをロードして GPU 上で高速推論するためのラッパー
        rectify_temp_net = tinycudann.Network(
            n_input_dims=26,
            n_output_dims=2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "SiLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 6
            }
        )
        load_pytorch_model_to_tinycuda(rectify_temp_net,diffusion_pytorch_weights,26,2)
    else:
        # tinycudann がない場合: 同じ重みを PyTorch モデルに直接ロードする
        # 推論速度は落ちるが数値的に同等の結果が得られる
        rectify_temp_net = NN_cond_pos_spherical_complicate(
            input_dim=6,output_dim=2,N_NEURONS=64,POSITIONAL_ENCODING_BASIS_NUM=5
        ).to("cuda")
        rectify_temp_net.load_state_dict(diffusion_pytorch_weights)
    rectify_temp_net.eval()

    T = args.timestep_rectify

    def dosampling(batchsize,omega_i,T):

        x_target_y = omega_i.repeat_interleave(batchsize,0)
        with torch.no_grad():
            x_alpha = pretrain_network.sample(x_target_y,batchsize * len(omega_i))
        x_base_samples = x_alpha.clone()

        x_target_y_tmp = positional_encoding_1(x_target_y, 5)
        ones = torch.ones(batchsize * len(omega_i),1,device='cuda')

        with torch.no_grad():
            for t in (range(T)):
                alpha = t / T * ones
                x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
                x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
                if _has_tinycudann:
                    # tinycudann はフラットな (N, 26) テンソルを受け取る
                    x_input = torch.cat([x_alpha_2d,alpha,x_target_y_tmp],dim = 1)
                    d_output = rectify_temp_net(x_input)
                else:
                    # PyTorch 版は (x, alpha, x_co) を別引数で受け取り、内部で位置符号化する
                    d_output = rectify_temp_net(x_alpha_2d, alpha, x_target_y)
                x_alpha = x_alpha + 1 / T * d_output

        return x_alpha,x_base_samples,x_target_y

    optimizer_D = torch.optim.Adam(diffusion_network.parameters(), lr=0.001)
    twopi = np.pi * 2
    pbar = tqdm(total=args.num_epochs_rectify)
    for iteration in (range(1,args.num_epochs_rectify)):
        
        omega_i_samples = stratified_sampling_2d(args.batchsize_rectify).cuda()
        if is_yarn:
            # ヤーン座標系: theta_i ∈ [-π/2, π/2]（繊維軸からの縦方向角）
            omega_i_samples[:,0] = omega_i_samples[:,0] * np.pi - np.pi / 2
        else:
            # 表面BSDF座標系: theta_i ∈ [0, π]（法線からの極角）
            omega_i_samples[:,0] = omega_i_samples[:,0] * np.pi
        omega_i_samples[:,1] = omega_i_samples[:,1] * 2 * np.pi - np.pi
        
        x_1,x_0,x_target_y = dosampling(args.num_samples_rectify,omega_i_samples,T)
        indices = torch.randperm(len(x_1))
        x_0 = x_0[indices, :]
        x_1 = x_1[indices, :]
        x_target_y = x_target_y[indices, :]
        omega_o = x_1
        omega_i = x_target_y
        alpha = torch.linspace(0,1,args.num_samples_rectify*args.batchsize_rectify,device='cuda').reshape(-1,1)
        tmp_dirc = omega_o[:,1] - x_0[:,1]
        omega_o[:,1] = torch.where(tmp_dirc < - np.pi,omega_o[:,1] + twopi , torch.where(tmp_dirc > np.pi,omega_o[:,1] - twopi,omega_o[:,1]))
        x_alpha = (1 - alpha) * x_0 + alpha * omega_o
        x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
        x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
        pred = diffusion_network(x_alpha_2d, alpha, omega_i)
        thetadirc = omega_o[:,0] - x_0[:,0]
        phidirc = torch.where(tmp_dirc < - np.pi,tmp_dirc + twopi , torch.where(tmp_dirc > np.pi,tmp_dirc - twopi,tmp_dirc))
        twod_dirc = torch.cat([thetadirc.reshape(-1,1),phidirc.reshape(-1,1)],dim=1)
        loss = torch.mean((pred -  (twod_dirc)) ** 2)
        loss.backward()
        optimizer_D.step()
        if iteration % args.show_iter == 0:
            pbar.set_description(f"Loss {loss.item():.10f}")
            pbar.update(args.show_iter)   
        if iteration % args.save_iter == 0:
            save_model(diffusion_network,save_dir,"brdf_rectify_network" + str(args.idx))
        optimizer_D.zero_grad()
    pbar.close()
    
    print("Finish training: rectify")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batchsize_pretrain", default = "4900000 * 2",type=eval_arg)
    parser.add_argument("--num_epochs_pretrain", default = "10000",type=eval_arg)
    
    parser.add_argument("--num_epochs_diffusion", default = "40000",type=eval_arg)
    parser.add_argument("--batchsize_diffusion", default = "4900000",type=eval_arg)
    # complex ネットワークは層が深くて backward の中間テンソルが大きいため、
    # simpler より小さいバッチサイズを指定する（デフォルト 500000 ≈ 1/10）
    parser.add_argument("--batchsize_diffusion_complex", default = "500000",type=eval_arg)

    parser.add_argument("--num_epochs_rectify", default = "40000",type=eval_arg)
    # tinycudann なし環境向けに小さな値を設定。
    # tinycudann あり環境では timestep_rectify=128, num_samples_rectify=2**16, batchsize_rectify=2**6 が推奨。
    parser.add_argument("--timestep_rectify", default = 8, type=int)
    parser.add_argument("--num_samples_rectify", default = "2**10",type=eval_arg)
    parser.add_argument("--batchsize_rectify", default = "2**4",type=eval_arg)
    
    parser.add_argument("--save_iter", default = "100",type=eval_arg)
    parser.add_argument("--show_iter", default = "10",type=eval_arg)
    parser.add_argument("--save_dir", default = "./checkpoints_new",type=str)
    parser.add_argument("--base_dir", default = "./measuredbsdfs",type=str)
    parser.add_argument("--idx", default = "9",type=int)
    parser.add_argument("--is_rectify", default = False,type=bool)
    # --is_yarn: ヤーン繊維BSDF（idx=26〜30）を学習するときに指定する
    # 指定すると theta の座標系が [-π/2, π/2] に切り替わり、
    # ヤーン専用のMCMCサンプラー（emcee_mcmc_yarn）が使われる
    parser.add_argument("--is_yarn", default = False, type=eval_arg)
    # --mcmc_only: MCMCサンプリングだけ実行してnpyを保存し、学習はスキップする
    # GPU不要・tinycudann不要でサンプルだけ先に作りたい場合に使う
    parser.add_argument("--mcmc_only", default = False, type=eval_arg)
    args = parser.parse_args()

    mybsdf_scalar = bsdf_materials[args.idx]

    # ヤーン素材かどうかをクラス型でも自動判定できるようにする
    is_yarn = args.is_yarn or isinstance(mybsdf_scalar, khungurn_bsdf)

    # チェックポイントの保存先フォルダ名をBSDFの種類で分ける
    # 例: bsdf_29_yarn（cotton）、bsdf_9_spherical（既存の表面BSDF）
    suffix = "yarn" if is_yarn else "spherical"
    prefix = "bsdf_" + str(args.idx) + "_" + suffix
    save_dir = os.path.join(args.save_dir, prefix)
    file_path = os.path.join(save_dir, "brdf_samples_emcee" + str(args.idx) + ".npy")

    def pdf_func(x):
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 4)
        r = mybsdf_scalar.eval(x[:,0:2], x[:,2:4])
        # ヤーンBSDFはPyTorchテンソルを返すためfloatに変換する
        # 既存の表面BSDFはdrjitスカラーを返すため変換不要だが、
        # float()はどちらにも適用できるため共通化している
        return float(r)

    if args.is_rectify:
        rectify_stage(args, save_dir, is_yarn=is_yarn)
    else:
        if os.path.exists(file_path):
            brdf_samples = np.load(file_path)
            print(f"Loaded existing samples: {file_path} ({brdf_samples.shape})")
        else:
            # ヤーン素材かどうかでMCMCサンプラーを切り替える
            if is_yarn:
                brdf_samples = emcee_mcmc_yarn(pdf_func, 40000, burn_in=10000)
            else:
                brdf_samples = emcee_mcmc_bsdf(pdf_func, 40000, burn_in=10000)
            os.makedirs(os.path.join(args.save_dir, prefix), exist_ok=True)
            np.save(os.path.join(args.save_dir, prefix,
                                 "brdf_samples_emcee" + str(args.idx) + ".npy"),
                    brdf_samples)
            print(f"Saved MCMC samples: {file_path} ({brdf_samples.shape})")

        # --mcmc_only が指定された場合はサンプル生成だけで終了する
        # （GPUなし環境や tinycudann が未インストールの環境向け）
        if args.mcmc_only:
            print("mcmc_only mode: skipping training stages.")
            exit(0)

        brdf_samples = torch.from_numpy(brdf_samples).to("cuda").type(torch.float32)

        # checkpoint が既に存在する場合はそのステージをスキップする
        # diffusion_stage 内部でも simpler/complex 個別にスキップする
        pretrain_ckpt = os.path.join(save_dir, "brdf_pretrain_network" + str(args.idx) + ".pth")

        if os.path.exists(pretrain_ckpt):
            print(f"Skipping pretrain: checkpoint found ({pretrain_ckpt})")
        else:
            pretrain_stage(args, brdf_samples, save_dir)

        diffusion_stage(args, brdf_samples, save_dir)
    
