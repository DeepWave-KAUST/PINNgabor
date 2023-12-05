#!/bin/sh

cd ../pinngabor/
tb_root=<your/absolute/path/to>/PINNgabor/exp/tensorboard/
run_root=<your/absolute/path/to>/PINNgabor/exp/
data_root=<your/absolute/path/to>/PINNgabor/data/

# training for 4hz
nohup python run.py pinn.regular_v=1.5 pinn.regularization=15.0 pinn.model_type=pinn net_name=gabor pinn.hidden_layers=[256,256,256] pinn.device=cuda:1 pinn.seed=9999 pinn.save_model_every=100 pinn.add_factor=-1.0 pinn.step_size=3000 pinn.gamma=0.1 tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root} >gabor.txt 2>&1 &  
nohup python run.py pinn.regular_v=1.5 pinn.regularization=15.0 pinn.model_type=pinn net_name=mlp pinn.hidden_layers=[256,256,256] pinn.device=cuda:3 pinn.seed=9999 pinn.save_model_every=100 pinn.add_factor=-1.0 tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root} >mlp_small.txt 2>&1 & 
nohup python run.py pinn.regular_v=1.5 pinn.regularization=15.0 pinn.model_type=pinn net_name=mlp pinn.hidden_layers=[512,512,512] pinn.device=cuda:3 pinn.seed=9999 pinn.save_model_every=100 pinn.add_factor=-1.0  tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root} >mlp_large.txt 2>&1 & 


# training for 16 hz
nohup python run.py pinn.regular_v=6.0 pinn.regularization=15.0 net_name=mlp pinn.model_type='pinn' pinn.fre=16.0 pinn.pre_train_fre=16.0 pinn.device='cuda:0' pinn.n_batches=16 pinn.scale=128.0 pinn.add_factor=-1.0 pinn.pde_loss_penelty=-1.0 pinn.epochs=50000 pinn.hidden_layers=[256,256,256] pinn.nx=201 pinn.nz=201 pinn.fine_ratio=2 pinn.add_number=0 pinn.seed=3407 tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root} >mlp_16hz.txt 2>&1 &
nohup python run.py pinn.regular_v=6.0 pinn.regularization=15.0 net_name=mlp pinn.model_type='pinn' pinn.fre=16.0 pinn.pre_train_fre=16.0 pinn.device='cuda:2' pinn.n_batches=16 pinn.scale=128.0 pinn.add_factor=-1.0 pinn.pde_loss_penelty=-1.0 pinn.hidden_layers=[512,512,512] pinn.nx=201 pinn.nz=201 pinn.fine_ratio=2 pinn.epochs=50000 pinn.add_number=0 pinn.seed=3407 tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root}  >mlp_16hz_large.txt 2>&1 &
nohup python run.py pinn.regular_v=6.0 pinn.regularization=15.0 net_name=gabor pinn.model_type='pinn' pinn.fre=16.0 pinn.pre_train_fre=16.0 pinn.device='cuda:3' pinn.n_batches=16 pinn.scale=128.0 pinn.add_factor=-1.0 pinn.pde_loss_penelty=-1.0 pinn.hidden_layers=[256,256,256] pinn.nx=201 pinn.nz=201 pinn.fine_ratio=2 pinn.add_number=0 tensorboard_root=${tb_root} hydra.run.dir=${run_root} pinn.data_root_path=${data_root} >gabor_16hz.txt 2>&1 & 

