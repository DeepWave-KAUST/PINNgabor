from utils.encoding import get_embedder
from utils.utils import *
from utils.vis import plot
from utils.TransferData2Dataset import DataProcessing
from model_zoo.pinnmodel import PINNmodel
from model_zoo import build_model
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import scipy.io as sio


def train(cfg, net, embedding_fn, train_data):
    device = torch.device(cfg.pinn.device)
    seed_torch(cfg.pinn.seed)
    
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.pinn.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.pinn.step_size, gamma=cfg.pinn.gamma)
    min_loss = 9e20
    sw = SummaryWriter(os.path.join(cfg.tensorboard_path))
    
    ub, lb = torch.Tensor([cfg.pinn.ub]).to(device), torch.Tensor([cfg.pinn.lb]).to(device)
    fdm_loss_pde = helmholtz_loss_pde(0.001,0.001,ub, lb, cfg.pinn.regularization, cfg.pinn.pde_loss_penelty, v_background=cfg.pinn.regular_v, device=device)
    
    ##############################################
    
    for epoch in range(cfg.pinn.epochs):
        epoch_loss, pde_loss, reg_loss = 0, 0, 0
        input_data = torch.Tensor(train_data.data[:,:])
        randperm = np.random.permutation(len(input_data))
        batch_size = int(len(input_data)/cfg.pinn.n_batches)
        for batch_idx in range(cfg.pinn.n_batches):
            start_,end_ = batch_idx*batch_size,(batch_idx+1)*batch_size
            randperm_idx = randperm[start_:end_]
            x_train, y_train, sx_train, u0_real_train, u0_imag_train, m_train, m0_train = input_data[randperm_idx,0:1].to(device),input_data[randperm_idx,1:2].to(device),input_data[randperm_idx,2:3].to(device),input_data[randperm_idx,3:4].to(device),input_data[randperm_idx,4:5].to(device),input_data[randperm_idx,5:6].to(device),input_data[randperm_idx,6:7].to(device)
            
            optimizer.zero_grad()
            x = x_train.clone().detach().requires_grad_(True)
            y = y_train.clone().detach().requires_grad_(True)
            sx = sx_train.clone().detach().requires_grad_(True)
            omega = 2 * torch.pi * cfg.pinn.fre
            x_input = torch.cat((x, y, sx), 1)
            x_input = normalizer(x_input, lb, ub)
            x_input = embedding_fn(x_input)
            du_pred_out = net(x_input)
            if cfg.pinn.regularization != 'None':
                loss, pde_loss, reg_loss = fdm_loss_pde(x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train,du_pred_out, net, embedding_fn, derivate_type=cfg.pinn.derivate_type)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().item()
                pde_loss += pde_loss.detach().cpu().item()
                reg_loss += reg_loss.detach().cpu().item()
            else:
                loss = fdm_loss_pde(x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train,du_pred_out, net, embedding_fn, derivate_type=cfg.pinn.derivate_type)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().item()
                
        scheduler.step()
        sw.add_scalar('train/loss', epoch_loss, epoch)
        sw.add_scalar('train/pdeloss', pde_loss, epoch)
        sw.add_scalar('train/regloss', reg_loss, epoch)
        if epoch % cfg.pinn.print_loss_every == 0:
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                if not os.path.exists(cfg.checkpoint_path):
                    os.makedirs(cfg.checkpoint_path)
                if cfg.pinn.encoding_config.encoding_type==1:
                    torch.save({'net':net.state_dict(),'emb':embedding_fn}, os.path.join(cfg.checkpoint_path, 'best_net.pth'))
                else:
                    torch.save({'net':net.state_dict()}, os.path.join(cfg.checkpoint_path, 'best_net.pth'))
            print(f'Epoch: {epoch}; Loss: {epoch_loss}; Equ Loss: {pde_loss}; Reg Loss: {reg_loss}')
        
        if epoch !=0 and epoch % cfg.pinn.save_model_every == 0:
            if cfg.pinn.encoding_config.encoding_type==1:
                torch.save({'net':net.state_dict(),'emb':embedding_fn}, os.path.join(cfg.checkpoint_path, f'net_{epoch}.pth'))
            else:
                torch.save({'net':net.state_dict()}, os.path.join(cfg.checkpoint_path, f'net_{epoch}.pth'))
            
        if epoch !=0 and epoch % cfg.pinn.test_every == 0:
            du_real_eval, du_imag_eval = test_ent(cfg, True, out_vis=True)
            sw.add_figure(f'Wavefield/real{cfg.pinn.fre}',plot(du_real_eval, cfg.pinn.vmin, cfg.pinn.vmax, cfg.pinn.axisx, cfg.pinn.axisz),epoch)
            sw.add_figure(f'Wavefield/imag{cfg.pinn.fre}',plot(du_imag_eval, cfg.pinn.vmin, cfg.pinn.vmax, cfg.pinn.axisx, cfg.pinn.axisz),epoch)
  
def test(net, embedding_fn, out_vis, cfg):
    device = torch.device(cfg.pinn.device)
    nx, nz = cfg.pinn.nx, cfg.pinn.nz
    x = 2.0 * (np.arange(nx).reshape(nx,1).repeat(nz,axis=0).reshape(nx*nz,1)*0.025/cfg.pinn.fine_ratio - 0.0) / ((nx-1)*0.025/cfg.pinn.fine_ratio - 0.0) - 1.0 #+ 0.005
    # x = 2.0 * (np.arange(nx).reshape(nx,1).repeat(nz,axis=0).reshape(nx*nz,1)*0.025/cfg.pinn.fine_ratio - 0.0) / ((nx-1)*0.025/cfg.pinn.fine_ratio - 0.0) - 1.0 + 0.005 # sensitity analysis
    y = 2.0 * (np.arange(nz).reshape(nz,1).repeat(nx,axis=1).T.reshape(nx*nz,1)*0.025/cfg.pinn.fine_ratio - 0.0) / ((nz-1)*0.025/cfg.pinn.fine_ratio - 0.0) - 1.0 #+ 0.005
    # y = 2.0 * (np.arange(nz).reshape(nz,1).repeat(nx,axis=1).T.reshape(nx*nz,1)*0.025/cfg.pinn.fine_ratio - 0.0) / ((nz-1)*0.025/cfg.pinn.fine_ratio - 0.0) - 1.0 + 0.005 # sensitity analysis
    sx = np.zeros_like(x)
    x_input = embedding_fn(torch.cat(((torch.Tensor(x)).to(device),(torch.Tensor(y)).to(device),(torch.Tensor(sx)).to(device)),1))
    du_pred_out = net(x_input.to(device))
    du_real_eval, du_imag_eval = du_pred_out[:,0:1], du_pred_out[:,1:2]
    if not out_vis:
        print('test done, current no ground truth')
        torch.save({'pred_real':du_real_eval.detach().cpu().numpy().reshape(nx,nz), 'pred_imag': du_imag_eval.detach().cpu().numpy().reshape(nx,nz)}, f'test_{cfg.net_name}_{cfg.pinn.scale}_{cfg.pinn.hidden_layers}_{cfg.pinn.fre}_{cfg.pinn.state_dict_file[-10:-5]}.pt')
    else:
        return du_real_eval.detach().cpu().numpy().reshape(nx,nz), du_imag_eval.detach().cpu().numpy().reshape(nx,nz)

def train_ent(cfg):
    embedding_fn, input_cha = get_embedder(cfg.pinn.encoding_config)
    if cfg.pinn.model_type == 'pinn':
        net = build_model(cfg.pinn.model_type, cfg.net_name, input_cha, cfg.pinn.out_channels, hidden_layers=cfg.pinn.hidden_layers, input_scale=cfg.pinn.scale, n_layers=len(cfg.pinn.hidden_layers))
    
    macs, params = get_model_complexity_info(net, (input_cha,), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
    ub, lb = np.array([cfg.pinn.ub]), np.array([cfg.pinn.lb])
    train_data = DataProcessing(cfg.pinn.data_root_path,'{}_{}Hz_train_data.mat'.format(cfg.pinn.model_name,cfg.pinn.fre),'train',ub,lb)
    if cfg.pinn.state_dict_file != 'None':
        state = torch.load(cfg.pinn.state_dict_file)
        net.load_state_dict(state['net'])
    train(cfg, net, embedding_fn, train_data)

def test_ent(cfg, is_train=False, out_vis=False):
    embedding_fn, input_cha = get_embedder(cfg.pinn.encoding_config)
    if cfg.pinn.model_type == 'pinn':
        net = build_model(cfg.pinn.model_type, cfg.net_name, input_cha, cfg.pinn.out_channels, hidden_layers=cfg.pinn.hidden_layers, input_scale=cfg.pinn.scale, n_layers=len(cfg.pinn.hidden_layers))
    
    if is_train:
        state = torch.load(os.path.join(cfg.checkpoint_path, 'best_net.pth'))
    else:
        state = torch.load(cfg.pinn.state_dict_file)
    state_dict = state['net']
    if cfg.pinn.encoding_config.encoding_type==1:
        embedding_fn.load_state_dict(state['emb'])
    net.load_state_dict(state_dict)
    net.to(cfg.pinn.device)
    if not out_vis:
        test(net, embedding_fn, out_vis, cfg)
    else:
        return test(net, embedding_fn, out_vis, cfg)
        
def pinn_ent(cfg):
    if cfg.task_type == 'train':
        train_ent(cfg)
    else:
        test_ent(cfg)
