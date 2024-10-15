import datetime
import os
from torch.utils.data import TensorDataset, DataLoader
from utils.config import args
from utils.UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
import argparse
from utils.preprocess import POI_traj_Dataset


# This code part from https://github.com/sunlin-ai/diffusion_tutorial


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def main(config, logger, exp_dir, args):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    if args.cuda == 'cpu':
        # temporal_unet = Temporal_UNet(config)
        spatial_unet = Spatial_UNet(config)
    else:
        # temporal_unet = Temporal_UNet(config).cuda()
        spatial_unet = Spatial_UNet(config).cuda()
    # print(unet)
    # traj = np.load(args.traj_path,
    #                allow_pickle=True)
    # traj = traj[:, :, :2]
    # head = np.load(args.head_path,
    #                allow_pickle=True)
    # traj = np.swapaxes(traj, 1, 2)
    # traj = torch.from_numpy(traj).float()
    # head = torch.from_numpy(head).float()

    # Load data
    data = POI_traj_Dataset(args.traj_path, args.gps_path, args.cat_path)

    traj = data.inter_trajectories
    gps_traj = np.swapaxes(data.gps_trajectories, 1, 2)
    mask_traj = data.masks

    # add a dimension to the end of the tensor
    traj = torch.from_numpy(traj).float()
    gps_traj = torch.from_numpy(gps_traj).float()
    mask_traj = torch.from_numpy(mask_traj).float()
    # add a dimension to the end of the tensor from (N,  336) to (N, 336, 1)

    mask_dataset = TensorDataset(mask_traj, gps_traj)
    mask_dataloader = DataLoader(mask_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    if args.cuda == 'cpu':
        beta = torch.linspace(config.diffusion.beta_start,
                              config.diffusion.beta_end, n_steps)
    else:
        beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    # beta = torch.linspace(config.diffusion.beta_start,
    #                       config.diffusion.beta_end, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    t_optim = torch.optim.AdamW(spatial_unet.parameters(), lr=lr)  # Optimizer


    # new filefold for save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, (mask_traj, gps_traj) in enumerate(mask_dataloader):
            if args.cuda == 'cpu':
                x0 = gps_traj
                attr = mask_traj
                t = torch.randint(low=0, high=n_steps,
                                  size=(len(x0) // 2 + 1,))
            else:
                x0 = gps_traj.cuda()
                attr = mask_traj.cuda()
                t = torch.randint(low=0, high=n_steps,
                                  size=(len(x0) // 2 + 1,)).cuda()


            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = spatial_unet(xt.float(), t, attr)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            t_optim.zero_grad()
            loss.backward()
            t_optim.step()
        if (epoch) % 50 == 0:
            m_path = model_save / f"unet_temporal_{epoch}.pt"
            torch.save(spatial_unet.state_dict(), m_path)
            m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    # add an ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NYC')
    parser.add_argument('--traj_path', type=str, default='data/NYC/train_set.csv')
    parser.add_argument('--gps_path', type=str, default='data/NYC/gps')
    parser.add_argument('--cat_path', type=str, default='data/NYC/category')
    parser.add_argument('--cuda', type=str, default='cpu')
    args = parser.parse_args()


    root_dir = Path(__name__).resolve().parents[0]
    result_name = 'Spatial_{}_steps={}_len={}_{}_bs={}'.format(
        args.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    exp_dir = root_dir / "DiffTraj" / result_name
    for d in ["results", "models", "logs","Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    main(config, logger, exp_dir, args)
