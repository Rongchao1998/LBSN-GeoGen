import os
from tqdm.notebook import tqdm
from utils.UNet import *
from utils.config import args
from utils.utils import *
import argparse
from utils.preprocess import POI_traj_Dataset

def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((1, length))
    resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[0])
    return resampled_trajectory.T


temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

# add an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()


# makedir for saving the generated trajectories
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

config = SimpleNamespace(**temp)

unet = Spatial_UNet(config).cuda()
# load the model
unet.load_state_dict(torch.load(args.model))

n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta=0.0
timesteps=100
batchsize = 512
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

lengths = 336
Gen_traj = []

data = POI_traj_Dataset('data/NYC/train_set.csv', 'data/NYC/gps', 'data/NYC/category')
mask_traj = data.masks
mask_traj = torch.from_numpy(mask_traj).float()[:batchsize]

for i in tqdm(range(1)):
    # Start with random noise
    x = torch.randn(batchsize, 1, config.data.traj_length).cuda()
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, mask_traj)
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:1,:]
    # resample the trajectory length
    for j in range(batchsize):
        new_traj = resample_trajectory(trajs[j].T, lengths[j])
        Gen_traj.append(new_traj)
    break

# save the generated trajectories
Gen_traj = np.array(Gen_traj)
print(Gen_traj.shape)
print(Gen_traj[0])
np.save(args.save_path + 'Gen_traj.npy', Gen_traj)