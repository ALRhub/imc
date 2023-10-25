import einops
from torch.utils.data import TensorDataset
from pathlib import Path
import numpy as np


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


import torch


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = dataset[0].size(0)

    def __iter__(self):
        self._i = 0

        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.dataset = [v[index_shuffle] for v in self.dataset]

        return self

    def __next__(self):

        i1 = self.batch_size * self._i
        i2 = min(self.batch_size * (self._i + 1), self.data_size)

        if i1 >= self.data_size:
            raise StopIteration()

        value = [v[i1:i2] for v in self.dataset]

        self._i += 1

        return value


class RelayKitchenDataset(TensorDataset):
    def __init__(self, data_directory, device="cuda"):
        data_directory = Path(data_directory)
        self.device = device
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks = transpose_batch_timestep(
            observations, actions, masks
        )
        self.masks = masks

        observations = observations.reshape(-1, 60)[masks.reshape(-1, ) == 1][:, :30]
        actions = actions.reshape(-1, 9)[masks.reshape(-1, ) == 1][:, :30]

        self.n_samples = observations.shape[0]

        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None

        self.observations = torch.from_numpy(self.normalize_obs(observations).astype(np.float32)).to(device)
        self.actions = torch.from_numpy(self.normalize_actions(actions).astype(np.float32)).to(device)

    def get_data_dim(self):
        return self.observations.shape[1], self.actions.shape[1]

    def normalize_obs(self, obs):
        self.obs_mean = obs.mean(0)
        self.obs_std = obs.std(0)
        return (obs - self.obs_mean) / self.obs_std

    def normalize_actions(self, actions):
        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0)
        return (actions - self.action_mean) / self.action_std

    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]

        return obs, act, idx

    def __len__(self):
        return self.n_samples
