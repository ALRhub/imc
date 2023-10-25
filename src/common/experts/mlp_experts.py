from src.common.experts.base_expert import BaseExpert
import torch
import einops

from src.common.mlps import ResidualMLPNetwork


class SingleHeadMlpExpert(torch.nn.Module, BaseExpert):
    def __init__(self, obs_dim, action_dim, n_components, hidden_dim, num_hidden_layer, device: str = 'cuda'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_components = n_components
        self.model = ResidualMLPNetwork(input_dim=obs_dim,
                                        output_dim=n_components * action_dim,
                                        hidden_dim=hidden_dim,
                                        num_hidden_layers=num_hidden_layer,
                                        dropout=0.,
                                        device=device)

    def log_likelihood(self, obs, act):
        diff = act[:, :, None] - self(obs)
        exp_term = torch.square(torch.linalg.norm(diff, axis=1))
        return -0.5 * exp_term

    def forward(self, obs):
        out_vec = self.model(obs)
        actions = einops.rearrange(
            out_vec,
            "B (A C) -> B A C",  # B = batch, A = action dim, C = number of components
            A=self.action_dim,
            C=self.n_components,
        )
        return actions

    def sample(self, obs, cmp_idx=None):
        actions = self(obs)
        if cmp_idx is None:
            return actions
        else:
            return actions[:, :, cmp_idx]
