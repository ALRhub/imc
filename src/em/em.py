import torch
import abc

import torch.utils.data as Data

from tqdm import tqdm

from src.common.inference_net import SoftCrossEntropyLoss
from src.common.moe_policy import MixtureOfExpertsPolicy
from src.franka_kitchen.dataloader import DataLoader


class ExpectationMaximization(abc.ABC):
    def __init__(
            self,
            obs_dim,
            action_dim,
            n_components,
            expert,
            expert_config,
            inference_net,
            inference_net_config,
            train_dataset,

            iterations=100,
            batchsize=1024,

            seed: int = 0,
            tol: float = 1e-3,

            logger=None,
            device='cuda',
            num_workers=16
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        #
        self.dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=num_workers,
        )

        self.expert = expert
        self.expert_config = expert_config
        self.inference_net = inference_net
        self.inference_net_config = inference_net_config
        self.device = device

        # Training params
        self.n_components = n_components
        self.expert_losses = []
        self.lower_bounds = []
        self.lower_bound = torch.inf
        self.tol = tol
        self.iterations = iterations
        self.converged = False
        self.gating_updated = False

        self.seed = seed

        self.logger = logger

        # Uniformly initialize the responsibilities log p(z|o)
        self.log_responsibilities = torch.log(torch.ones([len(train_dataset), n_components], device=device) / n_components)
        self.model = self.build_model()

        self.expert_optimizer = torch.optim.Adam(self.expert.parameters(),
                                                 lr=expert_config['learning_rate'])
        self.inference_net_optimizer = torch.optim.Adam(self.inference_net.parameters(),
                                                        lr=inference_net_config['learning_rate'])

    def build_model(self):
        return MixtureOfExpertsPolicy(action_dim=self.action_dim,
                                      obs_dim=self.obs_dim,
                                      experts=self.expert,
                                      inference_net=self.inference_net,
                                      n_components=self.n_components,
                                      device=self.device,
                                      logger=self.logger)

    def train(self):
        print('Training mixture of expert policy')
        for _ in tqdm(range(self.iterations)):
            expert_losses = []

            self.e_step()
            for _ in range(self.expert_config['expert_epochs']):
                for data in self.train_dataloader:
                    obs, act, idx = data
                    expert_loss = self.m_step_experts(obs, act, idx)
                    expert_losses.append(expert_loss)

            self.expert_losses.append(sum(expert_losses) / len(expert_losses))

        print('Training inference net')
        self.train_inference_net()

    def e_step(self):
        # Update the log responsibilities log p(z|o)
        with torch.no_grad():
            expert_log_likelihoods = self.model.expert_log_likelihoods(self.dataset.observations, self.dataset.actions)
            log_responsibilities = expert_log_likelihoods + self.log_responsibilities
            log_responsibilities = log_responsibilities - torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
            self.log_responsibilities = log_responsibilities

    def m_step_experts(self, obs, act, idx):
        # Update the expert parameters θ using weighted maximum likelihood estimation
        weights = self.log_responsibilities[idx].exp()
        expert_log_likelihoods = self.model.expert_log_likelihoods(obs, act)
        expert_loss = -(weights * expert_log_likelihoods).mean()

        self.expert_optimizer.zero_grad()
        expert_loss.backward()
        self.expert_optimizer.step()
        return expert_loss.item()

    def train_inference_net(self):
        inference_targets = self.log_responsibilities.exp()

        inference_dataset = Data.TensorDataset(self.dataset.observations, inference_targets)
        inference_loader = DataLoader(dataset=inference_dataset.tensors,
                                      batch_size=self.inference_net_config['batchsize'], shuffle=True, )
        self.inference_net.trained = True
        loss_fn = SoftCrossEntropyLoss()

        # Train the inference network parameters ϕ
        for _ in tqdm(range(self.inference_net_config['epochs'])):

            for data in inference_loader:
                observations, curriculum_weights = data

                log_posterior_pred = self.inference_net(observations)
                loss = loss_fn(log_posterior_pred, curriculum_weights)
                self.inference_net_optimizer.zero_grad()
                loss.backward()
                self.inference_net_optimizer.step()
