import torch


class MixtureOfExpertsPolicy:
    def __init__(self,
                 action_dim,
                 obs_dim,
                 experts,
                 inference_net,
                 n_components,
                 device='cuda',
                 logger=None):
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.device = device
        self.logger = logger

        self.experts = experts
        self.inference_net = inference_net

        # Initialize first component
        self.n_components = n_components
        # self.init_model()

    def sample(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            gating_probs = self.inference_net.probabilities(obs)

            thresh = torch.cumsum(gating_probs, dim=1)
            thresh[:, -1] = torch.ones(obs.shape[0], device=self.device)
            eps = torch.rand(size=[obs.shape[0], 1], device=self.device)
            comp_idx_samples = torch.argmax(1 * (eps < thresh), dim=-1)

            action_preds = self.experts(obs)  # n_samples x y_dim x n_components

            samples = torch.zeros((obs.shape[0], self.action_dim), device=self.device)
            for i in range(self.n_components):
                ctxt_samples_cmp_i_idx = torch.where(comp_idx_samples == i)[0]
                samples[ctxt_samples_cmp_i_idx, :] = action_preds[ctxt_samples_cmp_i_idx][:, :, i]
        return samples

    def expert_log_likelihoods(self, obs, act):
        """
        Returns: log p(y|x,o) for all o
        """
        return self.experts.log_likelihood(obs, act)

    def to_gpu(self):
        self.experts.to_gpu()
        self.inference_net.to_gpu()

    def to_cpu(self):
        self.experts.to('cpu')
        self.inference_net.to_cpu()
        self.device = 'cpu'

    # def log_posterior(self):
    #     """
    #     Returns: log q(o|c) = p~(c|o) / sum_o p~(c|o) for all x in dataset
    #     """
    #     log_posterior = self.log_curricula - torch.logsumexp(self.log_curricula, dim=1).reshape(-1, 1)
    #
    #     return log_posterior

    # def init_experts(self, expert_init=None):
    #
    #     self.expert_model = SingleHeadExpert(x_dim=self.x_dim,
    #                                          y_dim=self.y_dim,
    #                                          n_components=self.n_components,
    #                                          hidden_dim=self.experts['hidden_dim'],
    #                                          num_hidden_layers=self.experts['num_hidden_layer'],
    #                                          device=self.device)
    #
    #     if expert_init is not None:
    #         list(self.expert_model.model.parameters())[-2].data.copy_(torch.zeros([1, self.expert_model.hidden_dim]))
    #         list(self.expert_model.model.parameters())[-1].data.copy_(expert_init)
    #
    #     self.expert_learner = SingleHeadExpertLearner(model=self.expert_model,
    #                                                   x=self.x, y=self.y,
    #                                                   n_epochs=self.experts['n_epochs'],
    #                                                   batch_size=self.experts['batch_size'],
    #                                                   learning_rate=self.experts['learning_rate'],
    #                                                   weight_decay=self.experts['weight_decay'],
    #                                                   logger=self.logger,
    #                                                   device=self.device)

    # def init_model(self):
    #     # Initialize curriculum
    #     self.log_curricula = torch.ones([self.n_samples, self.n_components], device=self.device) * -1e3
    #
    #     kmeans = KMeans(n_clusters=self.n_components)
    #     kmeans.fit(self.x.to('cpu'))
    #     cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
    #
    #     expert_init = torch.zeros(self.n_components * self.y_dim, device=self.device)
    #
    #     for cmp_idx in range(self.n_components):
    #         center_idx = torch.argmin(torch.linalg.norm(self.x - cluster_centers[cmp_idx], dim=1))
    #         self.log_curricula[center_idx, cmp_idx] = 0.
    #         expert_init[cmp_idx * self.y_dim: (cmp_idx + 1) * self.y_dim] = self.y[center_idx]
    #     # Initialize experts
    #     self.init_experts(expert_init)

    # def check_gating_updated(self):
    #     if not self.gating_updated:
    #         # p(o|c)
    #         # gatings = np.exp(self.log_context_densities - torch.logsumexp(self.log_context_densities, 1).reshape(-1, 1))
    #         self.gating_net = ImcGatingNet(context_dim=self.x_dim,
    #                                        n_components=self.n_components,
    #                                        num_hidden_layer=self.inference_net['num_hidden_layer'],
    #                                        hidden_dim=self.inference_net['hidden_dim'],
    #                                        device=self.device
    #                                        ).to(self.device)
    #
    #         self.gating_learner = ImcGatingNetLearner(network=self.gating_net,
    #                                                   x=self.x,
    #                                                   n_epochs=self.inference_net['gating_net_epochs'],
    #                                                   batch_size=self.inference_net['gating_net_batch_size'],
    #                                                   learning_rate=self.inference_net['gating_net_learning_rate'],
    #                                                   device=self.device,
    #                                                   logger=self.logger)
    #
    #         # curricula = torch.exp(self.log_curricula)
    #         # self.gating_learner.fit(curricula)
    #         self.gating_learner.fit(torch.exp(self.log_posterior()))
    #         self.gating_updated = True
