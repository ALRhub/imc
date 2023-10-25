import random

from omegaconf import OmegaConf
import torch
import numpy as np

from src.common.experts.mlp_experts import SingleHeadMlpExpert
from src.common.inference_net import InferenceNet
from src.em.em import ExpectationMaximization
from src.franka_kitchen.dataloader import RelayKitchenDataset
from src.franka_kitchen.eval_model import eval_model
from src.util.path_utils import project_path

if __name__ == '__main__':
    # Load configs
    conf = OmegaConf.load(project_path('./config/em_franka_kitchen.yaml'))
    expert_conf = conf['expert_config']
    inference_net_conf = conf['inference_net_config']
    environment_conf = conf['environment_config']

    # Set the random seeds
    random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    np.random.seed(conf['seed'])
    torch.cuda.manual_seed_all(conf['seed'])

    # Fetch data
    data_dir = project_path('./src/franka_kitchen/data/')
    dataset = RelayKitchenDataset(data_dir, conf['device'])
    obs_dim, action_dim = dataset.get_data_dim()

    # Initialize experts, inference net
    expert = SingleHeadMlpExpert(obs_dim=obs_dim,
                                 action_dim=action_dim,
                                 n_components=conf['n_components'],
                                 hidden_dim=expert_conf['hidden_dim'],
                                 num_hidden_layer=expert_conf['num_hidden_layer'],
                                 device=conf['device'])

    inference_net = InferenceNet(obs_dim=obs_dim,
                                 n_components=conf['n_components'],
                                 num_hidden_layer=inference_net_conf['num_hidden_layer'],
                                 hidden_dim=inference_net_conf['hidden_dim'],
                                 device=conf['device'])

    # Initialize the algorithm
    em = ExpectationMaximization(obs_dim=obs_dim,
                                  action_dim=action_dim,
                                  n_components=conf['n_components'],
                                  expert=expert,
                                  expert_config=expert_conf,
                                  inference_net=inference_net,
                                  inference_net_config=inference_net_conf,
                                  train_dataset=dataset,
                                  iterations=conf['iterations'],
                                  batchsize=conf['batchsize'],
                                  seed=conf['seed'],
                                  tol=conf['tol'],
                                  device=conf['device'],
                                  num_workers=conf['num_worker']
                                  )

    # Train the model using EM
    em.train()

    # Evaluate the model
    eval_model(em.model, environment_conf['eval_traj'], dataset.obs_mean,
               dataset.obs_std, dataset.action_mean, dataset.action_std,
               environment_conf['render'])
