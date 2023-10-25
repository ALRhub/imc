# Information Maximizing Curriculum: A Curriculum-Based Approach for Imitating Versatile Skills

## Abstract 
Imitation learning uses data for training policies to solve complex tasks. However, when the training data is collected from human demonstrators, it often leads to multimodal distributions because of the variability in human actions. Most imitation learning methods rely on a maximum likelihood (ML) objective to learn a parameterized policy, but this can result in suboptimal or unsafe behavior due to the mode-averaging property of  the ML objective. In this work, we propose Information Maximizing Curriculum, a curriculum-based approach that assigns a weight to each data point and encourages the model to specialize in the data it can represent, effectively mitigating the mode-averaging problem by allowing the model to ignore data from modes it cannot represent. To cover all modes and thus, enable versatile behavior, we extend our approach to a mixture of experts (MoE) policy, where each mixture component selects its own subset of the training data for learning. A novel, maximum entropy-based objective is proposed  to achieve full coverage of the dataset, thereby enabling the policy to encompass all modes within the data distribution. We demonstrate the effectiveness of our approach on complex simulated control tasks using versatile human demonstrations, achieving superior performance compared to state-of-the-art methods.

## About this Code Base
This code base provides code for testing the IMC and EM algorithm on the Franka kitchen environment. 

## Installation Guide
- Setup a virtal Conda environment via
    ```sh
    conda env create --file=conda_env.yml
    ```
- Activate the environment:
    ```sh
    conda activate imc
    ```
- Clone the [Relay Policy Learning repository](https://github.com/google-research/relay-policy-learning) via
    ```sh
    git clone https://github.com/google-research/relay-policy-learning
    ```
- Install the MuJoCo 2.1.0 physics engine (see https://github.com/openai/mujoco-py#install-mujoco)
- Set PYTHONPATH to the root directory of the project (imc)
    ```sh
    conda env config vars set PYTHONPATH=<path to imc>/imc
    ```
- Additionally add the relay-policy-learning repo to the PYTHONPATH (relay-policy-learning/adept_envs)
    ```sh
    conda env config vars set PYTHONPATH=$PYTHONPATH:<path to relay-policy-learning>/relay-policy-learning/adept_envs
    ```

## Download the Dataset
- Download bet_data_release.tar.gz here: https://osf.io/983qz/ (taken from [this repository](https://github.com/notmahi/bet))
- Extract the tar.gz and copy files from /bet_data_release/kitchen/ into src/franka_kitchen/data/

## Running Experiments
For starting the experiment using IMC on Franka kitchen run 
```sh
python run_imc.py
```
for EM use
```sh
python run_em.py
```
The configuration files with hyperparameters for both algorithms can be found under config/.
The enable/disable rendering of the environment set the ```render``` flag in the environment_config to true/false.