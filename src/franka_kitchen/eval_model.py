import numpy as np
import torch

from src.franka_kitchen.franka_kitchen import KitchenAllV0


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def eval_model(model, n_trajectories, obs_mean, obs_std, action_mean, action_std,
               render=True):
    env = KitchenAllV0()

    task_sequence_dict = {'0': {},
                          '1': {},
                          '2': {},
                          '3': {},
                          '4': {},
                          '5': {},
                          }

    with torch.no_grad():


        for i in range(n_trajectories):
            print(f'Rollout {i}')
            obs = env.reset()

            completed_tasks_in_order = []

            done = False
            steps = 0
            while (not done) and (steps < 300):
                obs = obs[:30]
                obs = (obs - obs_mean) / obs_std

                obs_t = torch.from_numpy(obs.reshape(1, len(obs)).astype(np.float32))
                action = to_numpy(model.sample(obs_t).reshape(-1, )) * action_std + action_mean

                obs, reward, done, info = env.step(action)

                for task in info['completed_tasks']:
                    if not task in completed_tasks_in_order:
                        completed_tasks_in_order.append(task)

                if done or (steps >= 299):
                    try:
                        task_sequence_dict[f'{str(len(info["completed_tasks"]))}'][tuple(completed_tasks_in_order)] += 1
                    except:
                        task_sequence_dict[f'{str(len(info["completed_tasks"]))}'][tuple(completed_tasks_in_order)] = 1

                    for l in range(len(info["completed_tasks"])):
                        try:
                            task_sequence_dict[f'{str(l)}'][
                                tuple(completed_tasks_in_order[:l])] += 1
                        except:
                            task_sequence_dict[f'{str(l)}'][
                                tuple(completed_tasks_in_order[:l])] = 1

                if render:
                    env.render()
                steps += 1

    reward = 0
    task_entropies = []
    successes = []
    for l in range(1, 6):
        success = []
        for v in task_sequence_dict[f'{l}'].values():
            success.append(v)
        total_successes = sum(success) / n_trajectories
        task_probs = np.array(success) / sum(success)
        task_entropy = - np.sum(task_probs * np.log(task_probs))
        reward += total_successes

        task_entropies.append(task_entropy)
        successes.append(total_successes)

        print(f'tasks solved = {l}')
        print(f'successes = {total_successes}')
        print(f'entropy = {task_entropy}')
    print(f'reward = {reward}')

    return reward, successes, task_entropies