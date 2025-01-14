import torch
import logging
import os
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# 로그 설정
hostname = "mlagents_logs"
os.makedirs(f'./log/{hostname}', exist_ok=True)
ppo_file = f'./log/{hostname}/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    """
    버퍼 데이터를 변환하여 PPO 학습에 필요한 형태로 변환.
    """
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], [], [], [], []
    for e in buff:
        s_batch.append(e[0][0])
        goal_batch.append(e[0][1])
        speed_batch.append(e[0][2])
        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    return (
        np.asarray(s_batch),
        np.asarray(goal_batch),
        np.asarray(speed_batch),
        np.asarray(a_batch),
        np.asarray(r_batch),
        np.asarray(d_batch),
        np.asarray(l_batch),
        np.asarray(v_batch),
    )

def generate_action(state, policy, action_bound):
    """
    주어진 상태를 기반으로 행동 생성.
    """
    s, goal, speed = state

    s = torch.tensor(s).float().cuda()
    goal = torch.tensor(goal).float().cuda()
    speed = torch.tensor(speed).float().cuda()

    v, a, logprob, mean = policy(s, goal, speed)
    v, a, logprob = v.detach().cpu().numpy(), a.detach().cpu().numpy(), logprob.detach().cpu().numpy()
    scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])

    return v, a, logprob, scaled_action

def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    """
    리턴 값 계산.
    """
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value

    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i + 1] * dones[i] + rewards[i]

    return returns

def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    """
    GAE 및 타겟 생성.
    """
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]

    values = np.vstack((values, last_value))
    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in reversed(range(num_step)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        targets[t] = gae + values[t]

    advs = targets - values[:-1]
    return targets, advs

def ppo_update(policy, optimizer, batch_size, memory, epoch, coeff_entropy=0.02, clip_value=0.2):
    """
    PPO 업데이트 단계.
    """
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    for _ in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(range(len(advs))), batch_size=batch_size, drop_last=False)
        for indices in sampler:
            sampled_obs = torch.tensor(obss[indices]).float().cuda()
            sampled_goals = torch.tensor(goals[indices]).float().cuda()
            sampled_speeds = torch.tensor(speeds[indices]).float().cuda()
            sampled_actions = torch.tensor(actions[indices]).float().cuda()
            sampled_logprobs = torch.tensor(logprobs[indices]).float().cuda()
            sampled_targets = torch.tensor(targets[indices]).float().cuda()
            sampled_advs = torch.tensor(advs[indices]).float().cuda()

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(
                sampled_obs, sampled_goals, sampled_speeds, sampled_actions
            )

            ratio = torch.exp(new_logprob - sampled_logprobs)

            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 0.5 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger_ppo.info(f'Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Entropy: {dist_entropy.item()}')

    print('PPO Update Completed')
