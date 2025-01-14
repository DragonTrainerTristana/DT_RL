import os
import logging
import sys
import socket
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from model.net import MLPPolicy, CNNPolicy
from model.ppo import ppo_update_stage1, generate_train_data, generate_action, transform_buffer

# 학습 파라미터
MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5


def run(env, policy, policy_path, action_bound, optimizer):
    """
    ML-Agents 환경에서 정책 학습을 실행하는 함수
    
    env: Unity ML-Agents 환경
    policy: CNNPolicy (정책 네트워크)
    policy_path: 정책 저장 경로
    action_bound: 행동 범위
    optimizer: 옵티마이저
    """
    buff = []
    global_update = 0
    global_step = 0

    for id in range(MAX_EPISODES):
        env.reset()  # 환경 초기화
        decision_steps, terminal_steps = env.get_steps("BehaviorName")

        # 초기 상태 설정
        obs_stack = deque([decision_steps.obs[0]] * 3)
        goal = decision_steps.obs[1]  # 목표값 (예시)
        speed = decision_steps.obs[2]  # 속도값 (예시)
        state = [obs_stack, goal, speed]
        terminal = False
        ep_reward = 0
        step = 1

        while not terminal:
            # 정책에 따라 행동 생성
            v, a, logprob, scaled_action = generate_action(
                state_list=state,
                policy=policy,
                action_bound=action_bound
            )

            # 행동을 환경에 적용
            env.set_actions("BehaviorName", scaled_action)
            env.step()

            # 환경에서 결과 가져오기
            decision_steps, terminal_steps = env.get_steps("BehaviorName")
            r = decision_steps.reward
            ep_reward += np.sum(r)

            # 다음 상태 설정
            s_next = decision_steps.obs[0]
            obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = decision_steps.obs[1]  # 목표값 업데이트
            speed_next = decision_steps.obs[2]  # 속도값 업데이트
            state_next = [obs_stack, goal_next, speed_next]

            # HORIZON 주기마다 업데이트 준비
            if global_step % HORIZON == 0:
                last_v, _, _, _ = generate_action(
                    state_list=state_next,
                    policy=policy,
                    action_bound=action_bound
                )

            buff.append((state, a, r, terminal, logprob, v))

            if len(buff) > HORIZON - 1:
                s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                    transform_buffer(buff=buff)
                t_batch, advs_batch = generate_train_data(
                    rewards=r_batch, gamma=GAMMA, values=v_batch,
                    last_value=last_v, dones=d_batch, lam=LAMDA
                )
                memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                ppo_update_stage1(
                    policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                    epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                    num_env=1, frames=LASER_HIST, obs_size=OBS_SIZE, act_size=ACT_SIZE
                )

                buff = []
                global_update += 1

            global_step += 1
            state = state_next

        # 정책 저장
        if global_update != 0 and global_update % 20 == 0:
            torch.save(policy.state_dict(), os.path.join(policy_path, f'Stage1_{global_update}'))
            logger.info(f'모델 저장 완료 (업데이트 {global_update}회)')

        logger.info(f'Episode {id + 1}, Reward: {ep_reward}')


if __name__ == '__main__':
    # 로깅 설정
    hostname = socket.gethostname()
    log_dir = f'./log/{hostname}'
    os.makedirs(log_dir, exist_ok=True)

    output_file = os.path.join(log_dir, 'output.log')
    cal_file = os.path.join(log_dir, 'cal.log')

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # ML-Agents 환경 설정
    engine_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name="Your_Unity_Environment", side_channels=[engine_channel])

    # 정책 및 옵티마이저 초기화
    policy_path = 'policy'
    os.makedirs(policy_path, exist_ok=True)

    policy = CNNPolicy(frames=LASER_HIST, action_space=ACT_SIZE)
    policy.cuda()

    optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

    # 학습 시작
    try:
        run(env=env, policy=policy, policy_path=policy_path, action_bound=[[-1, 1], [1, 1]], optimizer=optimizer)
    except KeyboardInterrupt:
        logger.info('학습 중단됨')
    finally:
        env.close()
