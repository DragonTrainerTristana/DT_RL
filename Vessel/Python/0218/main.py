import os
import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from config import *
from networks import CNNPolicy
from memory import Memory
from functions import calculate_returns

def main():
    # 환경 설정
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="AirCombatRL", 
        side_channels=[channel],
        worker_id=1,
        base_port=5006
    )
    channel.set_configuration_parameters(time_scale=1.0)

    # 정책 네트워크와 옵티마이저 초기화
    policy = CNNPolicy(MSG_ACTION_SPACE, CONTINUOUS_ACTION_SIZE, FRAMES, N_AGENT).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # 텐서보드 로깅 설정
    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'logs'))

    # 학습 루프
    total_steps = 0
    for episode in range(NUM_EPISODES):
        memory = Memory()
        episode_reward = 0
        step = 0
        
        # 환경 초기화
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        while step < MAX_STEPS:
            # 상태 정보 추출
            state = decision_steps.obs[0]
            if len(decision_steps.obs) > 1:
                goal = decision_steps.obs[1]
                speed = decision_steps.obs[2]
            else:
                goal = np.zeros((N_AGENT, 2))
                speed = np.zeros((N_AGENT, 2))
            
            # 행동 선택
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                goal_tensor = torch.FloatTensor(goal).to(DEVICE)
                speed_tensor = torch.FloatTensor(speed).to(DEVICE)
                value, action, logprob, _ = policy(state_tensor, goal_tensor, speed_tensor)
            
            # 환경에 행동 적용
            action_tuple = ActionTuple(continuous=action.cpu().numpy())
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            # 다음 상태 정보 얻기
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # 보상과 종료 여부 처리
            reward = decision_steps.reward[0] if len(decision_steps) > 0 else terminal_steps.reward[0]
            done = len(terminal_steps) > 0
            
            # 경험 저장
            memory.add(state, goal, speed, action.cpu().numpy(), 
                      reward, done, value.cpu().numpy(), logprob.cpu().numpy())
            
            episode_reward += reward
            total_steps += 1
            step += 1
            
            # 업데이트 수행
            if total_steps % UPDATE_INTERVAL == 0:
                # 리턴값 계산
                returns = calculate_returns(
                    memory.rewards,
                    memory.dones,
                    memory.values[-1],
                    memory.values,
                    gamma=DISCOUNT_FACTOR
                )
                
                # PPO 업데이트
                for _ in range(N_EPOCH):
                    batch_indices = np.random.permutation(len(memory.states))
                    for start in range(0, len(memory.states), BATCH_SIZE):
                        end = start + BATCH_SIZE
                        batch_idx = batch_indices[start:end]
                        
                        # 배치 데이터 준비
                        states_batch = torch.FloatTensor(memory.states[batch_idx]).to(DEVICE)
                        goals_batch = torch.FloatTensor(memory.goals[batch_idx]).to(DEVICE)
                        speeds_batch = torch.FloatTensor(memory.speeds[batch_idx]).to(DEVICE)
                        actions_batch = torch.FloatTensor(memory.actions[batch_idx]).to(DEVICE)
                        returns_batch = torch.FloatTensor(returns[batch_idx]).to(DEVICE)
                        old_logprobs_batch = torch.FloatTensor(memory.logprobs[batch_idx]).to(DEVICE)
                        
                        # PPO 업데이트
                        values, logprobs, dist_entropy = policy.evaluate_actions(
                            states_batch, goals_batch, speeds_batch, actions_batch
                        )
                        
                        ratio = torch.exp(logprobs - old_logprobs_batch)
                        advantage = returns_batch - values
                        
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1-EPSILON, 1+EPSILON) * advantage
                        
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(values, returns_batch)
                        loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * dist_entropy
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                
                memory.clear()
            
            if done:
                break
        
        # 로깅
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        print(f"Episode {episode}, Steps: {step}, Reward: {episode_reward:.2f}")
        
        # 모델 저장
        if episode % SAVE_INTERVAL == 0:
            torch.save(policy.state_dict(), 
                      os.path.join(SAVE_PATH, f'policy_episode_{episode}.pth'))

    # 환경 종료
    env.close()

if __name__ == "__main__":
    main()