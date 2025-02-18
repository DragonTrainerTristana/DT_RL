import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import *
from utils.functions import log_normal_density

class MessageActor(nn.Module):
    """에이전트간 통신을 위한 메시지를 생성하는 네트워크"""
    def __init__(self, frames, msg_action_space, n_agent):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        self.logstd = nn.Parameter(torch.zeros(2*msg_action_space))

        # 상태를 처리하는 CNN 레이어
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, 
                                    kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, 
                                    kernel_size=3, stride=2, padding=1)
        
        # 특징을 처리하는 완전연결 레이어
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 = nn.Linear(256+2+2, 128)  # 256 + goal(2) + speed(2)
        
        # 메시지 생성을 위한 출력 레이어
        self.actor1 = nn.Linear(128, msg_action_space)  # [0,1] 범위 메시지
        self.actor2 = nn.Linear(128, msg_action_space)  # [-1,1] 범위 메시지

    def forward(self, x, goal, speed):
        """순전파 함수"""
        x = x.view(-1, self.frames, 512)
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = a.view(-1, self.n_agent, 256)

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        
        msg1 = torch.sigmoid(self.actor1(a))  # [0,1] 범위
        msg2 = torch.tanh(self.actor2(a))     # [-1,1] 범위
        msg = torch.cat((msg1, msg2), -1)
        
        logstd = self.logstd.expand_as(msg)
        std = torch.exp(logstd)
        msg = torch.normal(msg, std)
        
        logprob = log_normal_density(msg, msg, std=std, log_std=logstd)
        return msg, logprob, msg

class ControlActor(nn.Module):
    """행동을 결정하는 네트워크"""
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        
        # 관측 상태 처리를 위한 레이어
        self.act_obs_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                    kernel_size=5, stride=2, padding=1)
        self.act_obs_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=2, padding=1)
        self.act_obs_fc1 = nn.Linear(128*32, 256)
        self.act_obs_fc2 = nn.Linear(256+2+2, 128)
        self.act_obs_fc3 = nn.Linear(128, 4*msg_action_space)
        
        # 메시지와 상태를 결합하여 행동 생성
        self.act_fc1 = nn.Linear(4*msg_action_space+4*msg_action_space, 64)
        self.act_fc2 = nn.Linear(64+2+2, 128)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, y):
        """순전파 함수"""
        # 관측 상태 처리
        y = y.view(-1, self.frames, 512)
        a = F.relu(self.act_obs_cv1(y))
        a = F.relu(self.act_obs_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_obs_fc1(a))
        a = a.view(-1, self.n_agent, 256)

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_obs_fc2(a))
        a = F.relu(self.act_obs_fc3(a))

        x = torch.cat((a,x), dim=-1)
        act = self.act_fc1(x)
        act = act.view(-1, self.n_agent, 64)

        act = torch.cat((act, goal, speed), dim=-1)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean

class CNNPolicy(nn.Module):
    """전체 정책을 관리하는 네트워크"""
    def __init__(self, msg_action_space, ctr_action_space, frames, n_agent):
        super(CNNPolicy, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        
        # 메시지 액터와 컨트롤 액터 초기화
        self.msg_actor = MessageActor(frames, msg_action_space, n_agent)
        self.ctr_actor = ControlActor(frames, msg_action_space, ctr_action_space, n_agent)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

        # 크리틱(가치 평가) 네트워크
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, 
                                    kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, 
                                    kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed):
        """순전파 함수"""
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
            
        msg, _, _ = self.msg_actor(x, goal, speed)
        
        ctr_input = msg.sum(dim=1, keepdim=True)
        ctr_input = ctr_input.repeat((1, self.n_agent, 1))
        ctr_input = ctr_input - msg
        ctr_input = torch.cat((msg, ctr_input), 2)
        
        action, logprob, mean = self.ctr_actor(ctr_input, goal, speed, x)
        
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        x = x.view(-1, self.frames, 512)
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = v.view(-1, self.n_agent, 256)

        if len(goal.shape) < 3:
            goal = goal.unsqueeze(0)
        if len(speed.shape) < 3:
            speed = speed.unsqueeze(0)

        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        """주어진 행동의 가치와 확률을 평가하는 함수"""
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        
        return v, logprob, dist_entropy