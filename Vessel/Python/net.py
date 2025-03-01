import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

from model.utils import log_normal_density

class MessageActor(nn.Module):
    def __init__(self, frames, msg_action_space, n_agent):
        super(MessageActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        self.logstd = nn.Parameter(torch.zeros(2 * msg_action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128 * 32, 256)
        self.act_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.actor1 = nn.Linear(128, msg_action_space)
        self.actor2 = nn.Linear(128, msg_action_space)

    def forward(self, x, goal, speed):
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
        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean


class ControlActor(nn.Module):
    def __init__(self, frames, msg_action_space, ctr_action_space, n_agent):
        super(ControlActor, self).__init__()
        self.frames = frames
        self.n_agent = n_agent

        self.act_obs_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_obs_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_obs_fc1 = nn.Linear(128 * 32, 256)
        self.act_obs_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.act_obs_fc3 = nn.Linear(128, 4 * msg_action_space)

        self.act_fc1 = nn.Linear(4 * msg_action_space + 4 * msg_action_space, 64)
        self.act_fc2 = nn.Linear(64 + 2 + 2, 128)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x, goal, speed, y):
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

        x = torch.cat((a, x), dim=-1)
        act = F.relu(self.act_fc1(x))
        act = act.view(-1, self.n_agent, 64)

        act = torch.cat((act, goal, speed), dim=-1)
        act = torch.tanh(F.relu(self.act_fc2(act)))

        mean = self.mu(act)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean


class CNNPolicy(nn.Module):
    def __init__(self, msg_action_space, ctr_action_space, frames, n_agent):
        super(CNNPolicy, self).__init__()
        self.frames = frames
        self.n_agent = n_agent
        self.msg_actor = MessageActor(frames, msg_action_space, n_agent)
        self.ctr_actor = ControlActor(frames, msg_action_space, ctr_action_space, n_agent)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128 * 32, 256)
        self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        msg, _, _ = self.msg_actor(x, goal, speed)
        ctr_input = msg.sum(dim=1, keepdim=True).repeat((1, self.n_agent, 1)) - msg
        ctr_input = torch.cat((msg, ctr_input), dim=-1)
        action, logprob, mean = self.ctr_actor(ctr_input, goal, speed, x)

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

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
