import numpy as np
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
import math
from functools import reduce
import bisect

# 환경 및 학습 하이퍼파라미터 설정
state_size = 138  # RayPerceptionSensor의 크기
msg_action_space = 3  # MessageActor의 액션 공간 크기 
continuous_action_size = 2  # ControlActor의 액션 공간 크기
frames = 4  # 프레임 수 
n_agent = 4  # 에이전트 수

# 모델 로드 및 학습 모드 설정
load_model = False
train_mode = True

# PPO 하이퍼파라미터
discount_factor = 0.995
learning_rate = 3e-4
n_step = 1000
batch_size = 2048
n_epoch = 3
epsilon = 0.2
entropy_bonus = 0.01
critic_loss_weight = 0.5

# 학습 설정
grad_clip_max_norm = 0.5
run_step = 30000000 if train_mode else 0
test_step = 100000
print_interval = 1000
save_interval = 30000

# 환경 설정
env_name = "AirCombatRL"
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = os.path.join(".", "saved_models", env_name, "PPO", date_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
