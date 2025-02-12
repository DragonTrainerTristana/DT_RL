# 코드 분리할 거니까, 지금은 대충 이렇게 짜놓고 기능별로 class 별로 전부 나뉘게 할꺼임 ㅇㅇ
# 나중에 강의도 들어야지

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
env_name = "AirCombatRL" # 이건 맘대로 설정하면 됨
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # model 저장용
save_path = os.path.join(".", "saved_models", env_name, "PPO", date_time) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA 진짜 화남

# 행동 확률 계산하는 함수임.
# Policy Update 할 때 사용

# 입력 상태 정규화해야함
# Real-Time으로 평균과 표준편차를 계산함 -> 학습 안정성 위함임

# 학습전 (수집)된 경험을 학습하기 위한 형태로 변환함
# 상태, 행동, 보상 등을 배치형태로 구현해야함

# 미래 보상의 현재 가치를 계산해야함
# GAE(Generalized Advantage Estimation) 방식으로 리턴 값 계산함

