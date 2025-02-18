class Memory:
    """경험 저장을 위한 메모리 버퍼 클래스
    
    각 에피소드에서 얻은 경험(상태, 행동, 보상 등)을 저장하고 관리합니다.
    """
    def __init__(self):
        """메모리 버퍼 초기화"""
        self.states = []     # 상태
        self.goals = []      # 목표 위치
        self.speeds = []     # 현재 속도
        self.actions = []    # 수행한 행동
        self.rewards = []    # 받은 보상
        self.dones = []      # 에피소드 종료 여부
        self.values = []     # 상태 가치
        self.logprobs = []   # 행동의 로그 확률
    
    def clear(self):
        """메모리 버퍼 초기화"""
        self.__init__()

    def add(self, state, goal, speed, action, reward, done, value, logprob):
        """새로운 경험을 메모리에 추가
        
        Args:
            state: 현재 상태
            goal: 목표 위치
            speed: 현재 속도
            action: 수행한 행동
            reward: 받은 보상
            done: 에피소드 종료 여부
            value: 상태 가치
            logprob: 행동의 로그 확률
        """
        self.states.append(state)
        self.goals.append(goal)
        self.speeds.append(speed)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)

    def get_batch(self, batch_indices):
        """주어진 인덱스에 해당하는 배치 데이터 반환
        
        Args:
            batch_indices: 배치로 선택할 인덱스들
            
        Returns:
            선택된 배치 데이터
        """
        states = [self.states[i] for i in batch_indices]
        goals = [self.goals[i] for i in batch_indices]
        speeds = [self.speeds[i] for i in batch_indices]
        actions = [self.actions[i] for i in batch_indices]
        rewards = [self.rewards[i] for i in batch_indices]
        dones = [self.dones[i] for i in batch_indices]
        values = [self.values[i] for i in batch_indices]
        logprobs = [self.logprobs[i] for i in batch_indices]
        
        return states, goals, speeds, actions, rewards, dones, values, logprobs

    def __len__(self):
        """저장된 경험의 개수 반환"""
        return len(self.states)