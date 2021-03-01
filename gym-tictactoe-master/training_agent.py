import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

np.random.seed(123)
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'tictactoe' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registration.registry.env_specs[env]
import gym_tictactoe

GAMMA = 0.9  # Q_value的discount rate，以便計算未來reward的折扣回報
INITIAL_EPSILON = 0.3  # 貪婪選擇法的隨機選擇行為的程度
DECAY_RATIO = 0.998  # epsilon衰減的速度
FINAL_EPSILON = 0.01  # 當epsilon到達這個值就不再衰減
LEARNING_RATE = 0.0001  # 神經網路的學習率

REPLAY_SIZE = 10000  # 經驗回放空間
BATCH_SIZE = 200  # 小批量尺寸
TARGET_Q_STEP = 100	 # 目標網路訓練次數
SAVING_STEP = 500  # 儲存model次數


class DQN():  # DQN Agent
    def __init__(self, env):
        # 經驗池
        self.replay_buffer = deque()
        # 初始化參數
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.SIZE = env.SIZE
        self.state_dim = self.SIZE*self.SIZE+1
        self.action_dim = self.SIZE*self.SIZE

        # 創建模型
        self.model = self.build_net()
        self.target_model = self.build_net()
        self.target_q_step = TARGET_Q_STEP
        self.saving_step = SAVING_STEP

        # 讀取模型
        if os.path.exists('tictactoeAI.h5'):
            self.model.load_weights('tictactoeAI.h5')

    def build_net(self):  # 建構出神經網路
        # 兩層隱藏層，且皆為32層，因此神經網路為26>32>32>25
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='relu'))
        adam = optimizers.Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        
        return model

    def copyWeightsToTarget(self):  # 把訓練中的模型複製到目標模型上
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):  # epsilon貪婪選擇法
        q_value = self.model.predict(state)[0]
        min_v = q_value[np.argmin(q_value)]-1  # 設定最小值為最小的q_value-1
        valid_action = []
        for i in range(self.action_dim):
            if(state[0][i] == 0):
                valid_action.append(i)
            else:
                q_value[i] = min_v

        self.update_epsilon()
        if(random.random() <= self.epsilon):
            return valid_action[random.randint(0, len(valid_action) - 1)]
        else:
            return np.argmax(q_value)

    def update_epsilon(self):  # 更新epsilon值
        if(self.epsilon >= FINAL_EPSILON):
            self.epsilon *= DECAY_RATIO

    def remember(self, state, action, reward, next_state, done):  # 向經驗池新增資料
        item = [state, action, reward, next_state, done]
        self.replay_buffer.append(item)

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train()

    def train(self):
        self.time_step += 1
        # 從經驗池中隨機取樣一組batch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])

        # 計算訓練的的結果
        y_batch = self.model.predict(state_batch)
        q_value_batch = self.target_model.predict(next_state_batch)
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch[i][action_batch[i]] = reward_batch[i]
            else:
                y_batch[i][action_batch[i]] = reward_batch[i] + \
                                    GAMMA * np.max(q_value_batch[i])

        self.model.train_on_batch(state_batch, y_batch)

        # 同步目標網路
        if self.time_step % self.target_q_step == 0:
            self.copyWeightsToTarget()

        # 儲存神經網路
        if self.time_step % self.saving_step == 0:
            self.model.save_weights('tictactoeAI.h5', overwrite=True)

# ---------------------------------------------------------
ENV_NAME = 'tictactoe-v0'
EPISODE = 50000  # Episode limitation
STEP = 12  # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode

def main():
    # data存取「X贏」、「O贏」和「平手」的次數
    file = open('result.txt', 'r')
    data = file.read()
    if not data:
        data = [0]*3
    else:
        data = data.split(',')
        for i in range(len(data)):
            data[i] = int(data[i])
        file.close()
    file = open('result.txt', 'w+')

    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    SIZE = env.SIZE

    agent.copyWeightsToTarget()

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()

        camp = 1 # 'O':1 'X':-1
        state = np.reshape(state, [-1])
        state = np.append(state, camp)

        print('episode ', episode)

        # Train
        for step in range(STEP):
            # 計算下一步棋
            if step&1: #輪到X
                camp = -1
                if step == 11: # 如果是最後一輪
                    n = 3
                else:
                    n = 2
                for i in range(n): # 最後一輪連走三手，其他時候連走兩手
                    action = agent.egreedy_action(np.array([state]))
                    action = [action//SIZE, action%SIZE, camp]
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [-1])
                    if i < n-1: # 下一手換誰
                        camp = -1
                    else:
                        camp = 1
                    next_state = np.append(next_state, camp)
                    # 紀錄盤面
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
            else: # 輪到O
                camp = 1
                for i in range(2): # 連走兩手
                    action = agent.egreedy_action(np.array([state]))
                    action = [action//SIZE, action%SIZE, camp]
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [-1])
                    if i == 0: # 下一手換誰
                        camp = 1
                    else:
                        camp = -1
                    next_state = np.append(next_state, camp)
                    # 紀錄盤面
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
            #print(reward)
            
            # 統計誰勝利或平手的次數
            '''
            if done:
                if(reward == 10):
                    data[0] += 1
                elif(reward == -5):
                    data[1] += 1
                else:
                    data[2] += 1
            '''
    agent.model.save_weights('tictactoeAI.h5', overwrite=True)
    file.write(str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]))
    file.close()

if __name__ == "__main__":
    main()
