import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optimizer
from simple2fjsp import JobEnv


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity):
        alpha = 0.6
        beta = 0.4
        beta_increment_step = 1000
        beta_increment = (1 - beta) / beta_increment_step
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([observation, action, reward, next_observation, done])
        else:
            self.memory[self.pos] = [observation, action, reward, next_observation, done]
        self.priorities[self.pos] = max_prior
        self.pos += 1
        self.pos = self.pos % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (- self.beta)
        if self.beta < 1:
            self.beta += self.beta_increment
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        observation, action, reward, next_observation, done = zip(* samples)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, state_size)
        # self.fc2 = nn.Linear(state_size, state_size)
        self.value_out = nn.Linear(state_size, action_size)

    def forward(self, state):
        feature = F.relu(self.fc1(state))
        # feature = F.relu(self.fc2(feature))
        value = self.value_out(feature)
        return value


class Agent:
    def __init__(self, j_env, buffer_size=5000, batch_size=64, lr=1e-5):
        self.env = j_env
        self.state_size = self.env.state_num
        self.action_size = self.env.action_num
        self.gamma = 0.9
        self.init_epsilon = 0.9
        self.epsilon = self.init_epsilon
        # DDQNetwork
        self.dqn_local = DQN(self.state_size, self.action_size)
        self.optimizer = optimizer.Adam(self.dqn_local.parameters(), lr=lr)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.annealing_cnt = 2000

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.dqn_local(state)
        if random.random() > 1 - self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.epsilon < 1:
            self.epsilon += (1-self.init_epsilon)/self.annealing_cnt
        observation, action, reward, next_observation, done, indices, weights = self.memory.sample(self.batch_size)
        observation = torch.FloatTensor(observation)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        q_values = self.dqn_local(observation).gather(1, action.unsqueeze(1)).squeeze(1)
        q_next = self.dqn_local(next_observation).detach().gather(1, action.unsqueeze(1)).squeeze(1)
        target_values = reward + self.gamma * q_next*(1-done)

        priorities = torch.abs(target_values - q_values).detach().numpy()
        self.memory.update_priorities(indices, priorities)

        loss = (target_values - q_values).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, data_set):
        column = ["episode", "make_span", "reward", "min"]
        results = pd.DataFrame(columns=column, dtype=float)
        min_make_span = 10000
        converged = 0
        converged_value = []
        t0 = time.time()
        for i_epoch in range(8000):
            if time.time() - t0 > 3600:
                break
            state = self.env.reset()
            episode_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.memory.store(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.learn()
                if done:
                    if min_make_span > self.env.current_time:
                        min_make_span = self.env.current_time
                    # Episode: make_span: Episode reward
                    print('{}    {}    {:.2f} {}'.format(i_epoch, self.env.current_time, episode_reward, min_make_span))
                    results.loc[i_epoch] = [i_epoch, self.env.current_time, episode_reward, min_make_span]
                    converged_value.append(self.env.current_time)
                    if len(converged_value) >= 31:
                        converged_value.pop(0)
                    break
            converged = i_epoch
            if min(converged_value) == max(converged_value) and len(converged_value) >= 30:
                break
        if not os.path.exists('results'):
            os.makedirs('results')
        results.to_csv("results/" + str(self.env.case_name) + "_" + data_set + ".csv")
        return min(converged_value), converged, time.time() - t0, min_make_span


if __name__ == "__main__":
    data_set_name = "dqn-per"
    path = "../MK/"
    prefix = data_set_name
    param = [prefix, "converge_cnt", "total_time", "min"]
    for i in range(6):
        name = prefix + str(i)
        simple_results = pd.DataFrame(columns=param, dtype=int)
        for file_name in os.listdir(path):
            print(file_name + "========================")
            title = file_name.split('.')[0]
            env = JobEnv(title, path)
            model = Agent(env)
            simple_results.loc[title] = model.train(name)
        simple_results.to_csv(name + "_result.csv")

