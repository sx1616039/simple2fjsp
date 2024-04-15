import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from simple2fjsp import JobEnv

m_seed = 3407
# 设置seed
torch.manual_seed(m_seed)


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta, beta_increment):
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


class DualQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=32):
        super(DualQNetwork, self).__init__()

        self.fc = nn.Linear(state_size, hidden_dim)
        # self.adv_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.adv_out = nn.Linear(hidden_dim, action_size)

        # self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        # x = F.relu(self.adv_fc1(x))
        advantage = self.adv_out(x)
        # v = F.relu(self.value_fc1(x))
        value = self.value_out(x)
        return value + advantage - advantage.mean()


class Agent:
    def __init__(self, j_env, capacity=5000, batch_size=64):
        super(Agent, self).__init__()
        self.env = j_env
        self.state_dim = self.env.state_num
        self.action_dim = self.env.action_num
        self.case_name = self.env.case_name

        self.batch_size = batch_size  # update batch size
        self.init_epsilon = 0.9
        self.epsilon = self.init_epsilon
        self.gamma = 0.9  # reward discount
        self.LR = 1e-3  # learning rate
        self.update_steps = 100
        alpha = 0.6
        beta = 0.4
        beta_increment_step = 1000
        beta_increment = (1 - beta) / beta_increment_step
        self.learn_step_cnt = 0

        self.eval_net = DualQNetwork(self.state_dim, self.action_dim)
        self.target_net = DualQNetwork(self.state_dim, self.action_dim)
        self.eval_net.load_state_dict(self.target_net.state_dict())

        self.optimizer = optimizer.Adam(self.eval_net.parameters(), self.LR)
        self.loss_func = nn.MSELoss()
        self.memory = PrioritizedReplayBuffer(capacity, alpha=alpha, beta=beta, beta_increment=beta_increment)
        if not os.path.exists('param'):
            os.makedirs('param/net_param')
        self.annealing_cnt = 4000

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if np.random.random() > 1 - self.epsilon:
            with torch.no_grad():
                action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].item()
        else:
            action = np.random.choice(range(self.action_dim), 1).item()
        return action

    def save_params(self):
        torch.save(self.eval_net.state_dict(), 'param/net_param/' + self.env.case_name + 'eval_net.model')
        torch.save(self.target_net.state_dict(), 'param/net_param/' + self.env.case_name + 'target_net.model')

    def load_params(self):
        self.eval_net.load_state_dict(torch.load('param/net_param/' + self.env.case_name + 'eval_net.model'))
        self.target_net.load_state_dict(torch.load('param/net_param/' + self.env.case_name + 'target_net.model'))

    def learn(self):
        if len(self.memory) <= self.batch_size:
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

        q_value = self.eval_net(observation).gather(1, action.unsqueeze(1)).squeeze(1)
        argmax_actions = self.eval_net(next_observation).max(1)[1].detach()
        next_q_value = self.target_net(next_observation).detach().gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        loss = (expected_q_value - q_value).pow(2) * weights
        loss = loss.mean()
        priorities = torch.abs(expected_q_value - q_value).detach().numpy()
        self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.learn_step_cnt % self.update_steps == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_cnt += 1

    def train(self, data_set):
        column = ["episode", "make_span", "reward", "min"]
        results = pd.DataFrame(columns=column, dtype=float)
        min_make_span = 10000
        converged = 0
        converged_value = []
        t0 = time.time()
        for i_epoch in range(8000):
            update_cnt = 0
            if time.time() - t0 > 3600:
                break
            buffer_s, buffer_a, buffer_r, buffer_n, buffer_d = [], [], [], [], []
            state = self.env.reset()
            episode_reward = 0
            while True:
                update_cnt += 1
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                buffer_n.append(next_state)
                buffer_d.append(done)

                state = next_state
                episode_reward += reward
                self.learn()
                if done:
                    v_s_ = 0
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    for i in range(len(buffer_a)):
                        self.memory.store(buffer_s[i], buffer_a[i], discounted_r[i], buffer_n[i], buffer_d[i])
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


if __name__ == '__main__':
    data_set_name = "d3qn-per-discount-"
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
