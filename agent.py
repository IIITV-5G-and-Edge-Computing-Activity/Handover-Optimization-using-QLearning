import numpy as np
import random

class Agent:
    def __init__(self, rows, cols, actions, signal_availability):
        self.rows = rows
        self.cols = cols
        self.actions = actions
        self.Model = {}
        self.Q = {}
        for r in range(rows):
            for c in range(cols):
                self.Q[(r, c)] = {}
                for i in signal_availability.keys():
                    if i == (r, c):
                        for ant in signal_availability[i]:
                            self.Q[(r, c)][ant] = 0

    def antenna_selection(self, state, action, eps, signal_availability, Q):
        r, c = state
        if np.random.rand() < eps:
            for i in signal_availability.keys():
                if i == (r, c):
                    antenna = random.choice(signal_availability[i])
        else:
            for i in signal_availability.keys():
                if i == (r, c):
                    antenna = max(Q[(r, c)], key=Q[(r, c)].get)
        return antenna

    def action_selection(self, state):
        action = random.choice(self.actions)
        return action

    def Q_update(self, state, antenna, next_state, reward, lr, gamma, Q):
        next_max = max(list(Q[next_state].values()))
        Q[state][antenna] = (1 - lr) * Q[state][antenna] + lr * (reward + gamma * next_max)

    def Model_update(self, state, antenna, next_state, reward):
        if state not in self.Model.keys():
            self.Model[state] = {}
        self.Model[state][antenna] = (next_state, reward)

    def n_step_Q_update(self, n, lr, gamma, Q):
        for _ in range(n):
            rand_s = np.random.randint(len(self.Model.keys()))
            random_state = list(self.Model)[rand_s]
            rand_a = np.random.randint(len(self.Model[random_state].keys()))
            random_action = list(self.Model[random_state])[rand_a]
            next_state_r, reward_r = self.Model[random_state][random_action]
            next_max = max(list(Q[random_state].values()))
            Q[random_state][random_action] = (1 - lr) * Q[random_state][random_action] + lr * (reward_r + gamma * next_max)

    def reset(self, rows, cols, signal_availability):
        self.Q = {}
        for r in range(rows):
            for c in range(cols):
                self.Q[(r, c)] = {}
                for i in signal_availability.keys():
                    if i == (r, c):
                        for ant in signal_availability[i]:
                            self.Q[(r, c)][ant] = 0
        self.Model = {}
