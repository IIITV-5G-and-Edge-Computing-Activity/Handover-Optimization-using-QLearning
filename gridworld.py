import numpy as np

class GridWorld:
    def __init__(self, rows, cols, start, goal, antennas, pos):
        self.action_space = ['r', 'u', 'l', 'd']
        self.rows = rows
        self.cols = cols
        self.state = start
        self.Start = start
        self.Goal = goal
        self.Antennas = antennas
        self.done = False
        self.grid = np.zeros((self.rows, self.cols))
        self.grid[self.Goal] = 2
        for i in pos:
            self.grid[i] = -1

    def reset(self):
        self.state = self.Start
        self.done = False

    def step(self, action, antenna, eps, signal_availability, Q):
        r, c = self.state
        old_antenna = antenna
        if action == 'u':
            r -= 1
        elif action == 'd':
            r += 1
        elif action == 'r':
            c += 1
        elif action == 'l':
            c -= 1

        if np.random.rand() < eps:
            for i in signal_availability.keys():
                if i == (r, c):
                    antenna = np.random.choice(signal_availability[i])
        else:
            for i in signal_availability.keys():
                if i == (r, c):
                    antenna = max(Q[(r, c)], key=Q[(r, c)].get)

        new_antenna = antenna

        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.state = (r, c)

        if self.state == self.Goal:
            reward = 100
            handover = 0
            self.done = True
        else:
            if old_antenna != new_antenna:
                reward = -1  # Handover
                handover = 1
            else:
                reward = 0  # No Handover
                handover = 0

        return self.state, reward, handover, self.done, None

    def show_grid(self):
        self.grid[self.Start] = 1
        for r in range(self.rows):
            print(' -------------------------')
            output = ''
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    value = 'M'
                elif self.grid[r, c] == 0:
                    value = '0'
                elif self.grid[r, c] == -1:
                    value = 'A'
                elif self.grid[r, c] == 2:
                    value = 'G'
                output += ' | ' + value
            print(output + ' | ')
        print(' -------------------------')
