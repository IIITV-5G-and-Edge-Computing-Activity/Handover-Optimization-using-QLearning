from gridworld import GridWorld
from agent import Agent
import random
import numpy as np
from plotter import plot_metrics

# Define constants
Rows = 3
Cols = 6
Start = (2, 0)
Goal = (0, 5)
Antennas = {
    "A2": {"position": (2, 1), "Range": [(2, 0), (1, 1), (2, 2), (1, 0), (1, 2), (2, 1)]},
    "A1": {"position": (0, 0), "Range": [(1, 0), (1, 1), (0, 1), (0, 0)]},
    "A3": {"position": (0, 2), "Range": [(0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2)]},
    "A4": {"position": (1, 3), "Range": [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (1, 4), (0, 4), (0, 3), (1, 3)]},
    "A5": {"position": (0, 4), "Range": [(0, 3), (1, 3), (1, 4), (1, 5), (0, 5), (0, 4)]},
    "A6": {"position": (2, 5), "Range": [(1, 4), (2, 4), (1, 5), (2, 5)]},
}
pos = [(0, 0), (2, 1), (0, 2), (1, 3), (0, 4), (2, 5)]

# Generate signal availability
signal_availability = {
    (r, c): [a for a in Antennas.keys() if (r, c) in Antennas[a]["Range"]]
    for r in range(Rows)
    for c in range(Cols)
}

# Hyperparameters
Episodes = 6000
lr = 0.1
gamma = 0.95
eps = 0.82
n = 50
max_steps_per_episode = 150
eps_decaying_start = 0
eps_decaying_end = Episodes // 1.5
eps_decaying = eps / (eps_decaying_end - eps_decaying_start)

# Initialize environment and agent
env = GridWorld(Rows, Cols, Start, Goal, Antennas, pos)
agent = Agent(Rows, Cols, env.action_space, signal_availability)
agent.reset(Rows, Cols, signal_availability)

# Metrics for analysis
Rewards = []
steps_taken = []
Handovers = []

# Training loop
for ep in range(Episodes):
    print(f"Episode: {ep}")
    if eps_decaying_start < ep < eps_decaying_end:
        eps -= eps_decaying
    counter = 0
    env.reset()
    done = False
    state = env.state
    Reward_ep = 0
    Handover_ep = 0

    while not done and counter < max_steps_per_episode:
        action = agent.action_selection(state)
        antenna = agent.antenna_selection(state, action, eps, signal_availability, agent.Q)
        next_state, reward, handover, done, _ = env.step(action, antenna, eps, signal_availability, agent.Q)
        agent.Q_update(state, antenna, next_state, reward, lr, gamma, agent.Q)
        Reward_ep += reward
        Handover_ep += handover
        print(f"Episode: {ep} | State: {state} -> Next State: {next_state}, Reward: {reward}, Accumulated Rewards: {Reward_ep}, Handovers: {Handover_ep}")
        agent.Model_update(state, antenna, next_state, reward)
        state = next_state
        counter += 1

    Rewards.append(Reward_ep)
    Handovers.append(Handover_ep)
    steps_taken.append(counter)
    print("\n")

# Call plotter
plot_metrics(Episodes, Rewards, Handovers, steps_taken, eps, eps_decaying, "./metrics_plots")

# Summary
print(f"Total Episodes: {Episodes}")
print(f"Average Reward per Episode: {np.mean(Rewards):.2f}")
print(f"Total Handovers: {sum(Handovers)}")
print(f"Average Steps per Episode: {np.mean(steps_taken):.2f}")
print(f"Final Epsilon Value: {eps:.4f}\n\n")
