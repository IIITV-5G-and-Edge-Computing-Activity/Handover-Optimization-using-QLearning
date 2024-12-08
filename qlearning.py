import numpy as np
import gym
import os
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('MountainCar-v0')
env.reset()

def QLearn(env, learn_rate, discount, epsilon, min_eps, episodes):
    # Discretize the state space
    num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize the Q-table
    Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    # Epsilon reduction
    epsilon_decay = (epsilon - min_eps) / episodes

    reward_list = []
    total_reward_list = []

    for i in range(episodes):
        done = False
        total_reward = 0
        state = env.reset()

        # Convert the continuous state to discrete state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done:
            # Do not render the environment
            # Epsilon-greedy action selection
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Take the action and observe the next state and reward
            next_state, reward, done, info = env.step(action)

            # Convert the next state to discrete
            next_state_adj = (next_state - env.observation_space.low) * np.array([10, 100])
            next_state_adj = np.round(next_state_adj, 0).astype(int)

            # Update Q-value
            if done and next_state[0] >= 0.5:  # Goal achieved
                Q[state_adj[0], state_adj[1], action] = reward
            else:
                delta = learn_rate * (reward + discount * np.max(Q[next_state_adj[0], next_state_adj[1]]) - 
                                      Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            total_reward += reward
            state_adj = next_state_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= epsilon_decay

        # Collect rewards
        reward_list.append(total_reward)

        # Log progress
        print(f"Episode {i + 1}/{episodes} completed.")

        # Log total rewards every 100 episodes
        if (i + 1) % 100 == 0:
            total_reward = np.sum(reward_list)
            total_reward_list.append(total_reward)
            reward_list = []

    env.close()
    return total_reward_list


# Train the agent
rewards = QLearn(env, learn_rate=0.2, discount=1, epsilon=0.8, min_eps=0, episodes=6000)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward (per 100 episodes)')
plt.title('Reward vs Episodes')
plt.grid()

# Save the plot
save_folder = "./qlearning_plots/"
if not os.path.exists(save_folder):
        os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, "Plot.png"))
plt.close()
