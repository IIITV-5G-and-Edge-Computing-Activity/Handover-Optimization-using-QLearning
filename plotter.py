import matplotlib.pyplot as plt
import numpy as np
import os

# Define function to plot metrics and save them in a folder
def plot_metrics(Episodes, Rewards, Handovers, steps_taken, eps, eps_decaying, save_folder, sample_rate=100):
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Sample the data for plotting
    sampled_indices = range(0, Episodes, sample_rate)
    sampled_episodes = np.array(sampled_indices)
    sampled_rewards = np.array(Rewards)[sampled_indices]
    sampled_handovers = np.array(Handovers)[sampled_indices]
    sampled_steps = np.array(steps_taken)[sampled_indices]

    cumulative_rewards = np.cumsum(Rewards)
    moving_avg_rewards = [np.mean(Rewards[max(0, i-10):i+1]) for i in range(len(Rewards))]
    success_rate = [1 if reward > 0 else 0 for reward in Rewards]
    success_cumulative = np.cumsum(success_rate)
    sampled_cumulative_rewards = cumulative_rewards[sampled_indices]
    sampled_moving_avg_rewards = np.array(moving_avg_rewards)[sampled_indices]

    # Plot Rewards over Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_rewards, label='Rewards per Episode')
    plt.plot(sampled_episodes, sampled_moving_avg_rewards, label='Moving Average of Rewards (window=10)', linestyle='--')
    plt.title("Rewards vs. Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Rewards_vs_Episodes_Sampled.png"))
    plt.close()

    # Plot Handovers over Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_handovers, label='Handovers per Episode', color='orange')
    plt.title("Handovers vs. Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Handovers")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Handovers_vs_Episodes_Sampled.png"))
    plt.close()

    # Plot Success Rate
    plt.figure(figsize=(12, 6))
    plt.plot(range(Episodes), success_cumulative, label='Cumulative Success', color='red')
    plt.title("Cumulative Success Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Successes")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Cumulative_Success.png"))
    plt.close()

    # Plot Steps Taken per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_steps, label='Steps per Episode', color='green')
    plt.title("Steps Taken per Episode (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Steps_vs_Episodes_Sampled.png"))
    plt.close()

    # Plot Cumulative Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_cumulative_rewards, label='Cumulative Rewards', color='purple')
    plt.title("Cumulative Rewards over Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Cumulative_Rewards_Sampled.png"))
    plt.close()

    # Plot Histogram of Handovers
    plt.figure(figsize=(12, 6))
    plt.hist(Handovers, bins=50, color='skyblue', edgecolor='black')  # Increased number of bins
    plt.title("Histogram of Handovers")
    plt.xlabel("Number of Handovers")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_folder, "Histogram_of_Handovers.png"))
    plt.close()

    # Plot Epsilon Decay
    eps_values = [max(0, eps - i * eps_decaying) for i in range(Episodes)]
    sampled_eps_values = np.array(eps_values)[sampled_indices]
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_eps_values, label='Epsilon Decay', color='brown')
    plt.title("Epsilon Decay over Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Epsilon_Decay_Sampled.png"))
    plt.close()
