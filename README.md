# **Handover Optimization in 5G Networks Using Q-Learning 🛜**

This project addresses the challenge of optimizing handover management in 5G networks using Q-Learning, a reinforcement learning technique. In a 5G network, efficient handover management is crucial for ensuring low latency, high throughput, and seamless connectivity. This project models a dynamic grid-based system where an agent learns to navigate and minimize penalties (handover failures) while maximizing the quality of service (QoS) metrics.

Additionally, the project includes a separate Q-Learning implementation (`Q_learning.py`) to benchmark and compare the handover optimization results with classical reinforcement learning tasks, demonstrating the model's versatility and effectiveness.

<br>

> Associated Video Link: [Click here](https://youtu.be/wCoyhqpwW5o)
> <br>
> <br>
> Report PDF File: [Click here](Gr29EC431_Handover_Optimization_in_5G_Networks_Using_Q_Learning.pdf)
> 
<br>

🧾 **Course:** EC431 - 5G Communication and Network

👤 **Faculty:** Dr. Bhupendra Kumar

<br>

## **📚 Table of Contents**

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Setup and Installation](#setup-and-installation)  
4. [How It Works](#how-it-works)  
5. [Modules](#modules)  
6. [Usage](#usage)  
7. [Visualization](#visualization)  
8. [Key Features](#key-features)  
9. [Future Work](#future-work)  
10. [Contributors](#contributors)  

<br>

## **✨ Overview**

In a 5G network, handovers are a critical operation where user equipment (UE) switches from one cell tower (antenna) to another to maintain connectivity. Improper handovers can result in dropped connections and degraded QoS. This project models this scenario using reinforcement learning:  

- **5G Antenna Cells**: Represented as grid cells with overlapping signal ranges.  
- **Rewards and Penalties**: Based on the agent’s decisions regarding handovers.  
- **Reinforcement Learning**: The Q-Learning algorithm trains an agent to optimize the handover process dynamically.  

The project also includes a **benchmarking module** (`Q_learning.py`) to compare the results of the handover optimization system with a simpler, well-known reinforcement learning task, such as `MountainCar`.

<br>

## **🔖 Project Structure**

```plaintext
.
├── agent.py                    # Implements the Q-Learning algorithm for handover decisions.
├── gridworld.py                # Models the 5G grid environment with antennas.
├── main.py                     # Main script to train and evaluate the handover optimization agent.
├── plotter.py                  # Generates plots for performance analysis.
├── Q_learning.py               # A standalone implementation of Q-Learning on a classic task for comparison.
├── requirements.txt            # Lists the dependencies required for the project.
├── metrics_plots/              # Stores the training metrics for the handover optimization system.
│   ├── rewards_per_episode.png # Tracks rewards earned per episode.
│   ├── handovers_per_episode.png # Tracks the number of handovers.
│   ├── cumulative_success.png  # Tracks the cumulative success rate.
│   ├── steps_taken_per_episode.png # Tracks the steps taken per episode.
│   ├── cumulative_rewards.png  # Tracks the cumulative rewards over episodes.
│   ├── histogram_of_handovers.png # Displays the distribution of handovers.
│   └── epsilon_decay.png       # Shows the epsilon decay over episodes.
├── qlearning_plots/            # Stores plots for Q-Learning experiments.
│   └── plot.png                # Summarizes Q-Learning performance for standalone tasks.

```

<br>

## **🛠️ Setup and Installation**

### Prerequisites

- Python 3.8+  
- Required libraries: `numpy`, `matplotlib`, `gym`, `pyglet`

### Installation Steps

1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:  
   ```bash
   python main.py
   ```

4. Run the benchmarking script (optional):  
   ```bash
   python Q_learning.py
   ```

<br>

## **❓ How It Works**

### Environment Setup

- **5G Grid Model**:  
  - A grid where each cell represents a 5G antenna range.  
  - Antennas have overlapping coverage zones.

- **Agent Movement**:  
  - The agent navigates through the grid while switching between antenna cells.  
  - Penalties are applied for unnecessary handovers to incentivize efficient behavior.

- **Dynamic States**:  
  - Signal strength varies dynamically.  
  - The agent must adapt to changing conditions to maintain optimal connectivity.

### Training with Q-Learning

- **Q-Values**: Represent the expected rewards for each action in a given state.  
- **Rewards**: Encourages successful handovers and connectivity.  
- **Penalties**: Applied for failed or unnecessary handovers.  
- **Epsilon Decay**: Balances exploration and exploitation during training.

### Benchmarking with `Q_learning.py`

- Implements Q-Learning for the `MountainCar` environment to evaluate the algorithm's adaptability and performance in a simpler, static setup.  
- Generates comparative performance insights, highlighting the unique challenges of dynamic 5G environments.

<br>

## **🛡️ Modules**

### 1. **`agent.py`**
Defines the Q-Learning agent responsible for:  
- Selecting actions based on epsilon-greedy policy.  
- Updating Q-values dynamically.  
- Handling antenna selection and handover decisions.

### 2. **`gridworld.py`**
Models the 5G network environment, including:  
- Antenna configurations and signal ranges.  
- State transitions and movement rules.  
- Reward and penalty mechanisms.

### 3. **`main.py`**
- Initializes the environment and agent.  
- Runs the training process.  
- Logs metrics for analysis.

### 4. **`plotter.py`**
Generates plots for:  
- Episode rewards.  
- Handover efficiency.  
- Steps taken per episode.  
- Epsilon decay and other metrics.

### 5. **`Q_learning.py`**
A standalone Q-Learning implementation for benchmarking on tasks like `MountainCar`.

<br>

## **💻 Usage**

1. **Run the Training Script**:  
   ```bash
   python main.py
   ```

2. **Run the Benchmarking Script** (optional):  
   ```bash
   python Q_learning.py
   ```

3. **View Plots**:  
   Analyze the generated plots in the `metrics_plots/` and `qlearning_plots/` folders for performance insights.

<br>

## **🖼️ Visualization**

Generated plots include:  

- **Rewards per Episode**: Tracks agent performance improvements.  
- **Handovers per Episode**: Measures unnecessary handovers.  
- **Steps Taken per Episode**: Monitors navigation efficiency.  
- **Cumulative Success**: Tracks overall success rate across episodes.  
- **Epsilon Decay**: Observes the exploration-exploitation strategy.  

From `Q_learning.py`:  
- A single plot showing the reward progression for a classical RL task, aiding in comparing results with the 5G optimization system.

<br>

## **⚡ Key Features**

1. **Dynamic Handover Management**  
   - Models realistic 5G network scenarios with overlapping antenna coverage.

2. **Optimized Performance**  
   - Uses Q-Learning to minimize unnecessary handovers and optimize connectivity.

3. **Benchmarking Flexibility**  
   - Includes a classical RL task (`MountainCar`) for performance comparison.

4. **Detailed Visualization**  
   - Comprehensive plots for evaluating the agent’s performance.

5. **Reinforcement Learning Adaptability**  
   - Explores RL's application for both dynamic 5G problems and static environments.

<br>

## **⏭️ Future Work**

- **Multi-agent extension** for collaborative tasks.
- **Integration of DQN** for deep reinforcement learning.
- **Obstacle dynamics** to simulate real-world grid environments.

<br>

## **🙋‍♂️ Contributors**

- **Sanidhya Kumar**: [GitHub Profile](https://github.com/notsanidhyak)
- **Samanway Maji**: [GitHub Profile](https://github.com/samhere19)
- **Riya**: [GitHub Profile](https://github.com/notdatriya)
- **Shivang Bhargava**: [GitHub Profile](https://github.com/shivang0203)
- **Anamika Sadh**: [GitHub Profile](https://github.com/anasadh)
  
> For queries, reach out via [email](mailto:sanidhyakumar.31@gmail.com).

<br>

Feel free to reach out for further queries or discussions. Happy coding! ❤️

--- 
