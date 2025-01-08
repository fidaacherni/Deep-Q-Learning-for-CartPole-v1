Deep Q-Learning for CartPole-v1 🏋️‍♂️🎯
📌 Overview
This project is an implementation of Deep Q-Networks (DQN) for solving the CartPole-v1 environment using JAX. The goal is to train an AI agent to balance a pole on a moving cart by applying reinforcement learning techniques.

The project explores:
Vanilla DQN with experience replay and a target network.
Double DQN to mitigate Q-value overestimation.
Replay Buffer (FIFO) for efficient training.
Epsilon-Greedy Exploration for balancing exploitation and exploration.

🎯 Objectives
Implement DQN with JAX for training an optimal agent in CartPole-v1.
Develop a replay buffer (FIFO) to store experiences and improve sample efficiency.
Incorporate a target network to stabilize learning.
Implement Double DQN to correct Q-value overestimation.
Optimize hyperparameters to reach a score of 400+ after 250,000 steps.

🔥 Algorithms Implemented
1️⃣ Deep Q-Network (DQN)
Uses a fully connected neural network (MLP) to approximate the Q-function.
Implements experience replay for better sample efficiency.
Uses a target network to stabilize training.
2️⃣ Double DQN
Reduces Q-value overestimation bias by using separate networks for action selection and value estimation.
Modifies the Bellman target to select actions from the main network and estimate values from the target network.
3️⃣ Experience Replay Buffer (FIFO)
Stores transitions (s, a, r, s', done) in a fixed-size buffer.
Enables random sampling to break temporal correlations in training data.
Implements efficient memory management to replace old experiences.

🏋️ CartPole-v1 Environment
The CartPole-v1 environment is a classic control problem where:
The state space consists of (x, ẋ, θ, θ̇):
x: Cart position
ẋ: Cart velocity
θ: Pole angle
θ̇: Pole angular velocity
The action space is discrete (left or right).
The goal is to keep the pole balanced as long as possible.

🧠 Key Components
📌 Replay Buffer (buffer.py)
Implements a FIFO replay buffer using JAX.
Functions:
add_transition(): Adds a new experience to the buffer.
sample_transition(): Randomly samples experiences for training.
📌 Deep Q-Network (model.py)
Implements DQN architecture with hidden layers.
Functions:
select_action(): Uses ε-greedy policy for exploration.
compute_loss(): Computes DQN loss using Bellman equation.
update_target(): Periodically updates the target network.
📌 Training (trainer.py)
Defines DQN training pipeline.
Functions:
agent_update_step(): Executes one step of training and replay buffer update.
eval_agent(): Evaluates the agent using a greedy policy.
agent_iteration(): Runs multiple training and evaluation steps.

📈 Results & Performance
Vanilla DQN reaches an average score of 400+ after 250,000 steps.
Double DQN achieves better stability and higher returns.
Replay Buffer improves sample efficiency and prevents catastrophic forgetting.

🏆 Final Thoughts
Deep Q-Networks are effective for solving CartPole-v1, but Double DQN provides better stability.
Experience replay helps prevent catastrophic forgetting and improves convergence.
Future work: Implement Prioritized Experience Replay (PER) and Dueling DQN for further improvements.
