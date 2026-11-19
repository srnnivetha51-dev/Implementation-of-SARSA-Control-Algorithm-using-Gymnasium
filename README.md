# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---



## Problem Statement

To develop a Reinforcement Learning agent using the SARSA control algorithm that learns the optimal action-selection policy through interaction with the FrozenLake environment. The agent learns through trial and error to reach the goal state while avoiding holes and maximizing the cumulative reward.

---

## Software Requirements

- **Programming Language:** Python
- **Platform:** Google Colab / Jupyter Notebook
- **Libraries:** Gymnasium, NumPy, Matplotlib




## Environment Description



## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---




## Algorithm

1. Start the program.
2. Import the required libraries such as Gymnasium, NumPy and Matplotlib.
3. Create the FrozenLake environment using `gym.make()`.
4. Initialize the learning parameters such as alpha, gamma and epsilon.
5. Initialize the Q-table with zeros.
6. Repeat the training process for 10,000 episodes.
7. Reset the environment and obtain the initial state.
8. Select an action using the epsilon-greedy policy.
9. Perform the selected action and observe the next state and reward.
10. Check whether the episode is terminated or truncated.
11. If the episode is not finished, select the next action using epsilon-greedy.
12. Update the Q-value using the SARSA update rule.
13. If the episode is finished, update the Q-value using the terminal reward.
14. Repeat until the episode ends or reaches the maximum number of steps.
15. Store the total reward of each episode.
16. Reduce epsilon using the epsilon decay factor.
17. Calculate the state-value function using the maximum Q-value of each state.
18. Find the best action for each state and display the learned policy.
19. Calculate the average reward over the last 1000 episodes.
20. Plot the SARSA learning curve using Matplotlib.
21. Close the environment.
22. Stop the program.
## Python Program

```

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1", is_slippery=True)
num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995
state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))
def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[state])
episode_rewards = []
for episode in range(num_episodes):

    state, info = env.reset()
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if not done:
            next_action = epsilon_greedy_action(next_state, epsilon)

            # SARSA Update Rule
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

        else:
            # Terminal state update
            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

        total_reward += reward

        if done:
            break

        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------
print("NAME: S R NIVEDHITHA")
print("REG NO: 212225240102")
print("\nFinal Q-table:")
print(np.round(Q, 3))

# Calculate state values
state_values = np.max(Q, axis=1)

# Find best action for each state
learned_policy = np.argmax(Q, axis=1)

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", average_reward)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")

plt.grid(True)
plt.show()

env.close()




```
---

## Output


Final Q-table:

<img width="159" height="200" alt="Screenshot 2026-11-19 141352" src="https://github.com/user-attachments/assets/bb888d4f-139e-447a-a694-a73a4a44fa2b" />




Estimated State-Value Function:

<img width="178" height="56" alt="Screenshot 2026-11-19 141356" src="https://github.com/user-attachments/assets/40f1c6e0-c899-4fc8-8d3d-be9d8a1d24d3" />




Learned Policy:


<img width="234" height="78" alt="Screenshot 2026-11-19 141359" src="https://github.com/user-attachments/assets/7c982c7b-c7ff-4ab6-a873-12a140a33a03" />


Average reward over last 1000 episodes: 



<img width="455" height="261" alt="Screenshot 2026-11-19 141404" src="https://github.com/user-attachments/assets/b3281d73-9d2f-4350-a799-1d33a3aec735" />

---

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake environment. The agent learned action-value estimates through interaction with the environment using the epsilon-greedy policy. The final Q-table, estimated state-value function, learned policy, average reward and learning curve were generated successfully.

---

## Inference

- SARSA is an on-policy Reinforcement Learning algorithm.
- The agent learns through trial and error.
- The epsilon-greedy strategy balances exploration and exploitation.
- The Q-table stores the learned action values for each state.
- Epsilon decay gradually reduces random exploration during training.
- The learning curve helps analyze the agent's learning progress.
- The learned policy helps the agent select actions to reach the goal while avoiding holes.



```
---

