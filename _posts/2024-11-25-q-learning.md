## Optimizing Inventory Management with Q-Learning and the (s,S) Policy
Inventory management is one of the most critical components of supply chain operations. From managing costs to handling fluctuating demand, businesses continuously search for strategies to optimize stock levels. In this article, we compare two approaches to inventory control — Q-Learning and the (s,S) Policy — with a Random Policy serving as a reference point for comparison.

### 1. Introduction
Efficient inventory management involves balancing stock levels to minimize costs while meeting demand. Traditional methods like the (s,S) Policy rely on fixed thresholds to determine when and how much to restock. On the other hand, modern approaches like Q-Learning leverage reinforcement learning to dynamically adapt decisions to observed demand patterns and restocking costs.

The objective of this analysis is to evaluate:

1. The dynamic adaptability of Q-Learning.
2. The performance of the traditional (s,S) Policy.
3. How these strategies compare against a Random Policy, used as a control.

### 2. Demand Patterns
The simulated demand data includes both a seasonal component and a random component. The seasonal component captures predictable fluctuations such as holidays or production cycles, while the random component accounts for unexpected demand shocks, reflecting real-world uncertainties.

![Demand Data Generating Process](https://github.com/eacalatayudm/eacalatayudm.github.io/blob/main/_posts/images/demanddgp.png?raw=true "Demand Data Generating Process")

The demand graph below illustrates the interplay of seasonal and random fluctuations:

![Demand Series Graph](https://github.com/eacalatayudm/eacalatayudm.github.io/blob/main/_posts/images/demand_seasonality_graph.png?raw=true "Demand Series Graph")

### 3. The (s,S) Policy
The (s,S) policy is a widely used inventory control strategy, where restocking occurs whenever the inventory level falls below a lower threshold, s. When restocked, the stock is replenished up to an upper threshold, S.

Introduced in the mid-20th century, the (s,S) policy has remained a cornerstone of inventory management due to its simplicity and proven effectiveness in structured and predictable environments. Its rule-based framework makes it especially well-suited for scenarios where demand variability is low or moderately predictable.

However, this rigidity can become a limitation in dynamic contexts, where rapid demand fluctuations or supply chain disruptions necessitate more adaptive approaches.

The graph below illustrates the stock levels over time under the (s,S) policy:

![(s,S) Example Graph](https://github.com/eacalatayudm/eacalatayudm.github.io/blob/main/_posts/images/s_S_example.png?raw=true "(s,S) Example Graph")

The system triggers replenishment when inventory dips below the threshold s, restoring it to S at the start of the next period of time. This structured approach minimizes costs associated with stockouts and overstocking but may struggle to adjust dynamically to unforeseen demand patterns.

### 4. Q-Learning for Inventory Management
Q-Learning, a reinforcement learning algorithm, dynamically learns the optimal actions for restocking. By evaluating restocking costs, stockout penalties, and future demand, it adapts to minimize overall costs while handling fluctuations in demand. This algorithm represents a leap toward intelligent inventory systems, capable of adapting to ever-changing demand patterns. By learning from experience, it builds dynamic policies that go beyond static thresholds.

#### Training Convergence
The Q-Learning model was trained for 80,000 episodes. The graph below illustrates the evolution of rewards during training:

![Q-Learning Convergence](https://github.com/eacalatayudm/eacalatayudm.github.io/blob/main/_posts/images/q_learning_convergence.png?raw=true "Q-Learning Convergence")

Over time, the algorithm learns to make decisions that consistently improve rewards, as seen by the upward trend.

### 5. Comparing Policies: Q-Learning, (s,S), and Random
After training the Q-Learning model, we tested it alongside the (s,S) policy and Random Policy across 20,000 episodes. The focus of this comparison is to evaluate:

- The flexibility and effectiveness of Q-Learning.
- The reliability of the traditional (s,S) policy.
- A Random Policy as a control.

### 6. Reward Comparison
The graph below compares the rewards achieved by each policy:

![Reward Comparison](https://github.com/eacalatayudm/eacalatayudm.github.io/blob/main/_posts/images/comparison.png?raw=true "Reward Comparison")

- Q-Learning (Green) outperforms other approaches with a mean reward of -260.41, showcasing its ability to dynamically adjust to demand.
- The (s,S) Policy (Blue) achieves a mean reward of -308.23, reflecting its reliability in structured inventory control.
- The Random Policy (Red) lags significantly, with a mean reward of -429.91, underscoring the value of structured decision-making.

Q-Learning’s adaptability enables it to minimize costs in environments with high variability. However, its computational demands and the need for extensive training episodes make it less suited for smaller-scale operations or highly predictable demand.

### 7. Conclusion
This experiment highlights the effectiveness of Q-Learning in inventory management, offering flexibility and adaptability in dynamic environments. While traditional methods like the (s,S) Policy remain reliable, reinforcement learning provides a modern alternative for optimizing supply chains.

### 8. Code for Replication
If you’re interested in replicating this experiment or modifying it for your use case, here’s the complete Python code. It includes all the key components, from demand simulation to policy evaluation and comparison:

```python
# 0. Libraries
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import random
from random import normalvariate
import math as math
from stockpyl.finite_horizon import finite_horizon_dp
```

```python
# 1. Parameters
T = 24
p = 3
K = 1
c = 1.5
h = 1
h_terminal = 1
p_terminal = 3
gamma = 1.0
max_stock = 10  # Maximum stock allowed
num_train_episodes = 80000  # Number of train episodes
num_test_episodes = 20000  # Number of test episodes
alpha, gamma, epsilon = 0.01, 1, 0.1  #Parameters for Q-Learning
actions = [i for i in range(9)]
states = [i for i in range(24)]
state_action_space = [(state, action) for state in states for action in actions]
Q = {id: 0 for id in state_action_space}
```

```python
# 2. Functions
# Cost function
def cost_function(r):
    if r > 0:
        return K + c * r
    else:
        return 0

# Data Generating Process for the Demand
def dgp_demand(i):
    return max(0, int(5 * math.cos((i+1) * 2 * math.pi / 24) + 6 + random.gauss(0, 1)))

# Epsilon-Greedy Policy for Q-Learning
def epsilon_greedy(Q, state):
    if np.random.rand() < epsilon:
        return random.choice(actions)
    else:
        return np.argmax([Q[state, i] for i in actions])
```

```python
# 3. Demand Graph
demand = [dgp_demand(i) for i in range(T)]
plt.figure(figsize=(10, 6))
plt.plot(range(0, T), demand, 'bo-', label='Demand Series')
plt.plot(range(0, T), [5 * math.cos(i * 2 * math.pi / 23) + 6 for i in range(T)], 'g--', label='Seasonal Component')
plt.title('Demand Series with Seasonal Component')
plt.xlabel('Period')
plt.ylabel('Demand')
plt.xticks(range(24), labels=[f'{i+1}' for i in range(24)], rotation=0)
plt.grid(True)
plt.legend()
plt.savefig("demand_seasonality_graph.png", dpi=500)
plt.show()
```

```python
# 4. Random Policy Generation
def random_policy(stock, period):
    """Selects an action at random"""
    return random.choice(actions)
def print_episode_details(taken_actions, demands, final_stocks, rewards):
    print("\nExample of Random Policy Details for an Episode:")
    for i in range(T):
        print(f"Period {i + 1}: Action taken = {taken_actions[i]}, Demand = {demands[i]}, "
              f"Final Stock = {final_stocks[i]}, Reward = {rewards[i]:.2f}")
# Example of random policy for a single episode
def example_random_policy():
    stock = 0
    taken_actions = []
    demands = []
    final_stocks = []
    rewards = []

    for period in range(T):
        action = random_policy(stock, period)  # Using random policy here
        max_action = max(0, max_stock - stock)
        action = min(action, max_action)

        demand = dgp_demand(period)

        # Calculate costs and rewards
        income = 0
        cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
        profit = income - cost

        # Update final stock
        final_stock = max(0, action + stock - demand)

        # Store episode details
        taken_actions.append(action)
        demands.append(demand)
        final_stocks.append(final_stock)
        rewards.append(profit)

        stock = final_stock  # Update stock for the next period

    # Print episode details
    print_episode_details(taken_actions, demands, final_stocks, rewards)
# Call the function to show the exemplary episode with random policy
example_random_policy()
```

```python
# 5. Q-Learning Policy Training
# Store rewards during training
episode_rewards = []
actions_per_episode = {}  # To store actions of key episodes

print("Training the Model with Q-Learning:")
for episode in range(1, num_train_episodes + 1):
    state = 0
    done = False
    period_rewards = []
    stock = 0
    episode_actions = []  # Store actions for this episode

    while not done:
        max_action = max(0, max_stock - stock)
        action = epsilon_greedy(Q, state)
        action = min(action, max_action)

        demand = dgp_demand(state)
        next_state = state + 1

        income = 0
        cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
        profit = income - cost
        period_rewards.append(profit)
        reward = income - cost

        episode_actions.append(action)  # Store taken action

        if next_state == T:
            done = True
            Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
        else:
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state, np.argmax([Q[next_state, i] for i in actions])] - Q[state, action]
            )

        state = next_state
        stock = max(0, action + stock - demand)

    # Store total reward for the episode
    episode_rewards.append(sum(period_rewards))

    # Print actions at the start and every 2000 episodes
    if episode == 1 or episode % 2000 == 0:
        actions_per_episode[episode] = episode_actions
        print(f"Actions in episode {episode}: {episode_actions}")
# Reward Evolution Graph
plt.figure(figsize=(14, 6))
plt.plot(episode_rewards, label='Reward per Episode', color='mediumblue')
plt.title('Reward Evolution During Training')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward per Episode')
plt.grid(True)
plt.ylim([min(episode_rewards) - 10, max(episode_rewards) + 10])
plt.savefig("q_learning_convergence.png", dpi=500)  # Save the graph as a PNG file
plt.show()
```

```python
# 6. Generation of (s,S) Policy
# Simulate demand to calculate mean and standard deviation
n_years = 1000000
annual_means = []
annual_std_devs = []

for _ in range(n_years):
    annual_demand = [dgp_demand(i) for i in range(T)]
    annual_means.append(np.mean(annual_demand))
    annual_std_devs.append(np.std(annual_demand))

mean_of_means = np.mean(annual_means)
mean_of_std_devs = np.mean(annual_std_devs)
mu = mean_of_means
sigma = mean_of_std_devs
# Generate (s,S) policy
s, S, cost, _, _, _ = finite_horizon_dp(
    num_periods=T, holding_cost=h, stockout_cost=p, terminal_holding_cost=h_terminal,
    terminal_stockout_cost=p_terminal, purchase_cost=c, fixed_cost=K,
    demand_mean=mu, demand_sd=sigma, discount_factor=gamma
)
s = s[1:]
S = S[1:]
print(f"s in each period: {s}")
print(f"S in each period: {[int(x) for x in S]}")
# (s,S) policy based on restocking when stock is <= s
def apply_sS_policy(stock, period):
    if stock <= s[period]:
        return int(S[period] - stock)  # Restock up to S_t
    return 0  # No restock
# Simulate an episode using (s,S) policy
T = 24
stock = 0  # Initial stock
sS_actions = []
simulated_demand = [dgp_demand(i) for i in range(T)]

period_rewards_sS = []

for period in range(T):
    action = apply_sS_policy(stock, period)
    demand = simulated_demand[period]
    new_stock = max(0, action + stock - demand)
    
    # Calculate income and cost
    income = 0
    cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
    profit = income - cost
    
    # Save results
    sS_actions.append((action, demand, new_stock, profit))
    period_rewards_sS.append(profit)
    
    stock = new_stock

# Print actions taken and rewards during the episode
print(f"s in each period: {s}")
print(f"S in each period: {[int(x) for x in S]}")
print("\nExample of (s,S) Policy Details for an Episode:")
for i, (action, demand, new_stock, profit) in enumerate(sS_actions):
    print(f"Period {i+1}: Action taken = {action}, Demand = {demand}, New Stock = {new_stock}, Reward = {profit:.2f}")

# Example graph for (s,S) policy
s_example = 5
S_example = 8

# Define time axis
t = np.linspace(0, 4.5, 500)

# Create a list to store stock levels
stock = np.zeros_like(t)

# Simulate stock 'sawtooth' behavior
for i in range(len(t)):
    if int(t[i]) == 2:
        stock[i] = max(S_example - (S_example - 4) * (t[i] % 1), 4)  # Starts at 4
    elif int(t[i]) == 3:
        stock[i] = max(S_example - (S_example - 3) * (t[i] % 1), 3)  # Starts at 3
    elif t[i] % 1 == 0:
        stock[i] = S_example  # Restock to S when t is an integer
    else:
        stock[i] = max(S_example - (S_example - s_example) * (t[i] % 1), s_example)  # Normal behavior

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, stock, label="Stock", color='green', linewidth=2)

# Horizontal lines for s and S
plt.axhline(y=s_example, color='red', linestyle='--', label="s", linewidth=1.5)
plt.axhline(y=S_example, color='blue', linestyle='--', label="S", linewidth=1.5)

# Adjust horizontal axis to start at 0 and end at 4.5
plt.xlim(0, 4.5)

# Adjust x-axis ticks
plt.xticks([1, 2, 3, 4])

# Labels and legend
plt.xlabel("Time (t)")
plt.ylabel("Stock Level")
plt.title("Example of (s,S) Policy")
plt.yticks([])  # Remove vertical axis labels
plt.legend()
plt.grid(True)
plt.savefig("s_S_example.png", dpi=500)
plt.show()
```

```python
# 7. Testing Policies: Q-Learning, (s,S), and Random
# Generate common demand sequence
test_demand = []
for episode in range(num_test_episodes):
    episode_demand = [dgp_demand(i) for i in range(T)]
    test_demand.append(episode_demand)
# Test Q-learning policy
q_learning_rewards_test = []

for episode in range(num_test_episodes):
    state = 0
    done = False
    period_rewards_test = []
    stock = 0
    
    for period in range(T):
        action = np.argmax([Q[state, i] for i in actions])
        max_action = max(0, max_stock - stock)
        action = min(action, max_action)

        demand = test_demand[episode][period]
        next_state = state + 1

        income = 0
        cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
        profit = income - cost
        period_rewards_test.append(profit)

        state = next_state
        stock = max(0, action + stock - demand)

        if next_state == T:
            done = True

    q_learning_rewards_test.append(sum(period_rewards_test))
# Test (s,S) policy
sS_rewards_test = []

for episode in range(num_test_episodes):
    state = 0
    done = False
    period_rewards_test = []
    stock = 0
    
    for period in range(T):
        action = apply_sS_policy(stock, state)

        demand = test_demand[episode][period]
        next_state = state + 1

        income = 0
        total_cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
        profit = income - total_cost
        period_rewards_test.append(profit)

        state = next_state
        stock = max(0, action + stock - demand)
        stock = min(stock, max_stock)

        if next_state == T:
            done = True

    sS_rewards_test.append(sum(period_rewards_test))
# Test random policy
random_rewards_test = []

for episode in range(num_test_episodes):
    state = 0
    done = False
    period_rewards_test = []
    stock = 0
    
    for period in range(T):
        action = random_policy(stock, state)

        demand = test_demand[episode][period]
        next_state = state + 1

        income = 0
        total_cost = cost_function(action) + h * max(0, action + stock - demand) + p * max(0, demand - action - stock)
        profit = income - total_cost
        period_rewards_test.append(profit)

        state = next_state
        stock = max(0, action + stock - demand)
        stock = min(stock, max_stock)

        if next_state == T:
            done = True

    random_rewards_test.append(sum(period_rewards_test))
```

```python
# 8. Comparison of Results
average_q_reward = np.mean(q_learning_rewards_test)
average_sS_reward = np.mean(sS_rewards_test)
average_random_reward = np.mean(random_rewards_test)
# Plot rewards for each policy
plt.figure(figsize=(17, 6))
plt.plot(q_learning_rewards_test, label=f'RL Policy (mean: {average_q_reward:.2f})', color='green', alpha=0.6)
plt.plot(sS_rewards_test, label=f'(s,S) Policy (mean: {average_sS_reward:.2f})', color='blue', alpha=0.6)
plt.plot(random_rewards_test, label=f'Random Policy (mean: {average_random_reward:.2f})', color='red', alpha=0.6)

# Add lines for the means of each policy
plt.axhline(average_q_reward, color='green', linestyle='-', alpha=1)
plt.axhline(average_sS_reward, color='blue', linestyle='-', alpha=1)
plt.axhline(average_random_reward, color='red', linestyle='-', alpha=1)

# Graph configuration
plt.title('Comparison of Rewards: RL, (s,S), and Random Policies')
plt.xlabel('Number of Test Episodes')
plt.ylabel('Reward per Test Episode')
plt.xlim(0, num_test_episodes)
plt.legend(loc='best')
plt.grid(True)
plt.savefig("comparison.png", dpi=450)
plt.show()
```
