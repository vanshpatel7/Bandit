# Bandit Simulation

This project simulates a **Multi-Armed Bandit** problem to compare the performance of different reinforcement learning algorithms.

## Agents Implemented

1.  **Random Agent**: Picks an arm completely at random.
2.  **Epsilon-Greedy Agent**: Exploits the best arm most of the time but explores randomly with probability $\epsilon$.
3.  **UCB Agent**: Uses the Upper Confidence Bound (UCB1) algorithm to balance exploration and exploitation.
4.  **Thompson Sampling Agent**: Uses Bayesian updates (Beta distribution) to sample expected rewards.

## How to Run

Run the simulation using the controller script:

```bash
python3 controller.py --num_plays 1000
```

### Options
-   `--num_plays`: Number of rounds to simulate (default: 10,000).
-   `--input`: Path to the bandit configuration file (default: `bandits/input/test1.txt`).
-   `--show`: Display plots interactively.

## Output

The script generates two plots comparing the agents:
-   **Cumulative Rewards**: Total reward gathered over time.
-   **Pseudo-Regret**: The difference between the optimal strategy and the agent's performance.
