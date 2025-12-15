# run_bandits.py
# Compare several bandit agents on a Bernoulli multi-armed bandit.
# "Student-like" style: small steps, clear names, comments, optional debug prints.

import os
import random
import argparse
from typing import Dict, List, Tuple, Type
import matplotlib.pyplot as plt

# Local agents
from randomAgent import randomAgent
from GreedyAgent import epsGreedyAgent
from UCBAgent import UCBAgent
from ThompsonAgent import thompsonAgent

# Map CLI names -> classes
AGENTS_MAP: Dict[str, Type] = {
    "randomAgent": randomAgent,
    "epsGreedyAgent": epsGreedyAgent,
    "UCBAgent": UCBAgent,
    "thompsonAgent": thompsonAgent,
}

class Bandit:
    """
    Simple Bernoulli bandit.
    Each arm i pays 1 with probability p[i], else 0.
    """

    def __init__(self, probs: List[float]):
        self.arms: List[float] = probs  # store success probabilities per arm

    @classmethod
    def from_file(cls, filepath: str) -> "Bandit":
        """
        Expected file format:
        - line 0: header or count (ignored)
        - lines 1..: one float per line (probability for each arm)
        """
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        # Skip the first line (header)
        probs = [float(x) for x in lines[1:]]
        return cls(probs)

    def pull_arm(self, arm_idx: int) -> int:
        """Return 1 with probability p, else 0."""
        p = self.arms[arm_idx]
        return 1 if random.random() <= p else 0

    def getNumArms(self) -> int:
        return len(self.arms)

    def getMaxExpectedReward(self) -> float:
        """Best possible expected reward among all arms."""
        return max(self.arms)


def run_experiment(
    bandit: Bandit,
    agent_class: Type,
    num_plays: int,
    debug: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Run a single pass of num_plays using the given agent.
    Returns:
      cumulative_rewards[i] = sum of realized rewards up to step i
      regrets[i]            = cumulative pseudo-regret up to step i
                              (best expected reward - expected reward of chosen arm)
    """
    agent = agent_class()  # keep same API as provided agents
    history: List[Tuple[int, float]] = []

    cumulative_rewards: List[float] = [0.0] * num_plays
    regrets: List[float] = [0.0] * num_plays

    best_expected = bandit.getMaxExpectedReward()

    total_reward = 0.0
    total_regret = 0.0

    for t in range(num_plays):
        # Agent picks an arm using its internal policy
        arm_idx = agent.recommendArm(bandit, history)

        # Pull the arm and observe reward (0/1)
        reward = bandit.pull_arm(arm_idx)

        # Track realized cumulative reward
        total_reward += reward
        cumulative_rewards[t] = total_reward

        # Pseudo-regret: gap between best expected and chosen arm's expected
        # (This is not realized regret; it's expectation-based per step.)
        step_regret = best_expected - bandit.arms[arm_idx]
        total_regret += step_regret
        regrets[t] = total_regret

        # Let agent update via history (keeps same calling pattern as your code)
        history.append((arm_idx, reward))

        if debug and (t < 10 or (t + 1) % max(1, num_plays // 10) == 0):
            print(
                f"[t={t+1}] arm={arm_idx} reward={reward} "
                f"cum_reward={total_reward:.0f} step_regret={step_regret:.4f} cum_regret={total_regret:.2f}"
            )

    return cumulative_rewards, regrets


def plot_curves(
    curves: Dict[str, List[float]],
    num_plays: int,
    title: str,
    ylabel: str,
    out_path: str,
    show: bool,
) -> None:
    """
    Helper to plot one figure (either rewards or regrets).
    """
    plt.figure(figsize=(12, 8))
    xs = range(num_plays)

    for name, ys in curves.items():
        plt.plot(xs, ys, label=name)

    plt.xlabel("Number of Pulls")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, num_plays)
    plt.ylim(bottom=0)
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Bandit experiment runner")
    parser.add_argument(
        "--input",
        choices=["bandits/input/test0.txt", "bandits/input/test1.txt"],
        default="bandits/input/test1.txt",
        help="Bandit definition file",
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=10_000,
        help="Number of arm pulls to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (set for reproducibility)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print occasional step info",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (in addition to saving PNGs)",
    )
    args = parser.parse_args()

    # Optional reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # Build bandit from file
    bandit = Bandit.from_file(args.input)
    input_name = os.path.basename(args.input)

    # Run each agent and store results
    rewards_results: Dict[str, List[float]] = {}
    regret_results: Dict[str, List[float]] = {}

    for agent_name, agent_class in AGENTS_MAP.items():
        cum_rewards, regrets = run_experiment(
            bandit=bandit,
            agent_class=agent_class,
            num_plays=args.num_plays,
            debug=args.debug,
        )
        rewards_results[agent_name] = cum_rewards
        regret_results[agent_name] = regrets

    # Plot cumulative rewards
    rewards_png = f"bandit_rewards_comparison_{input_name}.png"
    plot_curves(
        curves=rewards_results,
        num_plays=args.num_plays,
        title=f"Cumulative Rewards by Agent ({input_name})",
        ylabel="Cumulative Reward",
        out_path=rewards_png,
        show=args.show,
    )

    # Plot pseudo-regret
    regret_png = f"bandit_regret_comparison_{input_name}.png"
    plot_curves(
        curves=regret_results,
        num_plays=args.num_plays,
        title=f"Pseudo-Regret by Agent ({input_name})",
        ylabel="Total Pseudo-Regret",
        out_path=regret_png,
        show=args.show,
    )

    # Print final numbers
    print("\n=== Final Totals ===")
    for agent_name, rewards in rewards_results.items():
        print(f"{agent_name:>15} | Final Cumulative Reward: {rewards[-1]:.0f}")
    for agent_name, regrets in regret_results.items():
        print(f"{agent_name:>15} | Final Total Pseudo-Regret: {regrets[-1]:.2f}")

    print(f"\nSaved plots:\n - {rewards_png}\n - {regret_png}")


if __name__ == "__main__":
    main()
