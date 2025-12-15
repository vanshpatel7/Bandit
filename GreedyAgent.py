# epsGreedyAgent.py
# Simple ε-greedy agent for multi-armed bandits.
# Student-style: plain lists, clear names, small steps, optional debug prints.

import random
from typing import List, Optional, Tuple

class epsGreedyAgent:
    def __init__(self, epsilon: float = 0.10, debug: bool = False):
        """
        epsilon: exploration probability (e.g., 0.10 = explore 10% of the time)
        debug:   if True, print occasional internal state
        """
        self.name: str = "Erin the Epsilon Greedy Agent"
        self.epsilon: float = epsilon
        self.debug: bool = debug

        # Per-arm running stats
        self.arm_counts: Optional[List[int]] = None   # how many times we've seen each arm
        self.arm_values: Optional[List[float]] = None # running mean reward for each arm

    def _ensure_init(self, num_arms: int) -> None:
        """Lazy-init arrays once we know bandit size."""
        if self.arm_counts is None or self.arm_values is None:
            self.arm_counts = [0] * num_arms
            self.arm_values = [0.0] * num_arms
            if self.debug:
                print(f"[init] created {num_arms} arms")

    def _ingest_history(self, history: List[Tuple[int, float]]) -> None:
        """
        Update running means using the new (arm, reward) pairs since last call.
        Assumes `history` contains only NEW observations since the previous call.
        """
        for arm_idx, reward in history:
            n_old = self.arm_counts[arm_idx]
            mean_old = self.arm_values[arm_idx]

            n_new = n_old + 1
            mean_new = mean_old + (reward - mean_old) / n_new

            self.arm_counts[arm_idx] = n_new
            self.arm_values[arm_idx] = mean_new

            if self.debug:
                print(f"[update] arm={arm_idx} reward={reward:.3f} "
                      f"count: {n_old}→{n_new} mean: {mean_old:.4f}→{mean_new:.4f}")

    def _argmax(self, xs: List[float]) -> int:
        """Return index of max value (plain Python, stable for student clarity)."""
        best_i = 0
        best_v = xs[0]
        for i, v in enumerate(xs):
            if v > best_v:
                best_i, best_v = i, v
        return best_i

    def recommendArm(self, bandit, history: List[Tuple[int, float]]) -> int:
        """
        Pick an arm via ε-greedy.
        bandit: has bandit.getNumArms()
        history: list of (arm_index, reward) since the last call
        """
        num_arms = bandit.getNumArms()
        self._ensure_init(num_arms)
        self._ingest_history(history)

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            choice = random.randrange(num_arms)
            if self.debug:
                print(f"[choose] explore -> arm {choice}")
            return choice

        # Otherwise exploit the current best mean
        choice = self._argmax(self.arm_values)
        if self.debug:
            print(f"[choose] exploit -> arm {choice} (mean={self.arm_values[choice]:.4f})")
        return choice
