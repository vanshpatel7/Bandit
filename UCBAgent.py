import math
import random
from typing import List, Optional, Tuple

class UCBAgent:
    def __init__(self, c: float = 2.0, debug: bool = False):
        """
        c: exploration parameter (often 2.0 or sqrt(2))
        debug: if True, print occasional internal state
        """
        self.name: str = "Ursula the UCB Agent"
        self.c: float = c
        self.debug: bool = debug

        # Per-arm running stats
        self.arm_counts: Optional[List[int]] = None
        self.arm_values: Optional[List[float]] = None
        self.total_counts: int = 0

    def _ensure_init(self, num_arms: int) -> None:
        if self.arm_counts is None or self.arm_values is None:
            self.arm_counts = [0] * num_arms
            self.arm_values = [0.0] * num_arms
            if self.debug:
                print(f"[init] created {num_arms} arms")

    def _ingest_history(self, history: List[Tuple[int, float]]) -> None:
        """
        Update running means using the new (arm, reward) pairs since last call.
        """
        for arm_idx, reward in history:
            n_old = self.arm_counts[arm_idx]
            mean_old = self.arm_values[arm_idx]

            n_new = n_old + 1
            mean_new = mean_old + (reward - mean_old) / n_new

            self.arm_counts[arm_idx] = n_new
            self.arm_values[arm_idx] = mean_new
            self.total_counts += 1

    def recommendArm(self, bandit, history: List[Tuple[int, float]]) -> int:
        """
        Pick an arm via UCB1.
        """
        num_arms = bandit.getNumArms()
        self._ensure_init(num_arms)
        self._ingest_history(history)

        # UCB1: If any arm has not been played, play it first.
        for i in range(num_arms):
            if self.arm_counts[i] == 0:
                if self.debug:
                    print(f"[choose] initial play -> arm {i}")
                return i

        # Otherwise, pick arm maximizing: mean + c * sqrt(ln(t) / n_i)
        best_i = -1
        best_ucb = -float('inf')
        
        # ln(t)
        ln_t = math.log(self.total_counts)

        for i in range(num_arms):
            mean = self.arm_values[i]
            n = self.arm_counts[i]
            bonus = self.c * math.sqrt(ln_t / n)
            ucb = mean + bonus
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_i = i
        
        if self.debug:
            print(f"[choose] max UCB -> arm {best_i} (val={best_ucb:.4f})")
            
        return best_i
