"""An epsilon-greedy bandit over strategy choices, rewarded by AUTOMATIC checks on the saved run record.

This is the simplest form of reinforcement learning: one state, a few actions (arms), and a reward after each action.
Here the arms are retrieval depths (how many papers are given to the agents), and the reward comes from the run:

  + 1/3 if the sandbox run succeeded,
  + 1/3 if the hypothesis cited a retrieved paper and no invented id,
  + 1/3 if the critic produced a verdict and the design/hypothesis check raised no mismatch warning.

The reward is a PROXY for a clean run. It does not measure scientific quality, and the language model never grades
itself. Whether this learning helps is not shown by this code: that needs many real runs (each takes minutes),
compared with a fixed and a random strategy. The tests only prove that the algorithm learns on simulated rewards.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_ARMS = (2, 3, 5)


def reward_from_record(record: Dict[str, Any]) -> float:
    """Reward in [0, 1] from the automatic checks on one saved run record."""
    reward = 0.0
    if record.get("execution_status") == "SUCCESS":
        reward += 1.0
    grounding = record.get("hypothesis_grounding") or {}
    if grounding.get("cited_ids") and not grounding.get("unknown_ids"):
        reward += 1.0
    exp, critic = record.get("experiment") or {}, record.get("critic") or {}
    notes = " ".join([str(exp.get("note", ""))] + [str(w) for w in record.get("warnings") or []])
    if critic.get("verdict") in ("SIGNIFICANT", "INCONCLUSIVE") and "Possible mismatch" not in notes:
        reward += 1.0
    return reward / 3.0


class StrategyBandit:
    def __init__(self, path: Path, arms: Sequence[int] = DEFAULT_ARMS, epsilon: float = 0.2,
                 rng: Optional[random.Random] = None) -> None:
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("arms must be a non-empty list of different values")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.path, self.arms, self.epsilon = Path(path), [int(a) for a in arms], epsilon
        self.rng = rng or random.Random()
        self.stats: Dict[str, Dict[str, float]] = {str(a): {"n": 0, "reward": 0.0} for a in self.arms}
        self._load()

    # ---- learning --------------------------------------------------------
    def mean(self, arm: int) -> float:
        s = self.stats[str(arm)]
        return s["reward"] / s["n"] if s["n"] else 0.0

    def choose(self) -> Tuple[int, str]:
        """(arm, reason). Every arm is tried once first, then: explore with probability epsilon, else use the best."""
        untried = [a for a in self.arms if self.stats[str(a)]["n"] == 0]
        if untried:
            return untried[0], "first try of this strategy"
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms), "exploring a random strategy"
        best = max(self.mean(a) for a in self.arms)
        return self.rng.choice([a for a in self.arms if self.mean(a) == best]), "best average reward so far"

    def update(self, arm: int, reward: float) -> None:
        if str(arm) not in self.stats:
            raise ValueError(f"unknown arm {arm}")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be between 0 and 1")
        self.stats[str(arm)]["n"] += 1
        self.stats[str(arm)]["reward"] += reward
        self._save()

    def best_arm(self) -> Optional[int]:
        tried = [a for a in self.arms if self.stats[str(a)]["n"]]
        return max(tried, key=self.mean) if tried else None

    def total_runs(self) -> int:
        return int(sum(s["n"] for s in self.stats.values()))

    def describe(self) -> List[str]:
        lines = [f"{self.total_runs()} run(s) recorded; epsilon = {self.epsilon}"]
        for a in self.arms:
            s = self.stats[str(a)]
            lines.append(f"  retrieve {a} papers: {int(s['n'])} run(s), average reward {self.mean(a):.2f}" if s["n"]
                         else f"  retrieve {a} papers: not tried yet")
        return lines

    # ---- saving ------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))["arms"]
            for arm in self.arms:
                item = saved.get(str(arm))
                if item:
                    self.stats[str(arm)] = {"n": int(item["n"]), "reward": float(item["reward"])}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"The bandit file {self.path} is damaged ({exc}). Delete it to start again.") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"arms": self.stats}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)           # never leave a half-written file
