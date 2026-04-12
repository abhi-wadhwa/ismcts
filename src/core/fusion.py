"""Strategy Fusion analysis -- demonstrating the pathology of determinized search.

Strategy fusion is the phenomenon where PIMC (determinized MCTS) can make
suboptimal decisions because it "fuses" strategies from different determinizations.

The problem arises in multi-step decisions. PIMC solves each determinized world
independently and then picks the first action that *looks* best on average.
But different worlds may require different follow-up actions, and the player
cannot condition their future play on hidden information.

This module constructs a proper two-step example and compares PIMC vs IS-MCTS.

Two-Step Fusion Game:
  - A hidden card is dealt: A or B (50/50), unknown to the player.
  - Step 1: Player chooses COMMIT or SAFE.
  - If SAFE: game ends, payoff = +0.4 (regardless of card).
  - If COMMIT: Step 2 -- player must choose LEFT or RIGHT.
      Card A: LEFT -> +1.0, RIGHT -> -1.0
      Card B: LEFT -> -1.0, RIGHT -> +1.0

  Optimal play: SAFE (+0.4), since after COMMIT the player still doesn't
  know the card and LEFT/RIGHT each yield expected value 0.0.

  PIMC's mistake: In world A, PIMC sees COMMIT->LEFT = +1.0 > SAFE = +0.4.
  In world B, PIMC sees COMMIT->RIGHT = +1.0 > SAFE = +0.4.
  PIMC reports COMMIT as having value +1.0 (the best follow-up in each world).
  But the player cannot play LEFT in world A and RIGHT in world B -- they must
  pick one! COMMIT actually yields E[payoff] = 0.0, which is worse than SAFE's +0.4.

  This is strategy fusion: PIMC fuses the strategy (COMMIT->LEFT) from world A
  with (COMMIT->RIGHT) from world B, reporting the combined value of +1.0 for
  COMMIT, even though no single strategy achieves this.
"""

from __future__ import annotations

import random
from typing import Any, Hashable

from src.games.game_base import GameState
from src.core.pimc import PIMC
from src.core.so_ismcts import so_ismcts_search


class FusionGameState(GameState):
    """A two-step game that demonstrates strategy fusion.

    Step 1: choose COMMIT or SAFE (SAFE ends game with +0.4).
    Step 2 (after COMMIT): choose LEFT or RIGHT.
    Payoffs depend on hidden card (A=0 or B=1).
    """

    def __init__(
        self,
        hidden_card: int | None = None,
        step1_action: str | None = None,
        step2_action: str | None = None,
    ):
        if hidden_card is None:
            self._card = random.randint(0, 1)  # 0=A, 1=B
        else:
            self._card = hidden_card
        self._step1 = step1_action
        self._step2 = step2_action

    @property
    def n_players(self) -> int:
        return 1

    def current_player(self) -> int:
        if self.is_terminal():
            return -1
        return 0

    def legal_actions(self) -> list[str]:
        if self._step1 is None:
            return ["COMMIT", "SAFE"]
        if self._step1 == "COMMIT" and self._step2 is None:
            return ["LEFT", "RIGHT"]
        return []

    def apply_action(self, action: Any) -> "FusionGameState":
        if self._step1 is None:
            return FusionGameState(
                hidden_card=self._card,
                step1_action=action,
                step2_action=None,
            )
        else:
            return FusionGameState(
                hidden_card=self._card,
                step1_action=self._step1,
                step2_action=action,
            )

    def is_terminal(self) -> bool:
        if self._step1 == "SAFE":
            return True
        if self._step1 == "COMMIT" and self._step2 is not None:
            return True
        return False

    def payoff(self, player: int) -> float:
        if self._step1 == "SAFE":
            return 0.4
        if self._step2 == "LEFT":
            return 1.0 if self._card == 0 else -1.0
        if self._step2 == "RIGHT":
            return -1.0 if self._card == 0 else 1.0
        return 0.0

    def information_set_key(self, player: int) -> Hashable:
        # Player does NOT see the hidden card, but knows their own actions
        return ("step1", self._step1, "step2", self._step2)

    def determinize(self, observer: int) -> "FusionGameState":
        return FusionGameState(
            hidden_card=random.randint(0, 1),
            step1_action=self._step1,
            step2_action=self._step2,
        )

    def __repr__(self) -> str:
        card = "A" if self._card == 0 else "B"
        return f"FusionGame(card={card}, step1={self._step1}, step2={self._step2})"


def demonstrate_strategy_fusion(
    n_pimc_det: int = 100,
    n_pimc_iter: int = 50,
    n_ismcts_iter: int = 2000,
    n_trials: int = 200,
) -> dict[str, Any]:
    """Run the fusion demonstration.

    Returns a dict with:
      - pimc_values: PIMC's estimated action values (step 1)
      - ismcts_values: IS-MCTS's estimated action values (step 1)
      - pimc_actions: distribution of PIMC step-1 choices
      - ismcts_actions: distribution of IS-MCTS step-1 choices
      - pimc_actual_payoff: average actual payoff when following PIMC
      - ismcts_actual_payoff: average actual payoff when following IS-MCTS
      - optimal_payoff: the theoretically optimal payoff (0.4 via SAFE)
    """
    pimc = PIMC(n_determinizations=n_pimc_det, n_iterations_per_world=n_pimc_iter)

    pimc_action_counts: dict[str, int] = {"COMMIT": 0, "SAFE": 0}
    ismcts_action_counts: dict[str, int] = {"COMMIT": 0, "SAFE": 0}
    pimc_payoffs: list[float] = []
    ismcts_payoffs: list[float] = []
    pimc_all_values: list[dict[str, float]] = []
    ismcts_all_values: list[dict[str, float]] = []

    for _ in range(n_trials):
        game = FusionGameState()

        # PIMC decision
        pimc_vals = pimc.search(game)
        pimc_best = max(pimc_vals, key=lambda a: pimc_vals.get(a, 0))
        pimc_action_counts[pimc_best] = pimc_action_counts.get(pimc_best, 0) + 1
        pimc_all_values.append(dict(pimc_vals))

        # Simulate PIMC's actual play: if COMMIT, must pick LEFT or RIGHT blindly
        if pimc_best == "SAFE":
            pimc_payoffs.append(0.4)
        else:
            # PIMC picks COMMIT but then must choose LEFT/RIGHT without seeing card
            # Use PIMC again for step 2
            step2_state = game.apply_action("COMMIT")
            step2_action = pimc.best_action(step2_state)
            result = step2_state.apply_action(step2_action)
            pimc_payoffs.append(result.payoff(0))

        # IS-MCTS decision
        root = so_ismcts_search(game, n_iterations=n_ismcts_iter)
        ismcts_stats = root.get_action_stats()
        if ismcts_stats:
            ismcts_best = max(ismcts_stats, key=lambda a: ismcts_stats[a]["visits"])
        else:
            ismcts_best = random.choice(game.legal_actions())
        ismcts_action_counts[ismcts_best] = ismcts_action_counts.get(ismcts_best, 0) + 1
        ismcts_values_dict = {a: s["value"] for a, s in ismcts_stats.items()}
        ismcts_all_values.append(ismcts_values_dict)

        if ismcts_best == "SAFE":
            ismcts_payoffs.append(0.4)
        else:
            step2_state = game.apply_action("COMMIT")
            from src.core.so_ismcts import so_ismcts_best_action
            step2_action = so_ismcts_best_action(step2_state, n_iterations=n_ismcts_iter // 2)
            result = step2_state.apply_action(step2_action)
            ismcts_payoffs.append(result.payoff(0))

    # Aggregate values
    actions = ["COMMIT", "SAFE"]
    avg_pimc: dict[str, float] = {}
    for a in actions:
        vals = [v.get(a, 0.0) for v in pimc_all_values if a in v]
        avg_pimc[a] = sum(vals) / len(vals) if vals else 0.0

    avg_ismcts: dict[str, float] = {}
    for a in actions:
        vals = [v.get(a, 0.0) for v in ismcts_all_values if a in v]
        avg_ismcts[a] = sum(vals) / len(vals) if vals else 0.0

    return {
        "pimc_values": avg_pimc,
        "ismcts_values": avg_ismcts,
        "pimc_actions": dict(pimc_action_counts),
        "ismcts_actions": dict(ismcts_action_counts),
        "pimc_actual_payoff": sum(pimc_payoffs) / len(pimc_payoffs),
        "ismcts_actual_payoff": sum(ismcts_payoffs) / len(ismcts_payoffs),
        "optimal_payoff": 0.4,
        "explanation": (
            "PIMC suffers from strategy fusion: in each determinization it sees "
            "the hidden card, so COMMIT followed by the correct LEFT/RIGHT gives "
            "+1.0, beating SAFE's +0.4. PIMC reports COMMIT as having value ~1.0. "
            "But the player cannot see the card -- after choosing COMMIT, they must "
            "guess LEFT/RIGHT blindly, yielding expected payoff 0.0. "
            "IS-MCTS correctly identifies SAFE (value 0.4) as optimal because "
            "it maintains a single tree over information sets and recognizes "
            "that the player cannot condition step 2 on the hidden card."
        ),
    }
