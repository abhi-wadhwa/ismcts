"""Perfect Information Monte Carlo (PIMC) -- determinized MCTS.

Algorithm:
  1. For each iteration:
     a. Sample a determinization consistent with the observer's info set.
     b. Run one iteration of standard (perfect-info) MCTS on that world.
     c. Back up the result.
  2. Aggregate action values across all determinizations.
  3. Choose the action with the highest average value.

This is the simplest approach but suffers from *strategy fusion*:
it may select an action that is optimal on average across worlds but
is not actually achievable by any single strategy (because the player
cannot distinguish the worlds).
"""

from __future__ import annotations

import math
import random
from typing import Any

from src.games.game_base import GameState


class _MCTSNode:
    """Standard MCTS node for a single determinized world."""

    __slots__ = ("state", "parent", "action", "children", "visits", "total_value",
                 "untried_actions")

    def __init__(self, state: GameState, parent: "_MCTSNode | None" = None,
                 action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[_MCTSNode] = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = list(state.legal_actions())

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.41) -> "_MCTSNode":
        """UCB1 selection."""
        best = None
        best_val = -float("inf")
        log_parent = math.log(self.visits)
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.total_value / child.visits
            explore = c * math.sqrt(log_parent / child.visits)
            ucb = exploit + explore
            if ucb > best_val:
                best_val = ucb
                best = child
        assert best is not None
        return best

    def expand(self) -> "_MCTSNode":
        action = self.untried_actions.pop()
        new_state = self.state.apply_action(action)
        child = _MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child


def _rollout(state: GameState, player: int) -> float:
    """Random playout from `state`, returning payoff for `player`."""
    s = state
    while not s.is_terminal():
        actions = s.legal_actions()
        a = random.choice(actions)
        s = s.apply_action(a)
    return s.payoff(player)


def _mcts_iteration(root: _MCTSNode, player: int) -> None:
    """One iteration of standard MCTS: select, expand, simulate, backprop."""
    node = root

    # Selection
    while not node.state.is_terminal() and node.is_fully_expanded():
        node = node.best_child()

    # Expansion
    if not node.state.is_terminal() and not node.is_fully_expanded():
        node = node.expand()

    # Simulation
    value = _rollout(node.state, player)

    # Backpropagation
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent


class PIMC:
    """Perfect Information Monte Carlo search.

    Performs determinized MCTS: samples worlds, runs standard MCTS on each,
    and aggregates action values.
    """

    def __init__(
        self,
        n_determinizations: int = 20,
        n_iterations_per_world: int = 50,
        exploration: float = 1.41,
    ):
        self.n_determinizations = n_determinizations
        self.n_iterations_per_world = n_iterations_per_world
        self.exploration = exploration

    def search(self, state: GameState) -> dict[Any, float]:
        """Run PIMC search, return action -> average value mapping."""
        player = state.current_player()
        action_values: dict[Any, list[float]] = {}

        for _ in range(self.n_determinizations):
            # Sample a determinization
            det_state = state.determinize(player)
            root = _MCTSNode(det_state)

            # Run MCTS on this determinized world
            for _ in range(self.n_iterations_per_world):
                _mcts_iteration(root, player)

            # Collect action values from the root's children
            for child in root.children:
                if child.visits > 0:
                    avg_val = child.total_value / child.visits
                    if child.action not in action_values:
                        action_values[child.action] = []
                    action_values[child.action].append(avg_val)

        # Average across determinizations
        result = {}
        for action, values in action_values.items():
            result[action] = sum(values) / len(values) if values else 0.0
        return result

    def best_action(self, state: GameState) -> Any:
        """Return the best action according to PIMC search."""
        values = self.search(state)
        if not values:
            actions = state.legal_actions()
            return random.choice(actions) if actions else None
        return max(values, key=values.get)  # type: ignore[arg-type]
