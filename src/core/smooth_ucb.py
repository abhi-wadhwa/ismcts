"""Smooth UCB for IS-MCTS.

Standard UCB assumes stationary reward distributions. In IS-MCTS, the
effective distribution at a node changes as different determinizations
visit different subsets of children, making the environment non-stationary.

Smooth UCB (Heinrich & Silver, 2015) addresses this by mixing the tree
policy with a uniform random baseline:

    pi_smooth(a) = eta * pi_tree(a) + (1 - eta) * uniform(a)

where:
  - eta increases toward 1 as visits grow: eta = max(0, (N - D) / N)
  - D is a dampening parameter
  - pi_tree(a) is the greedy UCB-selected action (deterministic)
  - uniform(a) = 1 / |legal_actions|

This "smooths out" the non-stationarity during early search when
statistics are noisy and determinizations cause high variance.
"""

from __future__ import annotations

import math
import random
from typing import Any, Hashable

from src.games.game_base import GameState


class _SmoothNode:
    """Node with smooth UCB statistics."""

    __slots__ = ("info_set_key", "player", "children", "visits",
                 "total_value", "availability_count", "action_from_parent")

    def __init__(self, info_set_key: Hashable, player: int,
                 action_from_parent: Any = None):
        self.info_set_key = info_set_key
        self.player = player
        self.children: dict[Any, _SmoothNode] = {}
        self.visits: int = 0
        self.total_value: float = 0.0
        self.availability_count: int = 0
        self.action_from_parent = action_from_parent

    def ucb_value(self, exploration: float, total_avail: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = exploration * math.sqrt(math.log(max(total_avail, 1)) / self.visits)
        return exploit + explore

    def smooth_select(
        self,
        legal_actions: list[Any],
        exploration: float,
        dampen: float,
    ) -> Any:
        """Select action using smooth UCB policy.

        Mixes greedy UCB selection with uniform random.
        """
        # Check for untried actions first
        untried = [a for a in legal_actions if a not in self.children]
        if untried:
            return random.choice(untried)

        # Compute smoothing parameter eta
        eta = max(0.0, (self.visits - dampen) / self.visits) if self.visits > 0 else 0.0

        # With probability (1 - eta), pick uniformly
        if random.random() > eta:
            return random.choice(legal_actions)

        # With probability eta, pick greedily via UCB
        total_avail = sum(
            self.children[a].visits for a in legal_actions if a in self.children
        )
        total_avail = max(total_avail, 1)

        best_action = None
        best_val = -float("inf")
        for a in legal_actions:
            if a in self.children:
                val = self.children[a].ucb_value(exploration, total_avail)
                if val > best_val:
                    best_val = val
                    best_action = a

        return best_action if best_action is not None else random.choice(legal_actions)

    def get_best_action(self) -> Any:
        if not self.children:
            return None
        return max(self.children, key=lambda a: self.children[a].visits)

    def get_action_stats(self) -> dict[Any, dict[str, float]]:
        stats = {}
        for action, child in self.children.items():
            stats[action] = {
                "visits": child.visits,
                "value": child.total_value / child.visits if child.visits > 0 else 0.0,
                "availability": child.availability_count,
            }
        return stats


def _rollout(state: GameState, player: int) -> float:
    s = state
    depth = 0
    while not s.is_terminal() and depth < 200:
        actions = s.legal_actions()
        if not actions:
            break
        s = s.apply_action(random.choice(actions))
        depth += 1
    return s.payoff(player) if s.is_terminal() else 0.0


def smooth_ucb_search(
    root_state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
    dampen: float = 50.0,
) -> _SmoothNode:
    """Run IS-MCTS with Smooth UCB selection.

    Parameters:
        root_state: The current game state.
        n_iterations: Number of MCTS iterations.
        exploration: UCB exploration constant.
        dampen: Smoothing dampening parameter D.
                Higher D means more random exploration for longer.

    Returns:
        Root node with accumulated statistics.
    """
    root_player = root_state.current_player()
    root_key = root_state.information_set_key(root_player)
    root = _SmoothNode(root_key, root_player)

    for _ in range(n_iterations):
        det_state = root_state.determinize(root_player)
        state = det_state
        node = root
        path: list[_SmoothNode] = [root]
        expanded = False

        while not state.is_terminal():
            legal = state.legal_actions()
            if not legal:
                break

            acting_player = state.current_player()

            if acting_player != root_player:
                # Opponent: random
                action = random.choice(legal)
                state = state.apply_action(action)
                continue

            if expanded:
                action = random.choice(legal)
                state = state.apply_action(action)
                continue

            # Update availability
            for a in legal:
                if a in node.children:
                    node.children[a].availability_count += 1

            # Smooth UCB selection
            action = node.smooth_select(legal, exploration, dampen)

            if action not in node.children:
                # Expand
                new_state = state.apply_action(action)
                new_key = (
                    new_state.information_set_key(root_player)
                    if not new_state.is_terminal()
                    else None
                )
                child = _SmoothNode(new_key, root_player, action_from_parent=action)
                child.availability_count = 1
                node.children[action] = child
                node = child
                path.append(node)
                state = new_state
                expanded = True
            else:
                state = state.apply_action(action)
                node = node.children[action]
                path.append(node)

        # Simulate
        value = (
            _rollout(state, root_player)
            if not state.is_terminal()
            else state.payoff(root_player)
        )

        # Backpropagate
        for n in path:
            n.visits += 1
            n.total_value += value

    return root


def smooth_ucb_best_action(
    state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
    dampen: float = 50.0,
) -> Any:
    """Convenience: run Smooth UCB IS-MCTS and return best action."""
    root = smooth_ucb_search(state, n_iterations, exploration, dampen)
    best = root.get_best_action()
    if best is None:
        actions = state.legal_actions()
        return random.choice(actions) if actions else None
    return best
