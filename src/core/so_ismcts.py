"""Single Observer IS-MCTS (SO-ISMCTS).

In SO-ISMCTS, only the *root player's* information sets form tree nodes.
At opponent nodes, we simply choose actions uniformly at random (or via
a simple policy) without building tree structure.

This is appropriate when:
  - We only care about the root player's decision.
  - We want to save memory by not storing opponent subtrees.
  - The opponent model is "average over determinizations".

Algorithm:
  1. Determinize from the root player's perspective.
  2. Descend the tree at root-player nodes (UCB + availability).
  3. At opponent nodes, select uniformly at random.
  4. Expand one new node (only for root player).
  5. Simulate and backpropagate.
"""

from __future__ import annotations

import math
import random
from typing import Any, Hashable

from src.games.game_base import GameState


class _SONode:
    """Tree node for the root player's information sets only."""

    __slots__ = ("info_set_key", "children", "visits", "total_value",
                 "availability_count", "action_from_parent")

    def __init__(self, info_set_key: Hashable, action_from_parent: Any = None):
        self.info_set_key = info_set_key
        self.children: dict[Any, _SONode] = {}
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


def so_ismcts_search(
    root_state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
) -> _SONode:
    """Run Single Observer IS-MCTS.

    Only the root player's nodes are stored in the tree.
    Opponent actions are chosen randomly during tree descent.
    """
    root_player = root_state.current_player()
    root_key = root_state.information_set_key(root_player)
    root = _SONode(root_key)

    for _ in range(n_iterations):
        det_state = root_state.determinize(root_player)
        state = det_state
        node: _SONode | None = root
        path: list[_SONode] = [root]
        expanded = False

        while not state.is_terminal():
            legal = state.legal_actions()
            if not legal:
                break

            acting_player = state.current_player()

            if acting_player != root_player:
                # Opponent node: choose randomly, no tree structure
                action = random.choice(legal)
                state = state.apply_action(action)
                continue

            # Root player's turn: use tree
            assert node is not None

            if not expanded:
                # Update availability
                for a in legal:
                    if a in node.children:
                        node.children[a].availability_count += 1

                # Select or expand
                untried = [a for a in legal if a not in node.children]
                if untried:
                    action = random.choice(untried)
                    new_state = state.apply_action(action)
                    new_key = new_state.information_set_key(root_player) if not new_state.is_terminal() else None
                    child = _SONode(new_key, action_from_parent=action)
                    child.availability_count = 1
                    node.children[action] = child
                    node = child
                    path.append(node)
                    state = new_state
                    expanded = True
                else:
                    # All tried: UCB selection
                    total_avail = sum(
                        node.children[a].visits for a in legal if a in node.children
                    )
                    best_action = None
                    best_val = -float("inf")
                    for a in legal:
                        if a in node.children:
                            val = node.children[a].ucb_value(exploration, total_avail)
                            if val > best_val:
                                best_val = val
                                best_action = a
                    if best_action is None:
                        break
                    state = state.apply_action(best_action)
                    node = node.children[best_action]
                    path.append(node)
            else:
                # After expansion, just play randomly
                action = random.choice(legal)
                state = state.apply_action(action)

        # Simulate from current state
        value = _rollout(state, root_player) if not state.is_terminal() else state.payoff(root_player)

        # Backpropagate
        for n in path:
            n.visits += 1
            n.total_value += value

    return root


def so_ismcts_best_action(
    state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
) -> Any:
    """Convenience: run SO-ISMCTS and return the best action."""
    root = so_ismcts_search(state, n_iterations, exploration)
    best = root.get_best_action()
    if best is None:
        actions = state.legal_actions()
        return random.choice(actions) if actions else None
    return best
