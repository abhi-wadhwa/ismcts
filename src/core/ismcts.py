"""Information Set MCTS (IS-MCTS) -- Cowling, Powley, Whitehouse (2012).

Key insight: instead of building a tree of game states, build a tree of
*information sets*. Each node in the tree corresponds to an information set
(what the acting player knows), not a specific state.

Different determinizations may traverse different paths through the tree,
but they share statistics at nodes that correspond to the same information set.

Availability:
  Not all actions are available in all determinizations. We use "availability
  counts" -- the number of times an action was legal when its parent was visited.
  UCB is computed using availability count instead of parent visit count:

    UCB = Q(a)/N(a) + c * sqrt(ln(Avail(a)) / N(a))

  where Avail(a) = number of times action `a` was available at the parent.
"""

from __future__ import annotations

import math
import random
from typing import Any, Hashable

from src.games.game_base import GameState


class ISMCTSNode:
    """A node in an IS-MCTS tree.

    Each node represents an information set for the acting player.
    Children are indexed by action.
    """

    __slots__ = ("info_set_key", "player", "children", "visits", "availability_count",
                 "total_value", "action_from_parent")

    def __init__(self, info_set_key: Hashable, player: int,
                 action_from_parent: Any = None):
        self.info_set_key = info_set_key
        self.player = player
        self.action_from_parent = action_from_parent
        self.children: dict[Any, ISMCTSNode] = {}
        self.visits: int = 0
        self.availability_count: int = 0  # how many times this node's action was available
        self.total_value: float = 0.0

    def ucb_value(self, exploration: float, parent_avail: int) -> float:
        """Compute UCB value using availability count."""
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = exploration * math.sqrt(math.log(parent_avail) / self.visits)
        return exploit + explore

    def select_child(self, legal_actions: list[Any], exploration: float) -> Any:
        """Select an action using UCB with availability counts.

        Only considers actions that are legal in the current determinization.
        Updates availability counts for all legal children.
        """
        # Count total availability for UCB denominator
        total_avail = sum(
            self.children[a].visits for a in legal_actions if a in self.children
        )
        total_avail = max(total_avail, 1)

        # Update availability counts for legal actions
        for a in legal_actions:
            if a in self.children:
                self.children[a].availability_count += 1

        best_action = None
        best_val = -float("inf")

        for a in legal_actions:
            if a not in self.children:
                # Untried action -> explore immediately
                return a
            child = self.children[a]
            val = child.ucb_value(exploration, total_avail)
            if val > best_val:
                best_val = val
                best_action = a

        return best_action

    def get_best_action(self) -> Any:
        """Return the most-visited child action (robust child selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda a: self.children[a].visits)

    def get_action_stats(self) -> dict[Any, dict[str, float]]:
        """Return statistics for each child action."""
        stats = {}
        for action, child in self.children.items():
            stats[action] = {
                "visits": child.visits,
                "value": child.total_value / child.visits if child.visits > 0 else 0.0,
                "availability": child.availability_count,
            }
        return stats


def _rollout(state: GameState, player: int) -> float:
    """Random playout returning payoff for `player`."""
    s = state
    depth = 0
    max_depth = 200
    while not s.is_terminal() and depth < max_depth:
        actions = s.legal_actions()
        if not actions:
            break
        s = s.apply_action(random.choice(actions))
        depth += 1
    if s.is_terminal():
        return s.payoff(player)
    return 0.0


def ismcts_search(
    root_state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
) -> ISMCTSNode:
    """Run IS-MCTS from `root_state`.

    Each iteration:
      1. Sample a determinization from the root player's perspective.
      2. Descend the IS tree using UCB with availability, expanding one node.
      3. Simulate (random playout).
      4. Backpropagate.

    Returns the root ISMCTSNode with accumulated statistics.
    """
    root_player = root_state.current_player()
    root_key = root_state.information_set_key(root_player)
    root = ISMCTSNode(root_key, root_player)

    for _ in range(n_iterations):
        # 1. Determinize
        det_state = root_state.determinize(root_player)

        # 2. Selection + Expansion
        node = root
        state = det_state
        path: list[tuple[ISMCTSNode, float]] = []

        while not state.is_terminal():
            legal = state.legal_actions()
            if not legal:
                break

            acting_player = state.current_player()
            action = node.select_child(legal, exploration)

            if action not in node.children:
                # Expand
                new_state = state.apply_action(action)
                new_player = new_state.current_player() if not new_state.is_terminal() else acting_player
                new_key = new_state.information_set_key(new_player) if not new_state.is_terminal() else None
                child = ISMCTSNode(new_key, new_player, action_from_parent=action)
                child.availability_count = 1
                node.children[action] = child
                state = new_state
                node = child
                break
            else:
                state = state.apply_action(action)
                node = node.children[action]

        # 3. Simulate
        value = _rollout(state, root_player)

        # 4. Backpropagate -- walk back up using parent tracking
        # Since IS tree nodes don't store parent refs, we re-trace the path
        # We need to redo the descent to build the path
        _backprop_iteration(root, det_state, root_player, value, exploration)

    return root


def _backprop_iteration(
    root: ISMCTSNode,
    det_state: GameState,
    root_player: int,
    value: float,
    exploration: float,
) -> None:
    """Re-trace the path through the tree and update visit/value counts.

    This mirrors the selection step but only updates statistics.
    """
    node = root
    state = det_state
    path: list[ISMCTSNode] = [root]

    while not state.is_terminal():
        legal = state.legal_actions()
        if not legal:
            break

        # Determine which action would be selected (deterministically replay)
        best_action = None
        best_val = -float("inf")
        has_untried = False

        total_avail = sum(
            node.children[a].visits for a in legal if a in node.children
        )
        total_avail = max(total_avail, 1)

        for a in legal:
            if a not in node.children:
                has_untried = True
                best_action = a
                break
            child = node.children[a]
            val = child.ucb_value(exploration, total_avail)
            if val > best_val:
                best_val = val
                best_action = a

        if best_action is None:
            break

        if best_action not in node.children:
            # This was the expansion point
            if best_action in node.children:
                node = node.children[best_action]
                path.append(node)
            break

        state = state.apply_action(best_action)
        node = node.children[best_action]
        path.append(node)

    # Update all nodes on the path
    for n in path:
        n.visits += 1
        n.total_value += value
