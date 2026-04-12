"""Multiple Observer IS-MCTS (MO-ISMCTS).

In MO-ISMCTS, *every* player's information sets form tree nodes.
Each player has their own tree, and during descent, we switch between
trees depending on whose turn it is.

Key differences from SO-ISMCTS:
  - Maintains a separate tree for each player.
  - During selection, uses the acting player's tree at each decision point.
  - Determinization is done from the root player's perspective, but all
    players' statistics are updated.
  - Better models opponent behavior than SO-ISMCTS.

Algorithm:
  1. Determinize from the root player's perspective.
  2. Descend using each player's tree at their decision nodes.
  3. Expand one new node per player tree (if applicable).
  4. Simulate and backpropagate into ALL players' trees.
"""

from __future__ import annotations

import math
import random
from typing import Any, Hashable

from src.games.game_base import GameState


class _MONode:
    """Tree node for one player's information set."""

    __slots__ = ("info_set_key", "player", "children", "visits",
                 "total_value", "availability_count", "action_from_parent")

    def __init__(self, info_set_key: Hashable, player: int,
                 action_from_parent: Any = None):
        self.info_set_key = info_set_key
        self.player = player
        self.children: dict[Any, _MONode] = {}
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


def mo_ismcts_search(
    root_state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
) -> dict[int, _MONode]:
    """Run Multiple Observer IS-MCTS.

    Maintains a separate tree root for each player.
    Returns dict mapping player_id -> their root node.
    """
    root_player = root_state.current_player()
    n_players = root_state.n_players

    # Create root nodes for each player's tree
    roots: dict[int, _MONode] = {}
    for p in range(n_players):
        key = root_state.information_set_key(p)
        roots[p] = _MONode(key, p)

    for _ in range(n_iterations):
        det_state = root_state.determinize(root_player)
        state = det_state

        # Track the path through each player's tree
        paths: dict[int, list[_MONode]] = {p: [roots[p]] for p in range(n_players)}
        current_nodes: dict[int, _MONode | None] = {p: roots[p] for p in range(n_players)}
        expanded: set[int] = set()

        while not state.is_terminal():
            legal = state.legal_actions()
            if not legal:
                break

            acting_player = state.current_player()
            node = current_nodes[acting_player]

            if node is None or acting_player in expanded:
                # Already expanded or lost tree reference: play randomly
                action = random.choice(legal)
                state = state.apply_action(action)
                # Advance other players' nodes if they have matching children
                for p in range(n_players):
                    if p != acting_player and current_nodes[p] is not None:
                        # Other players observe this action
                        pass
                continue

            # Update availability for legal actions
            for a in legal:
                if a in node.children:
                    node.children[a].availability_count += 1

            # Select or expand
            untried = [a for a in legal if a not in node.children]
            if untried:
                action = random.choice(untried)
                new_state = state.apply_action(action)
                if not new_state.is_terminal():
                    new_key = new_state.information_set_key(acting_player)
                else:
                    new_key = None
                child = _MONode(new_key, acting_player, action_from_parent=action)
                child.availability_count = 1
                node.children[action] = child
                current_nodes[acting_player] = child
                paths[acting_player].append(child)
                expanded.add(acting_player)
                state = new_state
            else:
                # UCB selection
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
                current_nodes[acting_player] = node.children[best_action]
                paths[acting_player].append(node.children[best_action])

        # Simulate
        if not state.is_terminal():
            terminal_state_for_payoff = state
            s = state
            depth = 0
            while not s.is_terminal() and depth < 200:
                actions = s.legal_actions()
                if not actions:
                    break
                s = s.apply_action(random.choice(actions))
                depth += 1
            terminal_state_for_payoff = s
        else:
            terminal_state_for_payoff = state

        # Backpropagate into EACH player's tree
        for p in range(n_players):
            if terminal_state_for_payoff.is_terminal():
                value = terminal_state_for_payoff.payoff(p)
            else:
                value = 0.0
            for n in paths[p]:
                n.visits += 1
                n.total_value += value

    return roots


def mo_ismcts_best_action(
    state: GameState,
    n_iterations: int = 1000,
    exploration: float = 0.7,
) -> Any:
    """Convenience: run MO-ISMCTS and return the best action for the current player."""
    roots = mo_ismcts_search(state, n_iterations, exploration)
    root_player = state.current_player()
    root = roots[root_player]
    best = root.get_best_action()
    if best is None:
        actions = state.legal_actions()
        return random.choice(actions) if actions else None
    return best
