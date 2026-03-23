"""Abstract base class for imperfect-information games.

Every game must support:
  - Identifying the current player
  - Listing legal actions
  - Applying actions to produce new states
  - Terminal detection and payoff computation
  - Information set identification (what the acting player *knows*)
  - Determinization (sampling hidden info consistent with observations)
"""

from __future__ import annotations

import abc
import copy
from typing import Any, Hashable, Sequence


class GameState(abc.ABC):
    """Abstract state for an imperfect-information game.

    Conventions:
      - Players are numbered 0, 1, ... (n_players - 1).
      - Payoffs are from each player's perspective: positive = good.
      - Information sets are hashable objects that capture what a player knows.
    """

    @abc.abstractmethod
    def current_player(self) -> int:
        """Return the index of the player to act (or -1 if terminal/chance)."""

    @abc.abstractmethod
    def legal_actions(self) -> list[Any]:
        """Return the list of legal actions in the current state."""

    @abc.abstractmethod
    def apply_action(self, action: Any) -> "GameState":
        """Return a *new* state resulting from taking `action`.

        Must not mutate self.
        """

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""

    @abc.abstractmethod
    def payoff(self, player: int) -> float:
        """Return the payoff for `player` (only valid at terminal states)."""

    @abc.abstractmethod
    def information_set_key(self, player: int) -> Hashable:
        """Return a hashable key representing what `player` knows.

        Two states that are indistinguishable to `player` must return the
        same key.  The key typically encodes the player's private info plus
        the public action history.
        """

    @abc.abstractmethod
    def determinize(self, observer: int) -> "GameState":
        """Sample a complete state consistent with `observer`'s information set.

        Returns a new GameState where all hidden information has been filled in
        uniformly at random from the possibilities consistent with what
        `observer` currently knows.
        """

    @property
    @abc.abstractmethod
    def n_players(self) -> int:
        """Return the number of players."""

    def clone(self) -> "GameState":
        """Deep-copy this state."""
        return copy.deepcopy(self)

    def legal_actions_for_determinization(self, action: Any) -> bool:
        """Check whether `action` is legal in this (possibly determinized) state.

        Default implementation just checks membership in legal_actions().
        Override for efficiency if needed.
        """
        return action in self.legal_actions()
