"""Liar's Dice -- a bluffing dice game with imperfect information.

Rules (2-player, 1 die each, 6-sided):
  - Each player rolls one die secretly.
  - Players alternate making claims about the total dice showing a face value.
    A claim is (quantity, face_value) meaning "there are at least `quantity` dice
    showing `face_value` among ALL dice on the table".
  - Each subsequent claim must be strictly higher than the previous:
    either higher quantity, or same quantity with higher face value.
  - Instead of making a claim, a player can challenge ("liar!").
  - On challenge, dice are revealed:
      - If the claim is TRUE (actual count >= claimed quantity), the challenger loses.
      - If the claim is FALSE, the claimer loses.
  - The loser gets payoff -1, the winner gets +1.

Information sets:
  - A player sees their own die and the sequence of public claims.
"""

from __future__ import annotations

import random
from typing import Any, Hashable

from src.games.game_base import GameState

CHALLENGE = "challenge"


def _all_claims() -> list[tuple[int, int]]:
    """Generate all valid claims in order: (quantity, face) sorted."""
    claims = []
    for qty in range(1, 3):  # max 2 dice total, so quantity in {1, 2}
        for face in range(1, 7):
            claims.append((qty, face))
    return claims


ALL_CLAIMS = _all_claims()
CLAIM_ORDER = {c: i for i, c in enumerate(ALL_CLAIMS)}


def claim_is_higher(new: tuple[int, int], old: tuple[int, int]) -> bool:
    """Check if `new` claim is strictly higher than `old`."""
    return CLAIM_ORDER[new] > CLAIM_ORDER[old]


class LiarsDiceState(GameState):
    """State for 2-player Liar's Dice (1 die each)."""

    def __init__(
        self,
        dice: tuple[int, int] | None = None,
        claims: tuple[tuple[int, int], ...] = (),
        challenged: bool = False,
    ):
        if dice is None:
            self._dice = (random.randint(1, 6), random.randint(1, 6))
        else:
            self._dice = dice
        self._claims = claims
        self._challenged = challenged

    @property
    def n_players(self) -> int:
        return 2

    def current_player(self) -> int:
        if self.is_terminal():
            return -1
        return len(self._claims) % 2

    def legal_actions(self) -> list[Any]:
        if self.is_terminal():
            return []
        actions: list[Any] = []
        if self._claims:
            # Can challenge
            actions.append(CHALLENGE)
            # Can make higher claims
            last = self._claims[-1]
            for c in ALL_CLAIMS:
                if claim_is_higher(c, last):
                    actions.append(c)
        else:
            # First claim: any claim is valid
            actions = list(ALL_CLAIMS)
        return actions

    def apply_action(self, action: Any) -> "LiarsDiceState":
        if action == CHALLENGE:
            return LiarsDiceState(
                dice=self._dice,
                claims=self._claims,
                challenged=True,
            )
        return LiarsDiceState(
            dice=self._dice,
            claims=self._claims + (action,),
            challenged=False,
        )

    def is_terminal(self) -> bool:
        return self._challenged

    def payoff(self, player: int) -> float:
        if not self._challenged:
            return 0.0
        # The last claim was made by the player before the challenger
        challenger = len(self._claims) % 2
        claimer = 1 - challenger
        last_claim = self._claims[-1]
        qty_claimed, face_claimed = last_claim

        # Count actual dice showing that face
        actual_count = sum(1 for d in self._dice if d == face_claimed)

        if actual_count >= qty_claimed:
            # Claim was true -> challenger loses
            loser = challenger
        else:
            # Claim was false -> claimer loses
            loser = claimer

        if player == loser:
            return -1.0
        else:
            return 1.0

    def information_set_key(self, player: int) -> Hashable:
        """Player knows their own die and the claim history."""
        return (self._dice[player], self._claims)

    def determinize(self, observer: int) -> "LiarsDiceState":
        """Sample opponent's die uniformly from 1..6."""
        my_die = self._dice[observer]
        opp_die = random.randint(1, 6)
        if observer == 0:
            new_dice = (my_die, opp_die)
        else:
            new_dice = (opp_die, my_die)
        return LiarsDiceState(
            dice=new_dice,
            claims=self._claims,
            challenged=self._challenged,
        )

    def __repr__(self) -> str:
        claims_str = ", ".join(
            f"{q}x{f}" for q, f in self._claims
        ) if self._claims else "(none)"
        status = " [CHALLENGED]" if self._challenged else ""
        return f"LiarsDice(dice={self._dice}, claims=[{claims_str}]{status})"
