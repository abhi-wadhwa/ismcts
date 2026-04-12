"""Kuhn Poker -- minimal imperfect-information poker game.

Rules (standard 3-card, 2-player variant):
  - Deck has cards {J, Q, K} (values 0, 1, 2).
  - Each player antes 1 chip and is dealt one card.
  - Player 0 acts first: Pass or Bet.
  - Depending on actions, player 1 then acts.
  - Possible sequences:
      Pass -> Pass  : showdown (pot = 2)
      Pass -> Bet   : player 0 can Pass (fold) or Bet (call)
      Bet  -> Pass  : player 1 folds, player 0 wins pot = 3
      Bet  -> Bet   : showdown (pot = 4)
  - At showdown, higher card wins.

Information sets:
  - A player sees only their own card and the public action history.
"""

from __future__ import annotations

import random
from typing import Any, Hashable

from src.games.game_base import GameState

CARDS = [0, 1, 2]  # J=0, Q=1, K=2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
PASS = "pass"
BET = "bet"


class KuhnPokerState(GameState):
    """State for Kuhn Poker."""

    def __init__(
        self,
        hands: tuple[int, int] | None = None,
        history: tuple[str, ...] = (),
        pot: tuple[int, int] = (1, 1),
    ):
        if hands is None:
            # Deal randomly
            cards = random.sample(CARDS, 2)
            self._hands: tuple[int, int] = (cards[0], cards[1])
        else:
            self._hands = hands
        self._history = history
        self._pot = pot

    @property
    def n_players(self) -> int:
        return 2

    def current_player(self) -> int:
        if self.is_terminal():
            return -1
        h = self._history
        if len(h) == 0:
            return 0
        if len(h) == 1:
            return 1
        # len(h) == 2 means: Pass-Bet sequence, player 0 to respond
        return 0

    def legal_actions(self) -> list[str]:
        if self.is_terminal():
            return []
        return [PASS, BET]

    def apply_action(self, action: Any) -> "KuhnPokerState":
        new_history = self._history + (action,)
        new_pot = list(self._pot)

        # When a player bets, they add 1 more to the pot
        cp = self.current_player()
        if action == BET:
            new_pot[cp] += 1

        return KuhnPokerState(
            hands=self._hands,
            history=new_history,
            pot=(new_pot[0], new_pot[1]),
        )

    def is_terminal(self) -> bool:
        h = self._history
        if len(h) < 2:
            return False
        # Terminal sequences: PP, PBP, PBB, BP, BB
        if h == (PASS, PASS):
            return True
        if h == (BET, PASS):
            return True
        if h == (BET, BET):
            return True
        if len(h) == 3:
            return True  # PBP or PBB
        return False

    def payoff(self, player: int) -> float:
        h = self._history
        # Fold cases
        if h == (PASS, BET, PASS):
            # Player 0 folded to player 1's bet
            winner = 1
        elif h == (BET, PASS):
            # Player 1 folded to player 0's bet
            winner = 0
        else:
            # Showdown
            if self._hands[0] > self._hands[1]:
                winner = 0
            else:
                winner = 1

        total_pot = self._pot[0] + self._pot[1]
        if player == winner:
            return total_pot - self._pot[player]  # net gain
        else:
            return -self._pot[player]  # net loss

    def information_set_key(self, player: int) -> Hashable:
        """Player knows their card + the action history."""
        return (self._hands[player], self._history)

    def determinize(self, observer: int) -> "KuhnPokerState":
        """Sample opponent's card uniformly from remaining cards."""
        my_card = self._hands[observer]
        remaining = [c for c in CARDS if c != my_card]
        opp_card = random.choice(remaining)
        if observer == 0:
            new_hands = (my_card, opp_card)
        else:
            new_hands = (opp_card, my_card)
        return KuhnPokerState(
            hands=new_hands,
            history=self._history,
            pot=self._pot,
        )

    def __repr__(self) -> str:
        h0 = CARD_NAMES[self._hands[0]]
        h1 = CARD_NAMES[self._hands[1]]
        hist = "-".join(self._history) if self._history else "(start)"
        return f"Kuhn(P0={h0}, P1={h1}, history={hist}, pot={self._pot})"
