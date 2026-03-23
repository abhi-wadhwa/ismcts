"""Phantom Tic-Tac-Toe -- imperfect-information variant of Tic-Tac-Toe.

Rules:
  - Standard 3x3 Tic-Tac-Toe but players cannot see each other's moves.
  - When a player tries to place on an occupied square, they are told
    "illegal" and must try again (this reveals some information).
  - A player knows: their own marks, which of their attempts were rejected,
    and the public move count.

Information sets:
  - Player knows their own placed marks + which squares they tried that
    were rejected (occupied by opponent).

Simplification for tree search:
  - We model the game such that on each turn, the current player picks
    from squares that are not their own marks and not previously rejected.
    If the square is actually occupied by the opponent, the move is
    "rejected" and the player must try again (same turn).
  - For search purposes, we treat each attempt as an action, with
    "rejected" attempts staying on the same player's turn.
"""

from __future__ import annotations

import random
from typing import Any, Hashable

from src.games.game_base import GameState

EMPTY = 0
X_MARK = 1  # Player 0
O_MARK = 2  # Player 1

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]


def _check_winner(board: list[int]) -> int:
    """Return X_MARK, O_MARK, or EMPTY."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return EMPTY


class PhantomTTTState(GameState):
    """State for Phantom Tic-Tac-Toe.

    The board is fully tracked internally, but players only see partial info.
    """

    def __init__(
        self,
        board: list[int] | None = None,
        current: int = 0,
        # Track what each player knows: their own marks + rejected squares
        known_own: tuple[frozenset[int], frozenset[int]] | None = None,
        known_rejected: tuple[frozenset[int], frozenset[int]] | None = None,
        move_count: int = 0,
    ):
        self._board = board if board is not None else [EMPTY] * 9
        self._current = current
        self._known_own = known_own or (frozenset(), frozenset())
        self._known_rejected = known_rejected or (frozenset(), frozenset())
        self._move_count = move_count

    @property
    def n_players(self) -> int:
        return 2

    def current_player(self) -> int:
        if self.is_terminal():
            return -1
        return self._current

    def legal_actions(self) -> list[int]:
        """Return squares the current player can attempt.

        A player won't re-try their own marks or previously rejected squares.
        They CAN try squares occupied by the opponent (which they don't know about),
        but for the search, we only offer actually-empty squares plus unknown-opponent
        squares. In a fully determinized state all hidden info is known, so we
        just return actually empty squares.
        """
        if self.is_terminal():
            return []
        p = self._current
        own = self._known_own[p]
        rejected = self._known_rejected[p]
        excluded = own | rejected
        return [i for i in range(9) if i not in excluded]

    def apply_action(self, action: int) -> "PhantomTTTState":
        p = self._current
        mark = X_MARK if p == 0 else O_MARK

        new_board = self._board[:]
        new_own_list = [set(s) for s in self._known_own]
        new_rej_list = [set(s) for s in self._known_rejected]

        if new_board[action] != EMPTY:
            # Square is occupied by opponent -> rejection
            new_rej_list[p].add(action)
            new_known_own = (frozenset(new_own_list[0]), frozenset(new_own_list[1]))
            new_known_rej = (frozenset(new_rej_list[0]), frozenset(new_rej_list[1]))
            # Same player tries again
            return PhantomTTTState(
                board=new_board,
                current=p,
                known_own=new_known_own,
                known_rejected=new_known_rej,
                move_count=self._move_count,
            )
        else:
            # Valid placement
            new_board[action] = mark
            new_own_list[p].add(action)
            new_known_own = (frozenset(new_own_list[0]), frozenset(new_own_list[1]))
            new_known_rej = (frozenset(new_rej_list[0]), frozenset(new_rej_list[1]))
            next_player = 1 - p
            return PhantomTTTState(
                board=new_board,
                current=next_player,
                known_own=new_known_own,
                known_rejected=new_known_rej,
                move_count=self._move_count + 1,
            )

    def is_terminal(self) -> bool:
        w = _check_winner(self._board)
        if w != EMPTY:
            return True
        return all(c != EMPTY for c in self._board)

    def payoff(self, player: int) -> float:
        w = _check_winner(self._board)
        if w == EMPTY:
            return 0.0  # draw
        mark = X_MARK if player == 0 else O_MARK
        return 1.0 if w == mark else -1.0

    def information_set_key(self, player: int) -> Hashable:
        """Player knows their own marks, rejected squares, and whose turn it is."""
        return (
            player,
            self._known_own[player],
            self._known_rejected[player],
            self._current,
            self._move_count,
        )

    def determinize(self, observer: int) -> "PhantomTTTState":
        """Fill in opponent's marks consistently with observer's knowledge.

        Observer knows:
          - Where their own marks are
          - Which squares were rejected (opponent is there)
          - The total move count (so they know how many opponent marks exist)

        We must place opponent marks on:
          - All rejected squares (known opponent locations)
          - Remaining opponent marks on random unknown squares
        """
        opp = 1 - observer
        opp_mark = X_MARK if opp == 0 else O_MARK
        my_mark = X_MARK if observer == 0 else O_MARK

        # How many marks does the opponent have?
        my_count = len(self._known_own[observer])
        # Total marks placed = move_count
        opp_count = self._move_count - my_count

        # Start with a fresh board with observer's marks
        new_board = [EMPTY] * 9
        for sq in self._known_own[observer]:
            new_board[sq] = my_mark

        # Place opponent marks on rejected squares (we know they're there)
        rejected = self._known_rejected[observer]
        for sq in rejected:
            new_board[sq] = opp_mark

        # Remaining opponent marks to place
        remaining_opp = opp_count - len(rejected)

        if remaining_opp > 0:
            # Available squares: not my marks, not rejected
            available = [
                i for i in range(9)
                if new_board[i] == EMPTY and i not in rejected
            ]
            if remaining_opp <= len(available):
                chosen = random.sample(available, remaining_opp)
                for sq in chosen:
                    new_board[sq] = opp_mark

        return PhantomTTTState(
            board=new_board,
            current=self._current,
            known_own=self._known_own,
            known_rejected=self._known_rejected,
            move_count=self._move_count,
        )

    def board_display(self, player: int | None = None) -> str:
        """Render the board. If player is given, show only what they know."""
        symbols = {EMPTY: ".", X_MARK: "X", O_MARK: "O"}
        if player is None:
            # Full board
            cells = [symbols[c] for c in self._board]
        else:
            my_mark = X_MARK if player == 0 else O_MARK
            cells = []
            for i in range(9):
                if i in self._known_own[player]:
                    cells.append(symbols[my_mark])
                elif i in self._known_rejected[player]:
                    cells.append("!")  # known opponent
                else:
                    cells.append("?")
        rows = []
        for r in range(3):
            rows.append(" ".join(cells[r * 3: r * 3 + 3]))
        return "\n".join(rows)

    def __repr__(self) -> str:
        symbols = {EMPTY: ".", X_MARK: "X", O_MARK: "O"}
        b = "".join(symbols[c] for c in self._board)
        return f"PhantomTTT(board={b}, turn=P{self._current}, moves={self._move_count})"
