"""Tests for Kuhn Poker game implementation."""

from __future__ import annotations

import random

import pytest

from src.games.kuhn_poker import KuhnPokerState, CARD_NAMES, PASS, BET, CARDS


class TestKuhnPokerRules:
    """Test that Kuhn Poker rules are correctly implemented."""

    def test_initial_state(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        assert state.n_players == 2
        assert state.current_player() == 0
        assert not state.is_terminal()
        assert state.legal_actions() == [PASS, BET]

    def test_hands_are_valid(self) -> None:
        random.seed(42)
        for _ in range(20):
            state = KuhnPokerState()
            assert state._hands[0] in CARDS
            assert state._hands[1] in CARDS
            assert state._hands[0] != state._hands[1]

    def test_pass_pass_showdown(self) -> None:
        """Pass-Pass: showdown with pot = 2."""
        state = KuhnPokerState(hands=(2, 0))  # K vs J
        state = state.apply_action(PASS)
        state = state.apply_action(PASS)
        assert state.is_terminal()
        assert state.payoff(0) == 1.0   # K wins, net gain = 2 - 1 = 1
        assert state.payoff(1) == -1.0  # J loses ante

    def test_pass_bet_pass_fold(self) -> None:
        """Pass-Bet-Pass: P0 folds, P1 wins."""
        state = KuhnPokerState(hands=(0, 2))  # J vs K
        state = state.apply_action(PASS)
        state = state.apply_action(BET)
        state = state.apply_action(PASS)
        assert state.is_terminal()
        assert state.payoff(0) == -1.0  # P0 folded, lost ante
        assert state.payoff(1) == 1.0   # P1 wins pot

    def test_pass_bet_bet_showdown(self) -> None:
        """Pass-Bet-Bet: showdown with pot = 4."""
        state = KuhnPokerState(hands=(2, 0))  # K vs J
        state = state.apply_action(PASS)
        state = state.apply_action(BET)
        state = state.apply_action(BET)
        assert state.is_terminal()
        assert state.payoff(0) == 2.0   # K wins, gains 2 from opponent
        assert state.payoff(1) == -2.0  # J loses bet + ante

    def test_bet_pass_fold(self) -> None:
        """Bet-Pass: P1 folds, P0 wins."""
        state = KuhnPokerState(hands=(1, 2))  # Q vs K
        state = state.apply_action(BET)
        state = state.apply_action(PASS)
        assert state.is_terminal()
        assert state.payoff(0) == 1.0   # P0 wins P1's ante
        assert state.payoff(1) == -1.0

    def test_bet_bet_showdown(self) -> None:
        """Bet-Bet: showdown with pot = 4."""
        state = KuhnPokerState(hands=(0, 2))  # J vs K
        state = state.apply_action(BET)
        state = state.apply_action(BET)
        assert state.is_terminal()
        assert state.payoff(0) == -2.0  # J loses
        assert state.payoff(1) == 2.0   # K wins

    def test_current_player_sequence(self) -> None:
        state = KuhnPokerState(hands=(1, 0))
        assert state.current_player() == 0
        state = state.apply_action(PASS)
        assert state.current_player() == 1
        state = state.apply_action(BET)
        assert state.current_player() == 0  # P0 responds to bet

    def test_information_set(self) -> None:
        """Same card + history = same info set, regardless of opponent's card."""
        s1 = KuhnPokerState(hands=(1, 0))  # Q vs J
        s2 = KuhnPokerState(hands=(1, 2))  # Q vs K
        assert s1.information_set_key(0) == s2.information_set_key(0)
        assert s1.information_set_key(1) != s2.information_set_key(1)

    def test_determinize(self) -> None:
        """Determinization preserves observer's card, randomizes opponent."""
        random.seed(42)
        state = KuhnPokerState(hands=(2, 0))  # K vs J
        for _ in range(20):
            det = state.determinize(0)
            assert det._hands[0] == 2  # Observer keeps K
            assert det._hands[1] in [0, 1]  # Opponent gets J or Q
            assert det._hands[1] != det._hands[0]

    def test_clone(self) -> None:
        state = KuhnPokerState(hands=(1, 2))
        clone = state.clone()
        assert clone._hands == state._hands
        assert clone._history == state._history


class TestKuhnPokerISMCTS:
    """Test IS-MCTS behavior on Kuhn Poker."""

    def test_ismcts_approaches_good_play(self) -> None:
        """With enough iterations, IS-MCTS should make reasonable decisions.

        With K (best card), betting is dominant.
        With J (worst card), passing is usually correct.
        """
        from src.core.so_ismcts import so_ismcts_search

        # King should bet
        random.seed(42)
        king_bets = 0
        n = 30
        for _ in range(n):
            state = KuhnPokerState(hands=(2, random.randint(0, 1)))
            root = so_ismcts_search(state, n_iterations=500)
            best = root.get_best_action()
            if best == BET:
                king_bets += 1
        assert king_bets > n * 0.5, f"K should bet often: {king_bets}/{n}"

    def test_zero_sum_payoffs(self) -> None:
        """Payoffs should sum to zero in all terminal states."""
        for h0 in CARDS:
            for h1 in CARDS:
                if h0 == h1:
                    continue
                for seq in [(PASS, PASS), (BET, PASS), (BET, BET),
                            (PASS, BET, PASS), (PASS, BET, BET)]:
                    state = KuhnPokerState(hands=(h0, h1))
                    for a in seq:
                        state = state.apply_action(a)
                    assert state.is_terminal()
                    assert state.payoff(0) + state.payoff(1) == 0.0
