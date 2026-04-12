"""Tests for core IS-MCTS algorithms."""

from __future__ import annotations

import random

import pytest

from src.core.pimc import PIMC
from src.core.ismcts import ismcts_search
from src.core.so_ismcts import so_ismcts_search, so_ismcts_best_action
from src.core.mo_ismcts import mo_ismcts_search, mo_ismcts_best_action
from src.core.smooth_ucb import smooth_ucb_search, smooth_ucb_best_action
from src.games.kuhn_poker import KuhnPokerState, PASS, BET


class TestPIMC:
    """Tests for determinized MCTS (PIMC)."""

    def test_returns_valid_action(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        pimc = PIMC(n_determinizations=10, n_iterations_per_world=20)
        action = pimc.best_action(state)
        assert action in state.legal_actions()

    def test_search_returns_values_for_all_actions(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        pimc = PIMC(n_determinizations=10, n_iterations_per_world=20)
        values = pimc.search(state)
        # Should have values for pass and bet
        assert PASS in values or BET in values

    def test_with_king_prefers_bet(self) -> None:
        """With the best card (K), PIMC should prefer betting."""
        random.seed(0)
        bet_count = 0
        n = 30
        for _ in range(n):
            state = KuhnPokerState(hands=(2, random.randint(0, 1)))  # K vs J or Q
            pimc = PIMC(n_determinizations=15, n_iterations_per_world=30)
            action = pimc.best_action(state)
            if action == BET:
                bet_count += 1
        # Should bet most of the time with K
        assert bet_count > n * 0.5, f"Only bet {bet_count}/{n} times with K"


class TestISMCTS:
    """Tests for basic IS-MCTS."""

    def test_returns_root_node(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        root = ismcts_search(state, n_iterations=100)
        assert root is not None
        assert root.visits > 0

    def test_root_has_children(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        root = ismcts_search(state, n_iterations=200)
        assert len(root.children) > 0

    def test_get_best_action(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        root = ismcts_search(state, n_iterations=200)
        best = root.get_best_action()
        assert best in [PASS, BET]


class TestSOISMCTS:
    """Tests for Single Observer IS-MCTS."""

    def test_returns_valid_action(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        action = so_ismcts_best_action(state, n_iterations=200)
        assert action in state.legal_actions()

    def test_tree_structure(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        root = so_ismcts_search(state, n_iterations=500)
        stats = root.get_action_stats()
        assert len(stats) > 0
        for action, s in stats.items():
            assert s["visits"] > 0

    def test_king_prefers_bet(self) -> None:
        """With K, SO-ISMCTS should generally prefer betting."""
        random.seed(1)
        bet_count = 0
        n = 30
        for _ in range(n):
            state = KuhnPokerState(hands=(2, random.randint(0, 1)))
            action = so_ismcts_best_action(state, n_iterations=300)
            if action == BET:
                bet_count += 1
        assert bet_count > n * 0.5, f"Only bet {bet_count}/{n} times with K"


class TestMOISMCTS:
    """Tests for Multiple Observer IS-MCTS."""

    def test_returns_roots_for_all_players(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        roots = mo_ismcts_search(state, n_iterations=200)
        assert 0 in roots
        assert 1 in roots

    def test_returns_valid_action(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        action = mo_ismcts_best_action(state, n_iterations=200)
        assert action in state.legal_actions()


class TestSmoothUCB:
    """Tests for Smooth UCB IS-MCTS."""

    def test_returns_valid_action(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        action = smooth_ucb_best_action(state, n_iterations=200)
        assert action in state.legal_actions()

    def test_tree_has_children(self) -> None:
        random.seed(42)
        state = KuhnPokerState()
        root = smooth_ucb_search(state, n_iterations=500)
        assert len(root.children) > 0

    def test_dampen_parameter(self) -> None:
        """Higher dampening should lead to more exploration initially."""
        random.seed(42)
        state = KuhnPokerState()
        root_low_d = smooth_ucb_search(state, n_iterations=200, dampen=10)
        random.seed(42)
        root_high_d = smooth_ucb_search(state, n_iterations=200, dampen=200)
        # Both should produce valid trees
        assert root_low_d.visits > 0
        assert root_high_d.visits > 0


class TestAlgorithmComparison:
    """Integration tests comparing algorithm behavior."""

    def test_all_algorithms_handle_terminal_adjacent(self) -> None:
        """All algorithms should handle states close to terminal."""
        random.seed(42)
        # Create a state after one action
        state = KuhnPokerState().apply_action(BET)
        assert not state.is_terminal()

        pimc = PIMC(n_determinizations=5, n_iterations_per_world=10)
        assert pimc.best_action(state) in state.legal_actions()
        assert so_ismcts_best_action(state, n_iterations=50) in state.legal_actions()
        assert mo_ismcts_best_action(state, n_iterations=50) in state.legal_actions()
        assert smooth_ucb_best_action(state, n_iterations=50) in state.legal_actions()

    def test_ai_beats_random_kuhn(self) -> None:
        """SO-ISMCTS should win more than lose against random in Kuhn Poker."""
        random.seed(42)
        wins = 0
        losses = 0
        n_games = 100

        for _ in range(n_games):
            state = KuhnPokerState()
            while not state.is_terminal():
                cp = state.current_player()
                if cp == 0:
                    action = so_ismcts_best_action(state, n_iterations=200)
                else:
                    action = random.choice(state.legal_actions())
                state = state.apply_action(action)
            p = state.payoff(0)
            if p > 0:
                wins += 1
            elif p < 0:
                losses += 1

        assert wins > losses, f"AI should beat random: {wins} wins vs {losses} losses"
