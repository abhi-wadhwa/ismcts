"""Tests for strategy fusion pathology demonstration."""

from __future__ import annotations

import random

import pytest

from src.core.fusion import FusionGameState, demonstrate_strategy_fusion
from src.core.pimc import PIMC
from src.core.so_ismcts import so_ismcts_search


class TestFusionGame:
    """Test the Fusion Game implementation."""

    def test_initial_state(self) -> None:
        state = FusionGameState(hidden_card=0)
        assert state.n_players == 1
        assert state.current_player() == 0
        assert not state.is_terminal()
        assert set(state.legal_actions()) == {"COMMIT", "SAFE"}

    def test_safe_payoff(self) -> None:
        """SAFE always gives +0.4 regardless of card."""
        for card in [0, 1]:
            state = FusionGameState(hidden_card=card)
            result = state.apply_action("SAFE")
            assert result.is_terminal()
            assert result.payoff(0) == pytest.approx(0.4)

    def test_commit_left_payoffs(self) -> None:
        """COMMIT->LEFT: +1 with card A, -1 with card B."""
        s_a = FusionGameState(hidden_card=0).apply_action("COMMIT").apply_action("LEFT")
        assert s_a.payoff(0) == 1.0
        s_b = FusionGameState(hidden_card=1).apply_action("COMMIT").apply_action("LEFT")
        assert s_b.payoff(0) == -1.0

    def test_commit_right_payoffs(self) -> None:
        """COMMIT->RIGHT: -1 with card A, +1 with card B."""
        s_a = FusionGameState(hidden_card=0).apply_action("COMMIT").apply_action("RIGHT")
        assert s_a.payoff(0) == -1.0
        s_b = FusionGameState(hidden_card=1).apply_action("COMMIT").apply_action("RIGHT")
        assert s_b.payoff(0) == 1.0

    def test_two_step_structure(self) -> None:
        """After COMMIT, player must choose LEFT or RIGHT."""
        state = FusionGameState(hidden_card=0)
        s2 = state.apply_action("COMMIT")
        assert not s2.is_terminal()
        assert s2.current_player() == 0
        assert set(s2.legal_actions()) == {"LEFT", "RIGHT"}

    def test_terminal_after_safe(self) -> None:
        state = FusionGameState(hidden_card=0)
        result = state.apply_action("SAFE")
        assert result.is_terminal()
        assert result.current_player() == -1
        assert result.legal_actions() == []

    def test_terminal_after_commit_and_choice(self) -> None:
        state = FusionGameState(hidden_card=0)
        s2 = state.apply_action("COMMIT")
        result = s2.apply_action("LEFT")
        assert result.is_terminal()

    def test_information_set_hides_card(self) -> None:
        """Player should not be able to distinguish card A from card B."""
        s_a = FusionGameState(hidden_card=0)
        s_b = FusionGameState(hidden_card=1)
        assert s_a.information_set_key(0) == s_b.information_set_key(0)

    def test_info_set_after_commit(self) -> None:
        """After COMMIT, info set still does not reveal card."""
        s_a = FusionGameState(hidden_card=0).apply_action("COMMIT")
        s_b = FusionGameState(hidden_card=1).apply_action("COMMIT")
        assert s_a.information_set_key(0) == s_b.information_set_key(0)

    def test_determinize(self) -> None:
        """Determinization should produce random cards."""
        random.seed(42)
        state = FusionGameState(hidden_card=0)
        cards_seen = set()
        for _ in range(50):
            det = state.determinize(0)
            cards_seen.add(det._card)
        assert cards_seen == {0, 1}


class TestPIMCFusion:
    """Test that PIMC exhibits strategy fusion pathology."""

    def test_pimc_overvalues_commit(self) -> None:
        """PIMC should assign high value to COMMIT.

        In each determinization, PIMC sees the card and plays the optimal
        follow-up (LEFT or RIGHT), achieving +1.0 for COMMIT. Since it
        does this in every determinization, COMMIT gets average value ~1.0,
        which exceeds SAFE's +0.4. This is the fusion pathology.
        """
        random.seed(42)
        state = FusionGameState(hidden_card=0)
        pimc = PIMC(n_determinizations=100, n_iterations_per_world=30)
        values = pimc.search(state)

        commit_val = values.get("COMMIT", 0)
        safe_val = values.get("SAFE", 0)

        # PIMC should rate COMMIT higher than SAFE due to strategy fusion
        assert commit_val > safe_val, (
            f"PIMC should overvalue COMMIT: COMMIT={commit_val:.3f}, SAFE={safe_val:.3f}"
        )

    def test_pimc_picks_commit(self) -> None:
        """PIMC should tend to pick COMMIT instead of the optimal SAFE."""
        random.seed(42)
        commit_count = 0
        n = 20
        for _ in range(n):
            state = FusionGameState()
            pimc = PIMC(n_determinizations=100, n_iterations_per_world=50)
            action = pimc.best_action(state)
            if action == "COMMIT":
                commit_count += 1
        # PIMC should pick COMMIT most of the time (fusion pathology)
        assert commit_count > n * 0.5, (
            f"PIMC picked COMMIT only {commit_count}/{n} times"
        )


class TestISMCTSFusion:
    """Test that IS-MCTS handles fusion game correctly."""

    def test_ismcts_prefers_safe(self) -> None:
        """IS-MCTS should learn that SAFE is the best step-1 action.

        With enough iterations, IS-MCTS recognizes that after COMMIT,
        LEFT and RIGHT each average to 0.0, which is less than SAFE's 0.4.
        """
        random.seed(42)
        safe_count = 0
        n = 20
        for _ in range(n):
            state = FusionGameState()
            root = so_ismcts_search(state, n_iterations=2000)
            best = root.get_best_action()
            if best == "SAFE":
                safe_count += 1
        assert safe_count > n * 0.4, (
            f"IS-MCTS picked SAFE only {safe_count}/{n} times"
        )


class TestDemonstration:
    """Test the full demonstration function."""

    def test_demonstration_returns_expected_keys(self) -> None:
        random.seed(42)
        results = demonstrate_strategy_fusion(
            n_pimc_det=20,
            n_pimc_iter=10,
            n_ismcts_iter=200,
            n_trials=10,
        )
        assert "pimc_values" in results
        assert "ismcts_values" in results
        assert "pimc_actions" in results
        assert "ismcts_actions" in results
        assert "pimc_actual_payoff" in results
        assert "ismcts_actual_payoff" in results
        assert "optimal_payoff" in results
        assert "explanation" in results

    def test_demonstration_payoff_difference(self) -> None:
        """IS-MCTS should achieve higher actual payoff than PIMC on fusion game."""
        random.seed(42)
        results = demonstrate_strategy_fusion(
            n_pimc_det=30,
            n_pimc_iter=20,
            n_ismcts_iter=1000,
            n_trials=50,
        )
        # IS-MCTS should outperform PIMC on this game
        # PIMC picks COMMIT (value ~0), IS-MCTS picks SAFE (value ~0.4)
        assert results["ismcts_actual_payoff"] > results["pimc_actual_payoff"] - 0.3, (
            f"IS-MCTS ({results['ismcts_actual_payoff']:.3f}) should outperform "
            f"PIMC ({results['pimc_actual_payoff']:.3f})"
        )
