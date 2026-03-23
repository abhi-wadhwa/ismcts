"""Quick demonstration of IS-MCTS algorithms.

Shows:
  1. PIMC vs SO-ISMCTS on Kuhn Poker
  2. AI vs Random on Phantom TTT
  3. Strategy fusion pathology
"""

from __future__ import annotations

import random

from src.core.pimc import PIMC
from src.core.so_ismcts import so_ismcts_search, so_ismcts_best_action
from src.core.mo_ismcts import mo_ismcts_best_action
from src.core.smooth_ucb import smooth_ucb_best_action
from src.core.fusion import demonstrate_strategy_fusion
from src.games.kuhn_poker import KuhnPokerState, CARD_NAMES
from src.games.phantom_ttt import PhantomTTTState
from src.games.liars_dice import LiarsDiceState


def demo_kuhn_poker() -> None:
    """Compare algorithms on Kuhn Poker."""
    print("=" * 60)
    print("  Demo 1: Kuhn Poker -- Algorithm Comparison")
    print("=" * 60)

    n_games = 200
    algorithms = {
        "PIMC": lambda s: PIMC(n_determinizations=20, n_iterations_per_world=25).best_action(s),
        "SO-ISMCTS": lambda s: so_ismcts_best_action(s, n_iterations=500),
        "MO-ISMCTS": lambda s: mo_ismcts_best_action(s, n_iterations=500),
        "Smooth UCB": lambda s: smooth_ucb_best_action(s, n_iterations=500),
        "Random": lambda s: random.choice(s.legal_actions()),
    }

    # Each algorithm plays as P0 against Random as P1
    print(f"\nEach algorithm plays {n_games} games as P0 vs Random P1:\n")
    for name, policy in algorithms.items():
        wins = 0
        total_payoff = 0.0
        for _ in range(n_games):
            state = KuhnPokerState()
            while not state.is_terminal():
                cp = state.current_player()
                if cp == 0:
                    action = policy(state)
                else:
                    action = random.choice(state.legal_actions())
                state = state.apply_action(action)
            p = state.payoff(0)
            total_payoff += p
            if p > 0:
                wins += 1
        print(f"  {name:12s}: {wins:3d} wins ({100*wins/n_games:.0f}%), "
              f"avg payoff = {total_payoff/n_games:+.3f}")


def demo_phantom_ttt() -> None:
    """AI vs Random on Phantom TTT."""
    print("\n" + "=" * 60)
    print("  Demo 2: Phantom TTT -- AI vs Random")
    print("=" * 60)

    n_games = 50
    wins = 0
    losses = 0
    draws = 0

    print(f"\nSO-ISMCTS (X, P0) vs Random (O, P1), {n_games} games:")
    for i in range(n_games):
        state = PhantomTTTState()
        while not state.is_terminal():
            cp = state.current_player()
            if cp == 0:
                action = so_ismcts_best_action(state, n_iterations=300)
            else:
                legal = state.legal_actions()
                action = random.choice(legal) if legal else None
                if action is None:
                    break
            state = state.apply_action(action)

        p = state.payoff(0)
        if p > 0:
            wins += 1
        elif p < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_games} games done...")

    print(f"\n  Wins:   {wins} ({100*wins/n_games:.0f}%)")
    print(f"  Losses: {losses} ({100*losses/n_games:.0f}%)")
    print(f"  Draws:  {draws} ({100*draws/n_games:.0f}%)")


def demo_liar_dice() -> None:
    """Quick Liar's Dice demonstration."""
    print("\n" + "=" * 60)
    print("  Demo 3: Liar's Dice -- SO-ISMCTS Sample Game")
    print("=" * 60)

    state = LiarsDiceState()
    print(f"\n  Initial state: {state}")
    print(f"  P0's die: {state._dice[0]}, P1's die: {state._dice[1]}")

    move_num = 0
    while not state.is_terminal():
        cp = state.current_player()
        action = so_ismcts_best_action(state, n_iterations=500)
        move_num += 1
        print(f"  Move {move_num}: Player {cp} -> {action}")
        state = state.apply_action(action)

    print(f"  Result: P0 payoff = {state.payoff(0):+.0f}, "
          f"P1 payoff = {state.payoff(1):+.0f}")


def demo_strategy_fusion() -> None:
    """Show strategy fusion pathology."""
    print("\n" + "=" * 60)
    print("  Demo 4: Strategy Fusion Pathology")
    print("=" * 60)

    print("\n  Running analysis (50 trials)...")
    results = demonstrate_strategy_fusion(
        n_pimc_det=50,
        n_pimc_iter=30,
        n_ismcts_iter=1000,
        n_trials=50,
    )

    print("\n  PIMC estimated action values:")
    for a in ["COMMIT", "SAFE"]:
        print(f"    {a}: {results['pimc_values'].get(a, 0):.3f}")

    print("\n  IS-MCTS estimated action values:")
    for a in ["COMMIT", "SAFE"]:
        print(f"    {a}: {results['ismcts_values'].get(a, 0):.3f}")

    print(f"\n  PIMC action choices:   {results['pimc_actions']}")
    print(f"  IS-MCTS action choices: {results['ismcts_actions']}")
    print(f"\n  PIMC actual payoff:   {results['pimc_actual_payoff']:.3f}")
    print(f"  IS-MCTS actual payoff: {results['ismcts_actual_payoff']:.3f}")
    print(f"  Optimal (always SAFE): {results['optimal_payoff']:.1f}")

    print(f"\n  {results['explanation']}")


def demo_tree_inspection() -> None:
    """Show how IS tree statistics look."""
    print("\n" + "=" * 60)
    print("  Demo 5: IS Tree Inspection")
    print("=" * 60)

    state = KuhnPokerState(hands=(2, 0))  # P0=K, P1=J
    print(f"\n  State: {state}")
    print(f"  Running SO-ISMCTS with 2000 iterations...\n")

    root = so_ismcts_search(state, n_iterations=2000)
    stats = root.get_action_stats()

    print(f"  Root visits: {root.visits}")
    print(f"  Action statistics:")
    for action, s in sorted(stats.items(), key=lambda x: -x[1]["visits"]):
        print(f"    {action:5s}: visits={s['visits']:4.0f}, "
              f"value={s['value']:+.3f}, "
              f"availability={s['availability']:.0f}")

    best = root.get_best_action()
    print(f"\n  Best action: {best}")
    print("  (With K, player should bet aggressively -- value should be high)")


if __name__ == "__main__":
    demo_kuhn_poker()
    demo_phantom_ttt()
    demo_liar_dice()
    demo_strategy_fusion()
    demo_tree_inspection()
    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("=" * 60)
