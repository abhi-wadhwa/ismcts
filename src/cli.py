"""Command-line interface for IS-MCTS experiments."""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any

from src.core.pimc import PIMC
from src.core.so_ismcts import so_ismcts_search, so_ismcts_best_action
from src.core.mo_ismcts import mo_ismcts_best_action
from src.core.smooth_ucb import smooth_ucb_best_action
from src.core.fusion import demonstrate_strategy_fusion
from src.games.kuhn_poker import KuhnPokerState, CARD_NAMES, PASS, BET
from src.games.phantom_ttt import PhantomTTTState


def play_kuhn_poker(algorithm: str = "so_ismcts", iterations: int = 1000) -> None:
    """Interactive Kuhn Poker: human (Player 0) vs AI (Player 1)."""
    print("=" * 50)
    print("  KUHN POKER -- You are Player 0")
    print("  Cards: J(0) < Q(1) < K(2)")
    print("  Actions: pass or bet")
    print("  Each player antes 1 chip.")
    print("=" * 50)

    wins = [0, 0]
    rounds = 0

    while True:
        rounds += 1
        print(f"\n--- Round {rounds} ---")
        game = KuhnPokerState()
        print(f"Your card: {CARD_NAMES[game._hands[0]]}")

        state = game
        while not state.is_terminal():
            cp = state.current_player()
            if cp == 0:
                # Human
                while True:
                    move = input("Your action (pass/bet): ").strip().lower()
                    if move in [PASS, BET]:
                        break
                    print("Invalid. Type 'pass' or 'bet'.")
                state = state.apply_action(move)
            else:
                # AI
                action = _get_ai_action(state, algorithm, iterations)
                print(f"AI plays: {action}")
                state = state.apply_action(action)

        p0 = state.payoff(0)
        p1 = state.payoff(1)
        print(f"Opponent had: {CARD_NAMES[state._hands[1]]}")
        if p0 > 0:
            print(f"You win {p0:.0f} chips!")
            wins[0] += 1
        elif p0 < 0:
            print(f"You lose {-p0:.0f} chips.")
            wins[1] += 1
        else:
            print("Draw.")

        print(f"Score: You {wins[0]} - AI {wins[1]}")
        cont = input("Play again? (y/n): ").strip().lower()
        if cont != "y":
            break

    print(f"\nFinal score: You {wins[0]} - AI {wins[1]}")


def play_phantom_ttt(algorithm: str = "so_ismcts", iterations: int = 1000) -> None:
    """Interactive Phantom TTT: human (Player 0, X) vs AI (Player 1, O)."""
    print("=" * 50)
    print("  PHANTOM TIC-TAC-TOE -- You are X (Player 0)")
    print("  You cannot see O's moves (shown as ?).")
    print("  Squares: 0-8 (row-major)")
    print("     0 | 1 | 2")
    print("     3 | 4 | 5")
    print("     6 | 7 | 8")
    print("  If your move hits an opponent square, you'll")
    print("  see '!' and must try again.")
    print("=" * 50)

    state = PhantomTTTState()

    while not state.is_terminal():
        cp = state.current_player()
        if cp == 0:
            print(f"\nYour view:\n{state.board_display(0)}")
            while True:
                try:
                    sq = int(input("Your move (0-8): ").strip())
                    if sq in state.legal_actions():
                        break
                    print("That square is not available to you.")
                except ValueError:
                    print("Enter a number 0-8.")
            new_state = state.apply_action(sq)
            if new_state.current_player() == 0 and not new_state.is_terminal():
                # Rejection -- square was occupied by opponent
                print("Rejected! That square is occupied by opponent.")
                state = new_state
            else:
                state = new_state
        else:
            action = _get_ai_action(state, algorithm, iterations)
            new_state = state.apply_action(action)
            # AI might also get rejected in phantom TTT
            while new_state.current_player() == 1 and not new_state.is_terminal():
                action = _get_ai_action(new_state, algorithm, iterations)
                new_state = new_state.apply_action(action)
            state = new_state
            print("AI has moved.")

    print(f"\nFinal board:\n{state.board_display()}")
    p0 = state.payoff(0)
    if p0 > 0:
        print("You win!")
    elif p0 < 0:
        print("AI wins!")
    else:
        print("Draw!")


def run_fusion_demo() -> None:
    """Run and display strategy fusion analysis."""
    print("=" * 60)
    print("  STRATEGY FUSION DEMONSTRATION")
    print("=" * 60)
    print("\nRunning comparison (this may take a moment)...\n")

    results = demonstrate_strategy_fusion(
        n_pimc_det=50,
        n_pimc_iter=30,
        n_ismcts_iter=1000,
        n_trials=100,
    )

    print("PIMC estimated action values:")
    for a in ["COMMIT", "SAFE"]:
        print(f"  {a}: {results['pimc_values'].get(a, 0):.3f}")

    print("\nIS-MCTS estimated action values:")
    for a in ["COMMIT", "SAFE"]:
        print(f"  {a}: {results['ismcts_values'].get(a, 0):.3f}")

    print(f"\nPIMC action distribution: {results['pimc_actions']}")
    print(f"IS-MCTS action distribution: {results['ismcts_actions']}")
    print(f"\nPIMC actual average payoff:   {results['pimc_actual_payoff']:.3f}")
    print(f"IS-MCTS actual average payoff: {results['ismcts_actual_payoff']:.3f}")
    print(f"Optimal payoff (always SAFE):  {results['optimal_payoff']:.3f}")
    print(f"\n{results['explanation']}")


def run_benchmark(n_games: int = 100, iterations: int = 500) -> None:
    """Benchmark PIMC vs SO-ISMCTS on Kuhn Poker."""
    print(f"Benchmarking PIMC vs SO-ISMCTS on Kuhn Poker ({n_games} games)...")

    pimc = PIMC(n_determinizations=20, n_iterations_per_world=25)
    pimc_wins = 0
    ismcts_wins = 0

    for i in range(n_games):
        game = KuhnPokerState()
        state = game

        while not state.is_terminal():
            cp = state.current_player()
            if cp == 0:
                # PIMC plays as P0
                action = pimc.best_action(state)
            else:
                # SO-ISMCTS plays as P1
                action = so_ismcts_best_action(state, n_iterations=iterations)
            state = state.apply_action(action)

        if state.payoff(0) > 0:
            pimc_wins += 1
        elif state.payoff(1) > 0:
            ismcts_wins += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_games} games completed...")

    draws = n_games - pimc_wins - ismcts_wins
    print(f"\nResults ({n_games} games):")
    print(f"  PIMC (P0):      {pimc_wins} wins ({100*pimc_wins/n_games:.1f}%)")
    print(f"  SO-ISMCTS (P1): {ismcts_wins} wins ({100*ismcts_wins/n_games:.1f}%)")
    print(f"  Draws:          {draws}")


def _get_ai_action(state: Any, algorithm: str, iterations: int) -> Any:
    """Get the AI's action using the specified algorithm."""
    if algorithm == "pimc":
        pimc = PIMC(n_determinizations=20, n_iterations_per_world=iterations // 20)
        return pimc.best_action(state)
    elif algorithm == "so_ismcts":
        return so_ismcts_best_action(state, n_iterations=iterations)
    elif algorithm == "mo_ismcts":
        return mo_ismcts_best_action(state, n_iterations=iterations)
    elif algorithm == "smooth_ucb":
        return smooth_ucb_best_action(state, n_iterations=iterations)
    else:
        return so_ismcts_best_action(state, n_iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Information Set MCTS for imperfect-information games"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Play Kuhn Poker
    kuhn_parser = subparsers.add_parser("kuhn", help="Play Kuhn Poker against AI")
    kuhn_parser.add_argument(
        "--algorithm", choices=["pimc", "so_ismcts", "mo_ismcts", "smooth_ucb"],
        default="so_ismcts"
    )
    kuhn_parser.add_argument("--iterations", type=int, default=1000)

    # Play Phantom TTT
    ttt_parser = subparsers.add_parser("phantom-ttt", help="Play Phantom Tic-Tac-Toe against AI")
    ttt_parser.add_argument(
        "--algorithm", choices=["pimc", "so_ismcts", "mo_ismcts", "smooth_ucb"],
        default="so_ismcts"
    )
    ttt_parser.add_argument("--iterations", type=int, default=1000)

    # Strategy fusion demo
    subparsers.add_parser("fusion", help="Demonstrate strategy fusion pathology")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark PIMC vs IS-MCTS")
    bench_parser.add_argument("--games", type=int, default=100)
    bench_parser.add_argument("--iterations", type=int, default=500)

    args = parser.parse_args()

    if args.command == "kuhn":
        play_kuhn_poker(args.algorithm, args.iterations)
    elif args.command == "phantom-ttt":
        play_phantom_ttt(args.algorithm, args.iterations)
    elif args.command == "fusion":
        run_fusion_demo()
    elif args.command == "benchmark":
        run_benchmark(args.games, args.iterations)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
