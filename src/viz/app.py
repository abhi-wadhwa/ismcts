"""Streamlit interactive UI for IS-MCTS exploration.

Features:
  1. IS Tree Viewer: visualize the information set tree after search
  2. Play against AI: Kuhn Poker or Phantom TTT
  3. Determinization Explorer: see which worlds are sampled
  4. PIMC vs IS-MCTS win rate comparison
"""

from __future__ import annotations

import random
import time
from typing import Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.core.pimc import PIMC
from src.core.so_ismcts import so_ismcts_search
from src.core.mo_ismcts import mo_ismcts_search
from src.core.smooth_ucb import smooth_ucb_search
from src.core.fusion import demonstrate_strategy_fusion, FusionGameState
from src.games.kuhn_poker import KuhnPokerState, CARD_NAMES, PASS, BET
from src.games.phantom_ttt import PhantomTTTState, X_MARK, O_MARK, EMPTY
from src.games.liars_dice import LiarsDiceState


st.set_page_config(
    page_title="IS-MCTS Explorer",
    page_icon="🌳",
    layout="wide",
)


def main() -> None:
    st.title("Information Set MCTS Explorer")
    st.markdown(
        "Explore imperfect-information game search: determinized MCTS (PIMC), "
        "SO-ISMCTS, MO-ISMCTS, Smooth UCB, and the strategy fusion pathology."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "IS Tree Viewer",
        "Play Against AI",
        "Determinization Explorer",
        "PIMC vs IS-MCTS Comparison",
    ])

    with tab1:
        render_tree_viewer()
    with tab2:
        render_play_against_ai()
    with tab3:
        render_determinization_explorer()
    with tab4:
        render_comparison()


def render_tree_viewer() -> None:
    """Visualize the IS tree after running search."""
    st.header("Information Set Tree Viewer")
    st.markdown(
        "Run IS-MCTS on a game and visualize the resulting tree. "
        "Nodes represent information sets, not individual states."
    )

    col1, col2 = st.columns(2)
    with col1:
        game_choice = st.selectbox(
            "Game", ["Kuhn Poker", "Phantom TTT", "Fusion Game"],
            key="tree_game",
        )
    with col2:
        algo_choice = st.selectbox(
            "Algorithm", ["SO-ISMCTS", "MO-ISMCTS", "Smooth UCB"],
            key="tree_algo",
        )

    iterations = st.slider("Iterations", 100, 5000, 1000, step=100, key="tree_iter")

    if st.button("Run Search", key="tree_run"):
        if game_choice == "Kuhn Poker":
            state = KuhnPokerState()
            st.write(f"Dealt: Player 0 has **{CARD_NAMES[state._hands[0]]}**, "
                     f"Player 1 has **{CARD_NAMES[state._hands[1]]}**")
        elif game_choice == "Phantom TTT":
            state = PhantomTTTState()
        else:
            state = FusionGameState()

        with st.spinner("Running search..."):
            if algo_choice == "SO-ISMCTS":
                root = so_ismcts_search(state, n_iterations=iterations)
            elif algo_choice == "MO-ISMCTS":
                roots = mo_ismcts_search(state, n_iterations=iterations)
                root = roots[state.current_player()]
            else:
                root = smooth_ucb_search(state, n_iterations=iterations)

        # Display tree statistics
        st.subheader("Root Node Statistics")
        stats = root.get_action_stats()
        if stats:
            chart_data = {
                "Action": [],
                "Visits": [],
                "Value": [],
            }
            for action, s in stats.items():
                chart_data["Action"].append(str(action))
                chart_data["Visits"].append(s["visits"])
                chart_data["Value"].append(s["value"])

            col_a, col_b = st.columns(2)
            with col_a:
                fig_visits = px.bar(
                    x=chart_data["Action"],
                    y=chart_data["Visits"],
                    title="Visit Counts per Action",
                    labels={"x": "Action", "y": "Visits"},
                )
                st.plotly_chart(fig_visits, use_container_width=True)
            with col_b:
                fig_value = px.bar(
                    x=chart_data["Action"],
                    y=chart_data["Value"],
                    title="Average Value per Action",
                    labels={"x": "Action", "y": "Value"},
                    color=chart_data["Value"],
                    color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(fig_value, use_container_width=True)

            best = root.get_best_action()
            st.success(f"Best action: **{best}** "
                       f"(visits: {stats[best]['visits']}, "
                       f"value: {stats[best]['value']:.3f})")

        # Show tree structure (BFS up to depth 3)
        st.subheader("Tree Structure (depth <= 3)")
        _render_tree_bfs(root, max_depth=3)


def _render_tree_bfs(root: Any, max_depth: int = 3) -> None:
    """Render tree structure as an expandable list."""
    queue = [(root, 0, "Root")]
    lines = []
    while queue:
        node, depth, label = queue.pop(0)
        if depth > max_depth:
            continue
        indent = "  " * depth
        v = node.visits
        val = node.total_value / v if v > 0 else 0.0
        lines.append(f"{indent}{label} [visits={v}, value={val:.3f}]")
        if hasattr(node, "children"):
            for action, child in node.children.items():
                queue.append((child, depth + 1, f"Action: {action}"))

    st.code("\n".join(lines) if lines else "(empty tree)")


def render_play_against_ai() -> None:
    """Play Kuhn Poker or Phantom TTT against the AI."""
    st.header("Play Against AI")

    game_choice = st.radio(
        "Choose game:", ["Kuhn Poker", "Phantom Tic-Tac-Toe"],
        key="play_game", horizontal=True,
    )

    algo = st.selectbox(
        "AI Algorithm",
        ["SO-ISMCTS", "PIMC", "MO-ISMCTS", "Smooth UCB"],
        key="play_algo",
    )
    ai_iterations = st.slider("AI iterations", 100, 3000, 500, step=100, key="play_iter")

    if game_choice == "Kuhn Poker":
        _play_kuhn_ui(algo, ai_iterations)
    else:
        _play_phantom_ttt_ui(algo, ai_iterations)


def _play_kuhn_ui(algo: str, iterations: int) -> None:
    """Kuhn Poker interactive UI."""
    if "kuhn_state" not in st.session_state:
        st.session_state.kuhn_state = None
        st.session_state.kuhn_score = [0, 0]
        st.session_state.kuhn_round = 0
        st.session_state.kuhn_messages = []

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Deal New Hand", key="kuhn_deal"):
            st.session_state.kuhn_state = KuhnPokerState()
            st.session_state.kuhn_round += 1
            st.session_state.kuhn_messages = [
                f"Round {st.session_state.kuhn_round}: "
                f"Your card is **{CARD_NAMES[st.session_state.kuhn_state._hands[0]]}**"
            ]
    with col2:
        st.write(f"Score -- You: {st.session_state.kuhn_score[0]} | "
                 f"AI: {st.session_state.kuhn_score[1]}")

    state = st.session_state.kuhn_state
    if state is None:
        st.info("Click 'Deal New Hand' to start.")
        return

    for msg in st.session_state.kuhn_messages:
        st.write(msg)

    if state.is_terminal():
        p0 = state.payoff(0)
        st.write(f"Opponent had: **{CARD_NAMES[state._hands[1]]}**")
        if p0 > 0:
            st.success(f"You win {p0:.0f} chips!")
        elif p0 < 0:
            st.error(f"You lose {-p0:.0f} chips.")
        else:
            st.info("Draw!")
        return

    cp = state.current_player()
    if cp == 0:
        col_pass, col_bet = st.columns(2)
        with col_pass:
            if st.button("Pass", key="kuhn_pass"):
                new_state = state.apply_action(PASS)
                st.session_state.kuhn_messages.append("You: **pass**")
                # Let AI respond if not terminal
                if not new_state.is_terminal() and new_state.current_player() == 1:
                    ai_action = _get_ai_action_ui(new_state, algo, iterations)
                    new_state = new_state.apply_action(ai_action)
                    st.session_state.kuhn_messages.append(f"AI: **{ai_action}**")
                if new_state.is_terminal():
                    p0 = new_state.payoff(0)
                    if p0 > 0:
                        st.session_state.kuhn_score[0] += 1
                    elif p0 < 0:
                        st.session_state.kuhn_score[1] += 1
                st.session_state.kuhn_state = new_state
                st.rerun()
        with col_bet:
            if st.button("Bet", key="kuhn_bet"):
                new_state = state.apply_action(BET)
                st.session_state.kuhn_messages.append("You: **bet**")
                if not new_state.is_terminal() and new_state.current_player() == 1:
                    ai_action = _get_ai_action_ui(new_state, algo, iterations)
                    new_state = new_state.apply_action(ai_action)
                    st.session_state.kuhn_messages.append(f"AI: **{ai_action}**")
                if new_state.is_terminal():
                    p0 = new_state.payoff(0)
                    if p0 > 0:
                        st.session_state.kuhn_score[0] += 1
                    elif p0 < 0:
                        st.session_state.kuhn_score[1] += 1
                st.session_state.kuhn_state = new_state
                st.rerun()
    else:
        # AI's turn
        ai_action = _get_ai_action_ui(state, algo, iterations)
        new_state = state.apply_action(ai_action)
        st.session_state.kuhn_messages.append(f"AI: **{ai_action}**")
        if new_state.is_terminal():
            p0 = new_state.payoff(0)
            if p0 > 0:
                st.session_state.kuhn_score[0] += 1
            elif p0 < 0:
                st.session_state.kuhn_score[1] += 1
        st.session_state.kuhn_state = new_state
        st.rerun()


def _play_phantom_ttt_ui(algo: str, iterations: int) -> None:
    """Phantom TTT interactive UI."""
    if "ttt_state" not in st.session_state:
        st.session_state.ttt_state = None
        st.session_state.ttt_messages = []

    if st.button("New Game", key="ttt_new"):
        st.session_state.ttt_state = PhantomTTTState()
        st.session_state.ttt_messages = ["Game started. You are X (Player 0)."]

    state = st.session_state.ttt_state
    if state is None:
        st.info("Click 'New Game' to start.")
        return

    for msg in st.session_state.ttt_messages:
        st.write(msg)

    if state.is_terminal():
        st.subheader("Final Board:")
        _render_board(state, None)
        p0 = state.payoff(0)
        if p0 > 0:
            st.success("You win!")
        elif p0 < 0:
            st.error("AI wins!")
        else:
            st.info("Draw!")
        return

    cp = state.current_player()
    if cp == 0:
        st.subheader("Your View:")
        _render_board(state, 0)

        legal = state.legal_actions()
        st.write(f"Available squares: {legal}")
        move = st.selectbox("Pick a square:", legal, key="ttt_move")
        if st.button("Place", key="ttt_place"):
            new_state = state.apply_action(move)
            if new_state.current_player() == 0 and not new_state.is_terminal():
                st.session_state.ttt_messages.append(
                    f"Square {move} rejected! Opponent is there."
                )
                st.session_state.ttt_state = new_state
            else:
                st.session_state.ttt_messages.append(f"You placed X on square {move}.")
                # AI turn
                if not new_state.is_terminal():
                    ai_state = new_state
                    while ai_state.current_player() == 1 and not ai_state.is_terminal():
                        ai_action = _get_ai_action_ui(ai_state, algo, iterations)
                        ai_state = ai_state.apply_action(ai_action)
                    st.session_state.ttt_messages.append("AI has moved.")
                    new_state = ai_state
                st.session_state.ttt_state = new_state
            st.rerun()
    else:
        # AI turn
        ai_state = state
        while ai_state.current_player() == 1 and not ai_state.is_terminal():
            ai_action = _get_ai_action_ui(ai_state, algo, iterations)
            ai_state = ai_state.apply_action(ai_action)
        st.session_state.ttt_messages.append("AI has moved.")
        st.session_state.ttt_state = ai_state
        st.rerun()


def _render_board(state: PhantomTTTState, player: int | None) -> None:
    """Render the TTT board as a grid."""
    symbols = {EMPTY: ".", X_MARK: "X", O_MARK: "O"}
    if player is None:
        cells = [symbols[c] for c in state._board]
    else:
        my_mark = X_MARK if player == 0 else O_MARK
        cells = []
        for i in range(9):
            if i in state._known_own[player]:
                cells.append(symbols[my_mark])
            elif i in state._known_rejected[player]:
                cells.append("!")
            else:
                cells.append("?")

    grid = ""
    for r in range(3):
        row = " | ".join(cells[r * 3: r * 3 + 3])
        grid += f"  {row}\n"
        if r < 2:
            grid += "  ---------\n"
    st.code(grid)


def render_determinization_explorer() -> None:
    """Show what determinizations look like for different games."""
    st.header("Determinization Explorer")
    st.markdown(
        "See how hidden information is sampled. Each determinization fills in "
        "unknown information randomly while staying consistent with what the "
        "observer knows."
    )

    game_choice = st.selectbox(
        "Game", ["Kuhn Poker", "Phantom TTT", "Liar's Dice"],
        key="det_game",
    )
    n_samples = st.slider("Number of determinizations", 5, 50, 10, key="det_n")

    if st.button("Sample Determinizations", key="det_run"):
        if game_choice == "Kuhn Poker":
            state = KuhnPokerState()
            observer = 0
            st.write(f"**True state:** {state}")
            st.write(f"**Observer (Player {observer}) sees card:** "
                     f"{CARD_NAMES[state._hands[observer]]}")
            st.write(f"**Observer does NOT know opponent's card.**")

            samples = []
            for i in range(n_samples):
                det = state.determinize(observer)
                samples.append({
                    "Sample": i + 1,
                    "My Card": CARD_NAMES[det._hands[observer]],
                    "Opponent Card (sampled)": CARD_NAMES[det._hands[1 - observer]],
                })
            st.table(samples)

            # Show distribution
            opp_cards = [s["Opponent Card (sampled)"] for s in samples]
            fig = px.histogram(x=opp_cards, title="Sampled Opponent Cards",
                               labels={"x": "Card"})
            st.plotly_chart(fig, use_container_width=True)

        elif game_choice == "Phantom TTT":
            # Set up a state with some moves played
            state = PhantomTTTState()
            # Simulate a few random moves
            for _ in range(3):
                if state.is_terminal():
                    break
                legal = state.legal_actions()
                if legal:
                    state = state.apply_action(random.choice(legal))

            observer = 0
            st.write(f"**True board (hidden):**")
            st.code(state.board_display(None))
            st.write(f"**Player {observer}'s view:**")
            st.code(state.board_display(observer))

            st.write("**Sampled determinizations:**")
            for i in range(min(n_samples, 5)):
                det = state.determinize(observer)
                st.write(f"Sample {i + 1}:")
                st.code(det.board_display(None))

        else:  # Liar's Dice
            state = LiarsDiceState()
            observer = 0
            st.write(f"**True state:** {state}")
            st.write(f"**Observer (Player {observer}) sees die:** {state._dice[observer]}")

            samples = []
            for i in range(n_samples):
                det = state.determinize(observer)
                samples.append({
                    "Sample": i + 1,
                    "My Die": det._dice[observer],
                    "Opponent Die (sampled)": det._dice[1 - observer],
                })
            st.table(samples)

            opp_dice = [s["Opponent Die (sampled)"] for s in samples]
            fig = px.histogram(x=opp_dice, title="Sampled Opponent Dice",
                               labels={"x": "Die Value"}, nbins=6)
            st.plotly_chart(fig, use_container_width=True)


def render_comparison() -> None:
    """Compare PIMC vs IS-MCTS win rates."""
    st.header("PIMC vs IS-MCTS Comparison")
    st.markdown(
        "Run multiple games between PIMC and IS-MCTS to compare their "
        "performance. Also demonstrates the strategy fusion pathology."
    )

    tab_a, tab_b = st.tabs(["Win Rate Comparison", "Strategy Fusion Demo"])

    with tab_a:
        game_choice = st.selectbox(
            "Game", ["Kuhn Poker", "Phantom TTT"],
            key="cmp_game",
        )
        n_games = st.slider("Number of games", 10, 200, 50, step=10, key="cmp_n")
        iterations = st.slider("Iterations per move", 100, 2000, 500, step=100, key="cmp_iter")

        if st.button("Run Comparison", key="cmp_run"):
            progress = st.progress(0)
            pimc_wins = 0
            ismcts_wins = 0
            draws = 0
            pimc = PIMC(n_determinizations=20, n_iterations_per_world=max(1, iterations // 20))

            for i in range(n_games):
                if game_choice == "Kuhn Poker":
                    state = KuhnPokerState()
                else:
                    state = PhantomTTTState()

                while not state.is_terminal():
                    cp = state.current_player()
                    if cp == 0:
                        action = pimc.best_action(state)
                    else:
                        root = so_ismcts_search(state, n_iterations=iterations)
                        best = root.get_best_action()
                        action = best if best is not None else random.choice(state.legal_actions())
                    state = state.apply_action(action)

                p0 = state.payoff(0)
                if p0 > 0:
                    pimc_wins += 1
                elif p0 < 0:
                    ismcts_wins += 1
                else:
                    draws += 1

                progress.progress((i + 1) / n_games)

            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("PIMC Wins", pimc_wins)
            col2.metric("IS-MCTS Wins", ismcts_wins)
            col3.metric("Draws", draws)

            fig = go.Figure(data=[
                go.Bar(name="PIMC (P0)", x=["Wins"], y=[pimc_wins], marker_color="steelblue"),
                go.Bar(name="IS-MCTS (P1)", x=["Wins"], y=[ismcts_wins], marker_color="coral"),
                go.Bar(name="Draws", x=["Wins"], y=[draws], marker_color="gray"),
            ])
            fig.update_layout(title=f"Win Rates over {n_games} games", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        st.subheader("Strategy Fusion Pathology")
        st.markdown("""
        **The Setup:** A hidden card is dealt (A or B, 50/50). The player makes two decisions:

        **Step 1:** Choose COMMIT or SAFE.
        - SAFE ends the game with payoff **+0.4** (regardless of card).
        - COMMIT leads to Step 2.

        **Step 2 (after COMMIT):** Choose LEFT or RIGHT.
        - Card A: LEFT=+1, RIGHT=-1
        - Card B: LEFT=-1, RIGHT=+1

        **Optimal strategy:** SAFE (+0.4), since after COMMIT the player still cannot
        see the card, so LEFT/RIGHT each yield expected value 0.0.

        **PIMC's mistake:** In each determinization, PIMC sees the card and finds
        COMMIT followed by the correct LEFT/RIGHT gives +1.0 > SAFE's +0.4.
        It reports COMMIT as having value ~1.0 -- but the player cannot condition
        their step-2 choice on the hidden card! This is **strategy fusion**.

        **IS-MCTS** correctly maintains a single tree over information sets and
        discovers that SAFE dominates.
        """)

        n_trials = st.slider("Number of trials", 20, 500, 100, step=20, key="fusion_n")
        if st.button("Run Fusion Demo", key="fusion_run"):
            with st.spinner("Running strategy fusion analysis..."):
                results = demonstrate_strategy_fusion(
                    n_pimc_det=50,
                    n_pimc_iter=30,
                    n_ismcts_iter=1000,
                    n_trials=n_trials,
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("PIMC")
                st.write("Estimated action values:")
                for a in ["COMMIT", "SAFE"]:
                    st.write(f"  {a}: {results['pimc_values'].get(a, 0):.3f}")
                st.write(f"Action distribution: {results['pimc_actions']}")
                st.metric("Actual Average Payoff",
                          f"{results['pimc_actual_payoff']:.3f}")

            with col2:
                st.subheader("IS-MCTS")
                st.write("Estimated action values:")
                for a in ["COMMIT", "SAFE"]:
                    st.write(f"  {a}: {results['ismcts_values'].get(a, 0):.3f}")
                st.write(f"Action distribution: {results['ismcts_actions']}")
                st.metric("Actual Average Payoff",
                          f"{results['ismcts_actual_payoff']:.3f}")

            st.info(f"Optimal payoff (always SAFE): {results['optimal_payoff']:.1f}")

            # Bar chart comparison
            actions = ["COMMIT", "SAFE"]
            fig = go.Figure(data=[
                go.Bar(
                    name="PIMC",
                    x=actions,
                    y=[results["pimc_values"].get(a, 0) for a in actions],
                    marker_color="steelblue",
                ),
                go.Bar(
                    name="IS-MCTS",
                    x=actions,
                    y=[results["ismcts_values"].get(a, 0) for a in actions],
                    marker_color="coral",
                ),
            ])
            fig.update_layout(
                title="Estimated Action Values: PIMC vs IS-MCTS",
                xaxis_title="Action",
                yaxis_title="Estimated Value",
                barmode="group",
            )
            fig.add_hline(y=0.4, line_dash="dash", line_color="green",
                          annotation_text="Optimal (SAFE=0.4)")
            st.plotly_chart(fig, use_container_width=True)


def _get_ai_action_ui(state: Any, algo: str, iterations: int) -> Any:
    """Get AI action for the UI."""
    if algo == "PIMC":
        pimc = PIMC(n_determinizations=20, n_iterations_per_world=max(1, iterations // 20))
        return pimc.best_action(state)
    elif algo == "SO-ISMCTS":
        root = so_ismcts_search(state, n_iterations=iterations)
        best = root.get_best_action()
        return best if best is not None else random.choice(state.legal_actions())
    elif algo == "MO-ISMCTS":
        from src.core.mo_ismcts import mo_ismcts_best_action
        return mo_ismcts_best_action(state, n_iterations=iterations)
    else:  # Smooth UCB
        from src.core.smooth_ucb import smooth_ucb_best_action
        return smooth_ucb_best_action(state, n_iterations=iterations)


if __name__ == "__main__":
    main()
