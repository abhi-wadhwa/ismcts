# Information Set Monte Carlo Tree Search

Implementations of search algorithms for **imperfect-information games** -- games where players have hidden state they cannot observe (private cards, hidden dice, unseen moves).

Standard MCTS assumes perfect information. In poker, phantom board games, or any setting with hidden state, we need algorithms that reason over **information sets** (the set of states indistinguishable to a player given their observations).

This project implements and compares four approaches, demonstrates the **strategy fusion pathology**, and provides interactive gameplay against AI opponents.

## Algorithms

### 1. Perfect Information Monte Carlo (PIMC)

The simplest approach: **determinize** the hidden information, solve as if perfect-information, repeat with many samples, and aggregate.

```
For each iteration:
    d ~ Determinize(state, observer)      # sample hidden info
    v = MCTS(d)                           # run standard MCTS
    accumulate action values from v
Return argmax of averaged action values
```

**Weakness:** suffers from **strategy fusion** -- it averages values across worlds the player cannot actually distinguish, potentially recommending actions that are optimal in no single consistent strategy.

### 2. Information Set MCTS (IS-MCTS)

*Cowling, Powley, Whitehouse (2012).* Build a single tree where nodes represent **information sets** rather than states. Different determinizations traverse different parts of the tree but share statistics at matching information sets.

Key modification -- **availability-based UCB:**

$$\text{UCB}(a) = \frac{Q(a)}{N(a)} + c\sqrt{\frac{\ln(\text{Avail}(a))}{N(a)}}$$

where $\text{Avail}(a)$ counts how many times action $a$ was legal when its parent was visited (not all actions are legal in all determinizations).

### 3. Single Observer IS-MCTS (SO-ISMCTS)

Only the **root player's** information sets form tree nodes. Opponent moves are sampled randomly without building tree structure. Saves memory and is appropriate when we only need the root player's decision.

### 4. Multiple Observer IS-MCTS (MO-ISMCTS)

Maintains a **separate tree for each player**. At each decision point, the acting player's tree is consulted. Provides a better opponent model than SO-ISMCTS at the cost of additional memory.

### 5. Smooth UCB

Addresses non-stationarity in IS-MCTS by mixing the tree policy with a uniform baseline:

$$\pi_{\text{smooth}}(a) = \eta \cdot \pi_{\text{tree}}(a) + (1 - \eta) \cdot \text{Uniform}(a)$$

where $\eta = \max\left(0, \frac{N - D}{N}\right)$ increases toward 1 as visit count $N$ grows (dampening parameter $D$ controls the rate).

## Strategy Fusion Pathology

**Strategy fusion** occurs when determinized search combines strategies from different worlds into a "fused" strategy that is achievable in none of them.

**Example (implemented in `src/core/fusion.py`):**

A two-step decision game. A hidden card is dealt (A or B, 50/50). The player (who cannot see the card) makes two decisions:

**Step 1:** COMMIT or SAFE. SAFE ends the game with payoff +0.4.

**Step 2 (after COMMIT):** LEFT or RIGHT.

| Card | COMMIT->LEFT | COMMIT->RIGHT | SAFE |
|------|-------------|---------------|------|
| A    | +1          | -1            | +0.4 |
| B    | -1          | +1            | +0.4 |

- **Optimal strategy:** play SAFE (expected value = 0.4), since after COMMIT the player still cannot see the card, making LEFT/RIGHT a coin flip with expected value 0.0.
- **PIMC's mistake:** In each determinization, PIMC sees the card and computes COMMIT -> optimal follow-up = +1.0, which beats SAFE's +0.4. So PIMC picks COMMIT. But the player cannot condition their step-2 choice on the hidden card! This is strategy fusion: PIMC fuses the strategy (COMMIT->LEFT) from world A with (COMMIT->RIGHT) from world B.
- **IS-MCTS** correctly maintains a single tree over information sets and discovers that SAFE dominates, because it recognizes that the step-2 decision shares a single information set across both worlds.

## Games

### Kuhn Poker
Three-card (J, Q, K), two-player poker. Each player antes 1 chip and receives one card. Players alternate Pass/Bet actions. Small enough to analyze exhaustively -- IS-MCTS should approach Nash equilibrium play.

### Liar's Dice
Two players each roll one die secretly. Players make escalating claims about the total dice and can challenge ("Liar!"). Tests bluffing and probabilistic reasoning under uncertainty.

### Phantom Tic-Tac-Toe
Standard 3x3 Tic-Tac-Toe but players cannot see opponent moves. Attempting to place on an occupied square results in rejection (revealing information). Tests spatial reasoning under partial observability.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Play Kuhn Poker against AI
python -m src.cli kuhn --algorithm so_ismcts --iterations 1000

# Play Phantom TTT against AI
python -m src.cli phantom-ttt --algorithm so_ismcts

# See strategy fusion demo
python -m src.cli fusion

# Benchmark PIMC vs IS-MCTS
python -m src.cli benchmark --games 100

# Run full demo
python examples/demo.py

# Launch interactive Streamlit UI
streamlit run src/viz/app.py
```

## Streamlit UI

The interactive dashboard provides four views:

1. **IS Tree Viewer** -- Run search and visualize the information set tree (visit counts, action values, tree structure).
2. **Play Against AI** -- Play Kuhn Poker or Phantom TTT against SO-ISMCTS, PIMC, MO-ISMCTS, or Smooth UCB.
3. **Determinization Explorer** -- See how hidden information is sampled for each game. Visualize the distribution of sampled worlds.
4. **PIMC vs IS-MCTS Comparison** -- Run batch games and compare win rates. Includes the strategy fusion demonstration.

## Project Structure

```
ismcts/
├── src/
│   ├── core/
│   │   ├── pimc.py            # Determinized MCTS
│   │   ├── ismcts.py          # Information Set MCTS
│   │   ├── so_ismcts.py       # Single Observer IS-MCTS
│   │   ├── mo_ismcts.py       # Multiple Observer IS-MCTS
│   │   ├── smooth_ucb.py      # Smooth UCB selection
│   │   └── fusion.py          # Strategy fusion analysis
│   ├── games/
│   │   ├── game_base.py       # Abstract game interface
│   │   ├── kuhn_poker.py      # Kuhn Poker
│   │   ├── liars_dice.py      # Liar's Dice
│   │   └── phantom_ttt.py     # Phantom Tic-Tac-Toe
│   ├── viz/
│   │   └── app.py             # Streamlit dashboard
│   └── cli.py                 # Command-line interface
├── tests/                     # Comprehensive test suite
├── examples/
│   └── demo.py                # Runnable demonstration
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## References

- Cowling, P.I., Powley, E.J., Whitehouse, D. (2012). "Information Set Monte Carlo Tree Search." *IEEE Transactions on Computational Intelligence and AI in Games*.
- Frank, I., Basin, D. (1998). "Search in Games with Incomplete Information: A Case Study Using Bridge Card Play." *Artificial Intelligence*.
- Heinrich, J., Silver, D. (2015). "Smooth UCT Search in Computer Poker." *IJCAI*.
- Long, J.R., Sturtevant, N.R., Buro, M., Furtak, T. (2010). "Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search." *AAAI*.

## License

MIT
