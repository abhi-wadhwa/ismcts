# ismcts

information set monte carlo tree search — MCTS for games where you can't see your opponent's cards.

## what this is

in perfect-information games (chess, go), MCTS builds a single tree. in imperfect-information games (poker, bridge, hanabi), you have to deal with hidden information. ISMCTS handles this:

- **determinization** — sample a possible world consistent with your information, run MCTS on that. repeat with many samples
- **SO-ISMCTS** — single-observer variant. one tree, multiple determinizations. each simulation samples a different possible world
- **MO-ISMCTS** — multiple-observer variant. separate trees for each player
- **strategy fusion** — combine strategies from different determinizations. the tricky part where naive approaches fail

## running it

```bash
pip install -r requirements.txt
python main.py
```

## the subtlety

determinization sounds simple but has a known failure mode: **strategy fusion**. if the optimal strategy differs across possible worlds, averaging them can produce a strategy that's terrible in all worlds. SO-ISMCTS mitigates this by sharing a single tree across determinizations, naturally blending information from different possible states.
