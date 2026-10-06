# Chess-Engine

An AlphaZero-style chess engine written in Python with PyTorch. A residual policy/value network is first **pretrained** with supervised learning on Lichess games, then improved through **self-play** reinforcement learning guided by Monte Carlo Tree Search (MCTS).

## How it works

1. **Data**: about 500k rated Lichess games are filtered (both players 1800+ Elo, no bullet, at least 10 moves, decisive or drawn result) and converted into `(board, move, outcome)` training samples.
2. **Pretraining**: the network learns to predict the move that was played (policy) and the final game result (value).
3. **Self-play**: the pretrained network plays games against itself using batched MCTS. Each position's MCTS visit distribution and the game outcome are stored in a replay buffer, and the network is trained on samples drawn from that buffer.
4. **Play**: MCTS with the trained network picks moves, either against itself or against a human.

### Network: `ChessNet`

| Component | Details |
|---|---|
| Input | `18 × 8 × 8` tensor |
| Body | 3×3 conv stem → 10 residual blocks, 128 channels |
| Policy head | 1×1 conv → linear layer → 4272 move logits |
| Value head | 1×1 conv → 256-unit FC → `tanh` scalar in `[-1, 1]` |
| Size | about 3.5M parameters |
| Loss | Policy cross-entropy + value MSE |

### Board encoding (18 planes)

| Planes | Meaning |
|---|---|
| 0–5 | White pawn, knight, bishop, rook, queen, king |
| 6–11 | Black pieces, same order |
| 12 | Side to move (`+1` white, `-1` black) |
| 13–16 | Castling rights (W-K, W-Q, B-K, B-Q) |
| 17 | En passant square |

### Move encoding (4272 outputs)

- **4096**: every `from-square × to-square` pair (64 × 64). Queen promotions use these indices.
- **176**: every promotion move to each piece, including underpromotions.

Illegal moves are masked out before the softmax during inference.

## Project structure

```
ChessEngineProject/
├── ChessEngine/
│   ├── models/
│   │   ├── ChessNet.py            # Residual policy/value network
│   │   ├── loss.py                # Policy + value loss
│   │   └── stockfish/             # Stockfish source/binary (not yet used in the pipeline)
│   ├── preprocessing/
│   │   ├── 01-getgamedata.py      # Stream and filter games from a Lichess .pgn.zst dump
│   │   ├── 02-splitgames.py       # Split into 10 PGN shards of 50k games
│   │   └── 03-pgnListToDataset.py # Encode shards to tensor datasets (shard_XXX.pt)
│   ├── training/
│   │   ├── pretrain.py            # Supervised pretraining on the shards
│   │   ├── selftrain.py           # SelfPlay: batched self-play and replay-buffer training
│   │   ├── selftrain_parallel.py  # Multi-process self-play (main RL entry point)
│   │   └── personaltest.py        # Engine vs. engine / engine vs. human
│   └── utils/
│       ├── board_encoding.py      # python-chess Board → tensor
│       ├── move_encoding.py       # Move ↔ policy index mappings
│       ├── infrence.py            # Single and batched evaluation with legal-move masking
│       ├── mcts.py                # MCTS and BatchMCTS (PUCT)
│       ├── replay_buffer.py       # Fixed-size, sparse-policy replay buffer
│       └── vectorized_mcts_workspace.py  # Work-in-progress vectorized MCTS
├── Data/          # PGNs and tensor shards (git-ignored)
└── Checkpoints/   # Model/optimizer checkpoints (git-ignored)
```

## Setup

Requires Python 3.11+. A CUDA GPU is strongly recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Pick the PyTorch build that matches your CUDA version: https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install chess numpy zstandard
```

## Usage

Run every script as a module from the project root so the `ChessEngine.*` imports resolve.

### 1. Prepare data

Download a monthly database from [database.lichess.org](https://database.lichess.org/) (`.pgn.zst`), then set `INPUT_FILE` in `01-getgamedata.py` to its path.

```bash
python -m ChessEngine.preprocessing.01-getgamedata      # → Data/500k_games.pgn
python -m ChessEngine.preprocessing.02-splitgames       # → Data/games_000.pgn … games_009.pgn
python -m ChessEngine.preprocessing.03-pgnListToDataset # → Data/shard_000.pt … shard_009.pt
```

### 2. Pretrain

```bash
python -m ChessEngine.training.pretrain
```

Each epoch trains on a fixed set of three shards, and the learning rate follows a cosine schedule. A checkpoint is written to `Checkpoints/pretraining_checkpoint-<epoch>` after every epoch. To start from scratch, set `FROM_CHECKPOINT = None`; to resume, set it to a checkpoint name.

### 3. Self-play training

```bash
python -m ChessEngine.training.selftrain_parallel
```

This loads the pretrained checkpoint (`CHECKPOINT`), runs self-play across 12 worker processes, and trains on the shared replay buffer. A checkpoint is saved after each iteration as `Checkpoints/selftraining_checkpoint_iteration-<n>`. You can tune the defaults in the `SelfPlay` constructor:

| Parameter | Default |
|---|---|
| Games per self-play batch | 768 |
| MCTS simulations per move | 100 |
| `c_puct` | 1.6 |
| Replay buffer capacity | 500,000 |
| Training batches / batch size | 1000 / 256 |
| Iterations | 16 |
| Max plies per game | 400 |

### 4. Play against it

```bash
python -m ChessEngine.training.personaltest
```

By default the script loads `FROM_CHECKPOINT` and plays the engine against itself, printing the board and the final PGN. To play as Black yourself, call `comp_vs_player(model, board)` instead and enter moves in SAN (e.g. `Nf6`).

## Checkpoint format

Checkpoints are saved with `torch.save` as a dict:

```python
{"model": model.state_dict(), "optimizer": optimizer.state_dict()}
```

Load a checkpoint into `ChessNet(18, 128, 10, 8, 4272)`.

## Acknowledgements

- Inspired by DeepMind's [AlphaZero](https://arxiv.org/abs/1712.01815).
- Training games from the [Lichess open database](https://database.lichess.org/).
- [python-chess](https://github.com/niklasf/python-chess) for move generation and PGN parsing.
- [Stockfish](https://github.com/official-stockfish/Stockfish) (GPLv3) is included under `ChessEngine/models/stockfish/`.
