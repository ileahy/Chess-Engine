#train model with existing grandmaster games
import torch
import chess.pgn
from pathlib import Path
from ChessEngine.utils.move_encoding import move_to_index
from ChessEngine.utils.board_encoding import board_to_tensor

SHARD_NUMBERS = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009"]

#create training dataset
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project/
INPUT_FILES = [BASE_DIR / "Data" / f"games_{SHARD_NUMBER}.pgn" for SHARD_NUMBER in SHARD_NUMBERS] #firsth game batch

result_to_value = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0
}


for idx, INPUT_FILE in enumerate(INPUT_FILES):
    states = [] #array of board tensors
    policies = [] #array of ints
    values = [] #array of floats
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers["Result"]
            value = result_to_value[result]

            board = game.board()   #just in case theres a weird starting position rather than use "chess.Board()"

            for move in game.mainline_moves():
                state = board_to_tensor(board)
                move_idx = move_to_index[move]
                side = 1.0 if board.turn == chess.WHITE else -1.0
                value_target = side * value

                states.append(state.to(torch.int8))
                policies.append(move_idx)
                values.append(value_target)

                board.push(move)

            count += 1
            if count % 1000 == 0:
                print(f"shard: {SHARD_NUMBERS[idx]}, game: {count} finished")

    torch.save(
        {
            "states": torch.stack(states),
            "policies": torch.tensor(policies, dtype=torch.int16),
            "values": torch.tensor(values, dtype=torch.float32),
        },
        BASE_DIR / "Data" / f"shard_{SHARD_NUMBERS[idx]}.pt",
    )

        