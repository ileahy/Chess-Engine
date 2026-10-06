import chess.pgn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project/
INPUT_FILE = BASE_DIR / "Data" / "500k_games.pgn"
OUTPUT_DIR = BASE_DIR / "Data"

games_per_file = 50000

current_file = 0
games_in_current = 0

# Open the first output file
out = open(OUTPUT_DIR / f"games_{current_file:03d}.pgn", "w", encoding="utf-8")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    while True:
        if games_in_current % 1000 == 0:
            print(f"{games_in_current} games added to file number: {current_file}")
        game = chess.pgn.read_game(f)

        if game is None:
            break

        if games_in_current == games_per_file:
            out.close()
            print(f"finished file {current_file} \n\n")

            current_file += 1
            games_in_current = 0

            out = open(
                OUTPUT_DIR / f"games_{current_file:03d}.pgn",
                "w",
                encoding="utf-8",
            )

        print(game, file=out, end="\n\n")
        games_in_current += 1

out.close()


