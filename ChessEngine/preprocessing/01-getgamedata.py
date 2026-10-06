import io
import zstandard as zstd
import chess.pgn


from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project/
INPUT_FILE = Path(r"D:\Downloads\lichess_db_standard_rated_2025-08.pgn.zst")
OUTPUT_FILE = BASE_DIR / "Data" / "500k_games.pgn"
MAX_GAMES = 500000

def extract_games():
    with open(INPUT_FILE, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
                count = 0

                while count < MAX_GAMES:
                    game = chess.pgn.read_game(text_stream)

                    if game is None:
                        break

                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))

                    #filter by elo
                    if white_elo < 1800 or black_elo < 1800:
                        continue
                    #ignore short games
                    if game.end().board().fullmove_number < 10:
                        continue
                    #non terminal results
                    if game.headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                        continue
                    #ignore bullet games
                    tc = game.headers.get("TimeControl", "")
                    if "+" in tc:
                        base = int(tc.split("+")[0])
                        if base < 180:
                            continue

                    if count % 1000 == 0:
                        print(f"Saving game {count + 1}/{MAX_GAMES}")
                    
                    out.write(str(game))
                    out.write("\n\n")

                    count += 1
                
                print(f"Finished. Saved {count} games.")


if __name__ == "__main__":
    extract_games()
