import torch
import chess
import chess.pgn
from ChessEngine.models.ChessNet import ChessNet
from pathlib import Path
from ChessEngine.utils.mcts import MCTS, MCTSNode

FROM_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "Checkpoints" / "pretraining_checkpoint-9-0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet(18, 128, 10, 8, 4272).to(device)
checkpoint = torch.load(FROM_CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
board = chess.Board()



def comp_vs_comp(model, board):
    with torch.no_grad():
        while True:
            root = MCTSNode(board)
            mcts = MCTS(model, root, simulations=1000)
            moves, visits = mcts.get_move()
            idx = torch.argmax(visits).item()
            move = moves[idx]
            san_move = board.san(move)
            board.push(move)
            print(board)
            print(san_move, end="\n\n")

            if (board.is_game_over()):
                break


    game = chess.pgn.Game.from_board(board)
    print(game)


def comp_vs_player(model, board):
    with torch.no_grad():
        while True:
            root = MCTSNode(board)
            mcts = MCTS(model, root, simulations=500)
            moves, visits = mcts.get_move()
            idx = torch.argmax(visits).item()
            move = moves[idx]
            san_move = board.san(move)
            board.push(move)
            print(board)
            print(san_move, end="\n\n")

            if (board.is_game_over()):
                break

            player_san = input("enter move: ")
            player_move = board.parse_san(player_san)
            board.push(player_move)
            print(board, end="\n\n")

            if (board.is_game_over()):
                break


    game = chess.pgn.Game.from_board(board)
    print(game)

comp_vs_comp(model, board)