#board to tensor conversions, and model evaluations for self play
import chess
import torch


PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

#13 channels (12 peice types white/black, move)

"""
def board_to_tensor(board):
    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    for piece_type in PIECE_MAP:
        for square in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(square, 8) # get quotient, remainder
            tensor[PIECE_MAP[piece_type]][row][col] = 1
        for square in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(square, 8)
            tensor[PIECE_MAP[piece_type] + 6][row][col] = 1

    tensor[12] = 1 if board.turn == chess.WHITE else -1
    return torch.tensor(tensor).unsqueeze(0) #add batch dim
"""

#shape (18, 8, 8)
def board_to_tensor(board):
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    #peice positions
    for square, piece in board.piece_map().items():
        plane = PIECE_MAP[piece.piece_type] + (0 if piece.color else 6)
        row, col = divmod(square, 8)
        tensor[plane, 7 - row, col] = 1

    #turn
    tensor[12].fill_(1.0 if board.turn else -1.0)

    #castling rights
    white_kingside = board.has_kingside_castling_rights(chess.WHITE)
    white_queenside = board.has_queenside_castling_rights(chess.WHITE)
    black_kingside = board.has_kingside_castling_rights(chess.BLACK)
    black_queenside = board.has_queenside_castling_rights(chess.BLACK)
    tensor[13].fill_(white_kingside)
    tensor[14].fill_(white_queenside)
    tensor[15].fill_(black_kingside)
    tensor[16].fill_(black_queenside)

    #en passant
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[17][7 - row, col] = 1

    return tensor

def batch_board_to_tensor(boards):
    return