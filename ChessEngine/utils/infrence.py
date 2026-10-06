import chess
import numpy as np
import torch
from ChessEngine.utils.move_encoding import move_to_index
from ChessEngine.utils.board_encoding import board_to_tensor, batch_board_to_tensor

@torch.no_grad()
def evaluate(board, model, policy_dim):  #returns policy and value output for a board
    
    #get policy output
    boardTensor = board_to_tensor(board).unsqueeze(0).to(next(model.parameters()).device)
    policy_logits, value = model(boardTensor) # shape (1, 4272)

    #create legal move mask
    mask = torch.zeros(policy_dim, device=policy_logits.device)
    for move in board.legal_moves:
        mask[move_to_index[move]] = 1
    

    #filter by legal moves
    policy_logits[:,mask == 0] = -float("inf")
    policy_output = torch.softmax(policy_logits, dim=1)

    return policy_output, value

@torch.no_grad()
def evaluateBatch(boards, model):
    batchSize = len(boards)

    #get policy output
    tensors = [board_to_tensor(board) for board in boards]                          # REPLACE WITH BATCHED BOARD TO TENSOR
    batch = torch.stack(tensors, dim=0).to(next(model.parameters()).device)
    policy_logits, value = model(batch) # shapes policy: (B, 4272) value: (B, 1)
    row_idxs = np.repeat(np.arange(batchSize), [len(list(b.legal_moves)) for b in boards])
    legal_idxs = np.concatenate([[move_to_index[m] for m in b.legal_moves] for b in boards])
    mask = np.zeros((batchSize, policy_logits.shape[1]), dtype=np.float32)
    mask[row_idxs, legal_idxs] = 1
    mask = torch.from_numpy(mask).to(next(model.parameters()).device)


    #filter by legal moves
    policy_logits[mask == 0] = -float("inf")
    policy_output = torch.softmax(policy_logits, dim=1)

    return policy_output, value





#NEXT USE MCTS TO PLAY GAME AND MAKE A MOVE

#board = chess.Board()
#move = chess.Move(1, 18)
#board.push(move)
#tensor = board_to_tensor(board)
#print(tensor[0][12])

