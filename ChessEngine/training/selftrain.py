#train using selfplay
import chess.pgn
import numpy as np
import torch
from torch import optim
from pathlib import Path
from ChessEngine.models.ChessNet import ChessNet
from ChessEngine.models.loss import chess_loss
import chess
from ChessEngine.utils.board_encoding import board_to_tensor
from ChessEngine.utils.mcts import MCTSNode, BatchMCTS
from ChessEngine.utils.replay_buffer import ReplayBuffer
from ChessEngine.utils.move_encoding import index_to_move

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = BASE_DIR / "Checkpoints"/ "pretraining_checkpoint-9"


class SelfPlay:
    def __init__(self, model, optimizer, selfplay_batches=1, selfplay_batchsize=768, training_batches=1000,
                 training_batchsize=256, simulations=100, c_puct=1.6, temperature=1.0, buffer_capacity=500000,
                 iterations=16, max_plies=400):
        self.model = model
        self.optimizer = optimizer
        self.selfplay_batches = selfplay_batches
        self.selfplay_batchsize = selfplay_batchsize
        self.training_batches = training_batches
        self.training_batchsize = training_batchsize
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.ReplayBuffer = ReplayBuffer(capacity=buffer_capacity)
        self.device = next(model.parameters()).device
        self.iterations = iterations
        self.max_plies = max_plies

    @staticmethod
    def apply_temperature(visits, temperature):
        if temperature == 0:
            policy = torch.zeros_like(visits)
            best_moves = torch.argmax(visits, dim=1, keepdim=True)
            policy.scatter_(1, best_moves, 1.0)
            return policy

        visits = visits ** (1 / temperature)
        return visits / visits.sum(dim=1, keepdim=True)

    def generate_positions(self):
        fullReplayBuffer = [[] for _ in range(self.selfplay_batchsize)]  # use replay buffer class later
        boards = [chess.Board() for _ in range(self.selfplay_batchsize)]
        root_list = [MCTSNode(boards[i]) for i in range(self.selfplay_batchsize)]
        active_mask = torch.ones(self.selfplay_batchsize, dtype=torch.bool)
        ply_counts = [0] * self.selfplay_batchsize
        with torch.no_grad():
            while True:
                active_idx = torch.nonzero(active_mask, as_tuple=True)[0].tolist() #index of active games

                #call model and generate mcts policy with temperature
                active_roots = [root_list[i] for i in active_idx]
                mcts = BatchMCTS(self.model, root_list=active_roots, simulations=self.simulations, c_puct = self.c_puct)
                mcts_visits = mcts.get_move()
                training_policy = mcts_visits / mcts_visits.sum(dim=1, keepdim=True) #train on data without temperature
                mcts_policy = SelfPlay.apply_temperature(mcts_visits, self.temperature)  #change so dynamic throught the game

                #replay buffer additions
                tensors = torch.stack([board_to_tensor(boards[i]) for i in active_idx], dim=0)
                for board_idx, state, policy in zip(active_idx, tensors, training_policy):
                    side = 1.0 if boards[board_idx].turn == chess.WHITE else -1.0
                    fullReplayBuffer[board_idx].append((state.to(torch.int8), policy, side))  #state = (18, 8, 8)

                #get moves and push to boards
                moves = torch.multinomial(mcts_policy, 1).squeeze(1)
                for move_index, board_index in enumerate(active_idx):
                    move = index_to_move[moves[move_index].item()]
                    boards[board_index].push(move)
                    ply_counts[board_index] += 1

                #update active mask
                for i in active_idx:
                    if boards[i].is_game_over(claim_draw=True) or ply_counts[i] >= self.max_plies:
                        active_mask[i] = False
                #break if no active games
                if not active_mask.any():
                    break

                #update root list:
                for idx, move_index in enumerate(moves):
                    board_index = active_idx[idx]
                    root = root_list[board_index]
                    pos = np.nonzero(root.child_move_idxs == move_index)[0].item()
                    new_root = root.child_nodes[pos]
                    new_root.make_root()
                    root_list[board_index] = new_root

                                

        #determine results
        results = torch.zeros(self.selfplay_batchsize, dtype=torch.float32)
        for idx, board in enumerate(boards):
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True).winner
                results[idx] = 0.0 if outcome is None else 1.0 if outcome else -1.0
            else:
                results[idx] = 0.0  #hit max_plies without a real result, treat as a draw

        for result, game in zip(results, fullReplayBuffer):
            for idx, (state, policy, side) in enumerate(game):
                self.ReplayBuffer.add(state, policy, side * result)

    def play(self):
        #create positions
        for batch in range(self.selfplay_batches):
            print(f"starting self play batch number: {batch}")
            self.generate_positions()

        #train model
        for batch in range(self.training_batches):
            states, policies, values = self.ReplayBuffer.sample(self.training_batchsize)
            states = states.to(self.device, dtype=torch.float32)
            policies = policies.to(self.device)
            values = values.to(self.device)

            policy_logits, value_pred = self.model(states)
            policy_loss, value_loss = chess_loss(policy_logits, value_pred, policies, values)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if batch % (self.training_batches // 2) == 0:
                print(f"batch: {batch}, loss: {loss.item()}")
        

    def run(self):
        for iteration in range(self.iterations):
            self.play()
            #create checkpoint
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, BASE_DIR / "Checkpoints" / f"selftraining_checkpoint_iteration-{iteration}")
#print loss at some point.

        
#8 batches,  batchsize 32
#replay buffer is 500_000, fill 10%, 50k psoitions => 625 games
#with 32 move per self play batch => 19 batches

#50k new positions * 3-5 ~= 225k. training batch size = 256
#training batches ~~ 800 => use 750
