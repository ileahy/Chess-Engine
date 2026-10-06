#mcts for generating moves
from ChessEngine.utils.infrence import evaluate, evaluateBatch
import torch
import math
from ChessEngine.utils.move_encoding import move_to_index
import numpy as np

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0, index_in_parent=None):
        self.board = board
        self.parent = parent
        self.index_in_parent = index_in_parent
        self.prior = prior

        #created by expand, np arrays
        self.moves = None
        self.child_nodes = None
        self.child_priors = None 
        self.child_visit_counts = None # np array, backprop()
        self.child_value_sums = None  # np array, backprop()
        self.child_move_idxs = None

        #maybe remove later
        self._visit_count = 0
        self._value_sum = 0.0

    @property
    def visit_count(self):
        return self._visit_count if self.parent is None else self.parent.child_visit_counts[self.index_in_parent]
    @property
    def value_sum(self):
        return self._value_sum if self.parent is None else self.parent.child_value_sums[self.index_in_parent]

    def value(self):
        vc = self.visit_count
        vs = self.value_sum
        return 0.0 if vc == 0 else vs / vc
    
    def make_root(self):
        self.parent = None
        self._visit_count = 0
        self._value_sum = 0.0


class MCTS:
    def __init__(self, model, root, simulations=100, c_puct=1.6):
        self.model = model
        self.root = root
        self.simulations = simulations
        self.c_puct = c_puct

    def select_child(self, node):
        exploration = self.c_puct * math.sqrt(node.visit_count) # single value

        vc = node.child_visit_counts

        child_value = np.divide(
            node.child_value_sums, vc,
            out=np.zeros_like(vc, dtype=float),
            where=vc != 0,
        )
        puct = -child_value + exploration * node.child_priors / (vc + 1)
        return node.child_nodes[np.argmax(puct)]
        

    def expand(self, node):
        moves = list(node.board.legal_moves)
        idxs = [move_to_index[m] for m in moves]
        policy, value = evaluate(node.board, self.model, 4272)
        policy = policy.squeeze(0)
        priors = policy[idxs].cpu().numpy()
        node.moves = moves
        node.child_priors = priors
        node.child_visit_counts = np.zeros(len(moves))
        node.child_value_sums = np.zeros(len(moves))
        node.child_move_idxs = np.array(idxs)
        child_nodes = []
        for i, m in enumerate(moves):
            child_board = node.board.copy(stack=False)
            child_board.push(m)
            child_nodes.append(MCTSNode(child_board, parent=node, prior=priors[i], index_in_parent=i))
        node.child_nodes = child_nodes

        return value.item()  

    def backpropagate(self, node, value):
        while node.parent is not None:
            node.parent.child_visit_counts[node.index_in_parent] += 1 #node visit count
            node.parent.child_value_sums[node.index_in_parent] += value #node value sum
            value = -value    #alternate value because switches from white to black
            node = node.parent
        node._visit_count += 1
        node._value_sum += value

    def run(self):
        for _ in range(self.simulations):
            node = self.root
            while node.child_nodes:
                node = self.select_child(node)
            if node.board.is_game_over():
                result = node.board.result()
                if result == "1-0":
                    value = 1.0 if node.board.turn else -1.0
                elif result == "0-1":
                    value = -1.0 if node.board.turn else 1.0
                else:
                    value = 0.0
            else:
                value = self.expand(node)
            self.backpropagate(node, value)

    def get_move(self):
        self.run()
        moves = self.root.moves
        visits = self.root.child_visit_counts
        visits = torch.tensor(visits, dtype=torch.float32) #raw counts
        visits = visits / visits.sum()
        
        return moves, visits



        




class BatchMCTS:
    def __init__(self, model, simulations, c_puct, root_list):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.root_list = root_list #list of MCTS nodes

    def select_child(self, node): #use puct
        exploration = self.c_puct * math.sqrt(node.visit_count) # single value

        vc = node.child_visit_counts

        child_value = np.divide(
            node.child_value_sums, vc,
            out=np.zeros_like(vc, dtype=float),
            where=vc != 0,
        )
        puct = -child_value + exploration * node.child_priors / (vc + 1)
        return node.child_nodes[np.argmax(puct)]

    #keep stack? for threefold?    
    def expand(self, nodes):
        boards = [node.board for node in nodes]
        policies, values = evaluateBatch(boards, self.model)

        for idx, node in enumerate(nodes):
            moves = list(node.board.legal_moves)
            idxs = [move_to_index[m] for m in moves]
            priors = policies[idx].squeeze(0)[idxs].cpu().numpy()
            node.moves = moves
            node.child_priors = priors
            node.child_visit_counts = np.zeros(len(moves))
            node.child_value_sums = np.zeros(len(moves))
            node.child_move_idxs = np.array(idxs)
            child_nodes = []
            for i, m in enumerate(moves):
                child_board = node.board.copy(stack=False)
                child_board.push(m)
                child_nodes.append(MCTSNode(child_board, parent=node, prior=priors[i], index_in_parent=i))
            node.child_nodes = child_nodes
        
        return values.squeeze(1).tolist()


    def backpropagate(self, node, value):
        while node.parent is not None:
            node.parent.child_visit_counts[node.index_in_parent] += 1 #node visit count
            node.parent.child_value_sums[node.index_in_parent] += value #node value sum
            value = -value    #alternate value because switches from white to black
            node = node.parent
        node._visit_count += 1
        node._value_sum += value

    def run(self):
        for _ in range(self.simulations):
            expand_nodes = []
            for root in self.root_list:
                node = root
                while node.child_nodes:
                    node = self.select_child(node)
                if node.board.is_game_over():
                    result = node.board.result()
                    if result == "1-0":
                        value = 1.0 if node.board.turn else -1.0
                    elif result == "0-1":
                        value = -1.0 if node.board.turn else 1.0
                    else:
                        value = 0.0
                    self.backpropagate(node, value)
                else:
                    expand_nodes.append(node)
            #expand/backprop all nodes at once
            if expand_nodes:
                values = self.expand(expand_nodes)
                for value, node in zip(values, expand_nodes):
                    self.backpropagate(node, value)

    def get_move(self):
        self.run()
        mcts_visits = torch.zeros(len(self.root_list), 4272)
        for batch, root in enumerate(self.root_list):
            mcts_visits[batch, root.child_move_idxs] = torch.from_numpy(root.child_visit_counts).float()
        return mcts_visits