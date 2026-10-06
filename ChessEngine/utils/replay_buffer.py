import torch

class ReplayBuffer:
    def __init__(self, board_channels=18, policy_size=4272, capacity=500000, max_moves=128):
        self.head = 0
        self.capacity = capacity
        self.policy_size = policy_size
        self.max_moves = max_moves
        self.full = False

        self.states = torch.zeros((capacity, board_channels, 8, 8), dtype=torch.int8)
        self.move_idxs = torch.zeros((capacity, max_moves), dtype=torch.int16)
        self.move_probs = torch.zeros((capacity, max_moves), dtype=torch.float32)
        self.n_moves = torch.zeros((capacity,), dtype=torch.int16)
        self.values = torch.zeros((capacity,))

    def _sparsify(self, policy):
        nz = torch.nonzero(policy, as_tuple=True)[0]
        n = nz.numel()
        if n > self.max_moves:
            top = torch.topk(policy[nz], self.max_moves)
            nz = nz[top.indices]
            n = self.max_moves
        idxs = torch.zeros(self.max_moves, dtype=torch.int16)
        probs = torch.zeros(self.max_moves, dtype=torch.float32)
        idxs[:n] = nz.to(torch.int16)
        probs[:n] = policy[nz]
        return idxs, probs, n

    def add(self, state_tensor, policy, value):
        idxs, probs, n = self._sparsify(policy)
        self.states[self.head] = state_tensor.to(torch.int8)
        self.move_idxs[self.head] = idxs
        self.move_probs[self.head] = probs
        self.n_moves[self.head] = n
        self.values[self.head] = value
        self.head = (self.head + 1) % self.capacity
        if self.head == 0:
            self.full = True

    #for merging worker process buffers. already sparse data
    def add_batch_sparse(self, states, move_idxs, move_probs, n_moves, values):
        for i in range(states.shape[0]):
            self.states[self.head] = states[i].to(torch.int8)
            self.move_idxs[self.head] = move_idxs[i]
            self.move_probs[self.head] = move_probs[i]
            self.n_moves[self.head] = n_moves[i]
            self.values[self.head] = values[i]
            self.head = (self.head + 1) % self.capacity
            if self.head == 0:
                self.full = True

    def sample(self, batch_size):
        range = self.capacity if self.full else self.head
        indices = torch.randint(range, (batch_size,))

        states = self.states[indices].to(torch.float32)
        values = self.values[indices]
        move_idxs = self.move_idxs[indices].long()
        move_probs = self.move_probs[indices]

        policies = torch.zeros((batch_size, self.policy_size))
        policies.scatter_add_(1, move_idxs, move_probs)

        return states, policies, values

    def __len__(self):
        return self.capacity if self.full else self.head
