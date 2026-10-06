#train using selfplay, parallelized across CPU processes
import torch
import torch.multiprocessing as mp
from torch import optim
from pathlib import Path
from ChessEngine.models.ChessNet import ChessNet
from ChessEngine.training.selftrain import SelfPlay
from ChessEngine.utils.replay_buffer import ReplayBuffer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = BASE_DIR / "Checkpoints" / "pretraining_checkpoint-9"

MODEL_ARGS = (18, 128, 10, 8, 4272)


def _extract_samples(buffer: ReplayBuffer):
    n = buffer.capacity if buffer.full else buffer.head
    return (
        buffer.states[:n].clone(),
        buffer.move_idxs[:n].clone(),
        buffer.move_probs[:n].clone(),
        buffer.n_moves[:n].clone(),
        buffer.values[:n].clone(),
    )


def _worker_loop(worker_id, device_str, task_queue, result_queue):
    device = torch.device(device_str)
    model = ChessNet(*MODEL_ARGS).to(device)
    model.eval()

    while True:
        cmd, payload = task_queue.get()
        if cmd == "stop":
            break
        elif cmd == "weights":
            model.load_state_dict(payload)
        elif cmd == "play":
            print(f"worker {worker_id} started")
            n_games, simulations, c_puct, temperature = payload
            worker_buffer_capacity = n_games * 400  #generous headroom
            worker_sp = SelfPlay(
                model, optimizer=None,
                selfplay_batchsize=n_games,
                simulations=simulations, c_puct=c_puct, temperature=temperature,
                buffer_capacity=worker_buffer_capacity,
            )
            with torch.no_grad():
                worker_sp.generate_positions()
            print(f"worker {worker_id} finished")
            result_queue.put(_extract_samples(worker_sp.ReplayBuffer))


class ParallelSelfPlay(SelfPlay):
    def __init__(self, *args, n_workers=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_workers = min(n_workers, self.selfplay_batchsize)

        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._task_queues = [ctx.Queue() for _ in range(self.n_workers)]
        self._procs = [
            ctx.Process(target=_worker_loop, args=(i, str(self.device), self._task_queues[i], self._result_queue))
            for i in range(self.n_workers)
        ]
        for p in self._procs:
            p.start()
        self._push_weights()

    def _push_weights(self):
        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        for q in self._task_queues:
            q.put(("weights", state_dict))

    def generate_positions(self):
        games_per_worker = self.selfplay_batchsize // self.n_workers
        remainder = self.selfplay_batchsize % self.n_workers

        n_dispatched = 0
        for i, q in enumerate(self._task_queues):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games == 0:
                continue
            q.put(("play", (n_games, self.simulations, self.c_puct, self.temperature)))
            n_dispatched += 1

        for _ in range(n_dispatched):
            states, move_idxs, move_probs, n_moves, values = self._result_queue.get()
            self.ReplayBuffer.add_batch_sparse(states, move_idxs, move_probs, n_moves, values)

    def close(self):
        for q in self._task_queues:
            q.put(("stop", None))
        for p in self._procs:
            p.join()

    def run(self):
        print("starting", end="\n")
        for _ in range(5): #TEMPORARY REMOVE LATER #TEMPORARY REMOVE LATER #TEMPORARY REMOVE LATER #TEMPORARY REMOVE LATER#TEMPORARY REMOVE LATER #TEMPORARY REMOVE LATER
            self.generate_positions()
        for iteration in range(self.iterations):
            self.play()
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, BASE_DIR / "Checkpoints" / f"selftraining_checkpoint_iteration-{iteration}")
            if iteration < self.iterations - 1:
                self._push_weights()  #refresh workers with newly trained weights for next iteration
        self.close()

#replay buffer capacity we want to fill percent thats trained on.
#training_batches * training_batchsize = samples ==> 256,000 = 50%
#fill training_batches * batchzise in replay buffer.
# generate positions => selfplay_batchsize boards = 768 games per batch ~= 50k positions
#generate positions 5 times to get to "full buffer" from training standpoint
#num_generate_positions = (training_batches * training_batchsize) // (selfplay_batchsize * 60)


import time
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(*MODEL_ARGS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.train()

    start_time = time.perf_counter()
    self_play = ParallelSelfPlay(model, optimizer, n_workers=12)
    self_play.run()
    end_time = time.perf_counter()
    print(end_time - start_time)
