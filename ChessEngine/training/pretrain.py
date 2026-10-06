import torch
from torch.utils.data import Dataset
from pathlib import Path
#from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ChessEngine.models.ChessNet import ChessNet
from ChessEngine.models.loss import chess_loss
import torch.optim as optim
import gc

FROM_CHECKPOINT = "pretraining_checkpoint-9-0"

#didnt want to use random so i just hardcoded
epochs = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [0,3,6],
    [1,4,7],
    [2,5,9], 
    [0,4,9],
    [1,3,5],
    [2,6,8],
    [7,8,9]
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ChessDataset(Dataset):
    def __init__(self, shards):
        
        states = []
        policies = []
        values = []

        for shard in shards:
            SHARD_FILE = BASE_DIR / "Data" / shard
            data = torch.load(SHARD_FILE)
            states.append(data["states"])
            policies.append(data["policies"])
            values.append(data["values"])
            del data

        self.states = torch.cat(states, dim=0)
        self.policies = torch.cat(policies, dim=0)
        self.values = torch.cat(values, dim=0)

        del states
        del policies
        del values
        gc.collect()


    def __len__(self):
        return len(self.states)


    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.policies[idx],
            self.values[idx]
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(18, 128, 10, 8, 4272).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(epochs), eta_min=1e-6)
    if FROM_CHECKPOINT:
        checkpoint = torch.load(
            BASE_DIR / "Checkpoints" / FROM_CHECKPOINT,
            map_location=device
        )  
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    model.train()

    #loop throguh predefined epochs
    for epoch, shards in enumerate(epochs):
        dataset = ChessDataset([f"shard_{shard:03d}.pt" for shard in shards])

        #create dataloader - maybe change num_workers depending on device, mine has 16 core cpu  REMOVED TRAIN AND TEST SPLIT TO GIVE MORE DATA
        train_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        #train
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            states = states.to(device, dtype=torch.float32, non_blocking=True)
            policies = policies.to(device, dtype=torch.long, non_blocking=True)
            values = values.to(device, dtype=torch.float32, non_blocking=True)

            policy_logits, value_pred = model(states)
            policy_loss, value_loss = chess_loss(policy_logits, value_pred, policies, values)
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if batch_idx % 1000 == 0:
                print(f"epoch: {epoch}, batch: {batch_idx}, loss: {loss}, policy loss: {policy_loss}, value loss: {value_loss}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, BASE_DIR / "Checkpoints" / f"pretraining_checkpoint-{epoch}")

        scheduler.step()

        del train_loader
        del dataset
        gc.collect()
        



if __name__ == "__main__":
    main()
