import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Second conv layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x  = x + residual  # skip connection
        out = F.relu(x)
        return out


class ResidualTower(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential( *[ResidualBlock(channels) for _ in range(num_blocks)] )

    def forward(self, x):
        return self.blocks(x)
    

class PolicyHead(nn.Module):
    def __init__(self, channels, board_size=8, policy_dim=4272):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

        flat_dim = 2 * board_size * board_size
        self.fc = nn.Linear(flat_dim, policy_dim)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.flatten(1)  # (batch, 2*H*W)
        x = self.fc(x)    # logits, softmax later
        return x


class ValueHead(nn.Module):
    def __init__(self, channels, board_size=8, hidden_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

        flat_dim = board_size * board_size
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    

# in_channels=18, hidden_channels=128, num_blocks=10, board_size=8, policy_dim=4272
class ChessNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_blocks, board_size=8, policy_dim=4272):
        super().__init__()
        self.expansion_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.tower = ResidualTower(hidden_channels, num_blocks)
        self.policy_head = PolicyHead(hidden_channels, board_size, policy_dim)
        self.value_head = ValueHead(hidden_channels, board_size)

    def forward(self, x):
        x = self.expansion_layer(x)
        x = self.tower(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# around 3.5 million params with 18 input and 128 hidden channels with 10 blocks
