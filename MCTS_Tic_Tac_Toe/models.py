import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Simple neural network that predicts both policy (which moves to play)
    and value (who is likely to win) for TicTacToe.
    
    Input: tensor of shape (batch, 9) representing board state
    Output:
        - policy_logits: tensor (batch, 9) → scores for each move
        - value: tensor (batch,1) → estimated outcome in [-1,1]
    """
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(9, 64)   # first hidden layer
        self.fc2 = nn.Linear(64, 64)  # second hidden layer

        self.policy_head = nn.Linear(64, 9)  # outputs logits for each cell
        self.value_head = nn.Linear(64, 1)   # outputs single value

    def forward(self, x):
        h = F.relu(self.fc1(x))  # apply ReLU activation
        h = F.relu(self.fc2(h))

        policy_logits = self.policy_head(h)     # raw move scores
        value = torch.tanh(self.value_head(h))  # map to [-1,1] for win/loss

        return policy_logits, value


def build_model():
    """Factory used by autograder."""
    return Model()


