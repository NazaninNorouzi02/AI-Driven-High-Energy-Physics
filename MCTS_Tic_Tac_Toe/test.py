import torch
from models import TicTacToeNet

model = TicTacToeNet()

# ورودی: یک برد ۹تایی
board = torch.tensor([
    [1, 0, -1,   0, 1, 0,   0, 0, -1]
], dtype=torch.float32)

policy, value = model(board)

print("policy:", policy)
print("value:", value)
