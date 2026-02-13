import torch
from .models import Model  
from .play import play_game


if __name__ == "__main__":

    modelA = Model()
    modelA.load_state_dict(torch.load('modelA.pth'))
    modelA.eval()  # Set to evaluation mode

    modelB = Model()
    modelB.load_state_dict(torch.load('modelB.pth'))
    modelB.eval()  # Set to evaluation mode

    result = play_game(modelA, modelB)

    print("Final Scores:", result)


