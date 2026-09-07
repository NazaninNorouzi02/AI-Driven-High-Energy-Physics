import torch

try:
    from .models import Model
    from .game import TicTacToe
    from .mcts import mcts
    from .tree import Node
except ImportError:
    from models import Model
    from game import TicTacToe
    from mcts import mcts
    from tree import Node


def print_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    for i in range(0, 9, 3):
        print(" ".join(symbols[x] for x in board[i:i+3]))
    print()


def play_vs_ai(model_path="checkpoints/model.pth", sims=50):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = TicTacToe()
    print("\nYou are O (-1). AI is X (+1).")
    print_board(env.board)

    while env.winner() is None:
        if env.current_player == 1:
            print("AI thinking...")
            root = Node(env.clone())
            move = mcts(root, model, n_simulations=sims)
            env.play(move)
            print("AI played:", move)
        else:
            legal = env.moves()
            print("Your move. Legal options:", legal)
            try:
                m = int(input("Enter move (0–8): "))
            except:
                print("Please enter a number.")
                continue

            if m not in legal:
                print("Illegal move!")
                continue

            env.play(m)

        print_board(env.board)

    winner = env.winner()
    if winner == 1:
        print("AI wins!")
    elif winner == -1:
        print("You win!")
    else:
        print("Draw!")


if __name__ == "__main__":
    play_vs_ai()
