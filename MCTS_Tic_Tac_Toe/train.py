import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

# safe imports to work as script or package
try:
    from .models import Model, build_model
    from .mcts import mcts
    from .game import TicTacToe
    from .tree import Node
except Exception:
    from models import Model, build_model
    from mcts import mcts
    from game import TicTacToe
    from tree import Node

def play_self_game(model, sims=50):
    """
    Play one self-play game using MCTS guided by `model`.
    Return: list of (board_list, pi_vector, player) and final winner (1/-1/0).
    pi_vector is a length-9 vector of visit-count probabilities from root at each turn.
    """
    records = []
    env = TicTacToe()

    while env.winner() is None:
        root = Node(env.clone())     # clone board for MCTS
        move = mcts(root, model, n_simulations=sims)   

        # build visit-probability array pi
        pi = [0.0] * 9
        total_visits = 0
        for m, child in root.children.items():
            total_visits += child.visits
        if total_visits == 0:
            # fallback: put uniform mass on legal moves
            legal = env.moves()
            for m in legal:
                pi[m] = 1.0 / len(legal)
        else:
            for m, child in root.children.items():
                pi[m] = child.visits / total_visits

        # record current board, pi, and current player
        records.append((env.board[:], pi, env.current_player))

        # play the chosen move
        env.play(move)

    winner = env.winner()  # 1, -1, or 0
    return records, winner

def train(num_games=200, sims=50, epochs=3, batch_size=32, lr=1e-3, device="cpu"):
    device = torch.device(device)
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    dataset = []  # list of (board, pi, z)

    print("Self-play to collect data...")
    for g in range(num_games):
        recs, winner = play_self_game(model, sims=sims)
        # convert each record into training targets
        for board, pi, player in recs:
            # z = outcome from the perspective of the player who moved at that state
            z = winner * player  # winner=1 means X won; multiply by player to get value for player-to-move
            dataset.append((board, pi, z))
        if (g+1) % 25 == 0:
            print(f"  collected from {g+1} games, dataset size = {len(dataset)}")

    if not dataset:
        print("No data collected; exiting.")
        return

    # Simple training loop
    print("Training on collected data...")
    for epoch in range(epochs):
        random.shuffle(dataset)
        losses = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            boards = torch.tensor([b for b,_,_ in batch], dtype=torch.float32, device=device)
            pis = torch.tensor([p for _,p,_ in batch], dtype=torch.float32, device=device)
            zs = torch.tensor([[z] for _,_,z in batch], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits, vals = model(boards)   # logits shape (B,9), vals shape (B,1)

            # policy loss: cross entropy expects raw logits and target class indices, but we have a distribution.
            # Use negative log-likelihood with pi as target distribution: L = - sum(pi * log_softmax(logits))
            logp = torch.log_softmax(logits, dim=1)
            policy_loss = - (pis * logp).sum(dim=1).mean()

            value_loss = mse(vals, zs)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}  loss={sum(losses)/len(losses):.4f}")

    # save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    self_play_games = 10000
    mcts_simulations = 50
    num_epochs = 10

    train(
        num_games=self_play_games,
        sims=mcts_simulations,
        epochs=num_epochs,
        batch_size=64,
        lr=1e-3,
        device="cpu"
    )


