# ✨ Tic-Tac-Toe with MCTS + Neural Network

**This assignment was done with guidance from [Shadi Akbari](https://github.com/ShadiAkbari).**

This project implements Tic-Tac-Toe using **Monte Carlo Tree Search (MCTS)** guided by a **neural network**, inspired by AlphaZero. The goal is to see how an AI can **learn to play through self-play**, starting from zero knowledge, and improve over time.

---

## 🎯 Overview

The AI learns to play Tic-Tac-Toe without hard-coded rules. It has two components:

1. **Neural Network**
   - Architecture: 2 hidden layers, 64 units each, fully connected  
   - Input: 9-dimensional vector representing the board  
   - Outputs:
     - **Policy** → probabilities of moves
     - **Value** → expected outcome from current player’s perspective
     
2. **Monte Carlo Tree Search (MCTS)**
   - Runs multiple simulations from the current board  
   - Chooses moves using both **exploration** and **exploitation**  

---

## 💡 How MCTS Works

Each move is selected via repeated simulations:

1️⃣ **Selection**  
- Traverse the tree starting at the current board  
- Choose child nodes with the **highest UCB score**  
- Balances:
  - Exploration: trying less-visited moves  
  - Exploitation: choosing promising moves  

2️⃣ **Expansion**  
- If a node is unvisited and game not finished:
  - Network predicts **policy & value**  
  - Only legal moves are added as children  

3️⃣ **Evaluation**  
- Neural network estimates value for current player:  
  - `+1` → likely win  
  - `0` → draw  
  - `-1` → likely loss  

4️⃣ **Backpropagation**  
- Value is propagated up the tree  
- Updates visit counts and node totals  
- After all simulations, move with **highest visit count** is chosen  

---

## 🔄 Self-Play and Training

AI improves via repeated self-play:

- **Collect data** from multiple games:
  - Board state
  - Improved MCTS move probabilities (π)
  - Final outcome (z)
- **Train the network**:
  - Loss = policy_loss + value_loss
- **Repeat**:
  - New model → better MCTS → stronger AI

### Training Parameters
- Self-play games: 10,000 (default, can increase)  
- MCTS simulations per move: 50 (can increase for stronger AI)  
- Epochs: 10 (training iterations over collected data)  
- Batch size: 64  
- Optimizer: Adam, learning rate = 1e-3  

> These parameters can be adjusted to generate a smarter model checkpoint.

---

## ⚡ Running the Project

- **Training:** `train.py`  
- **AI vs AI:** `play.py`  
- **Human vs AI:** `play_vs_ai.py`  

**Human vs AI example:**
- Board is printed as a 9-element array,
- Enter your move index when prompted.  
- AI responds automatically using the trained model.

- **Checkpoint:** `checkpoints/model.pth`

---

## 📂 Folder Contents

- `models.py` → neural network  
- `mcts.py` → MCTS logic  
- `train.py` → self-play and training loop  
- `play.py` → AI vs AI  
- `play_vs_ai.py` → Human vs AI  
- `game.py` & `tree.py` → frozen game logic  
- `checkpoints/model.pth` → trained model  

---
