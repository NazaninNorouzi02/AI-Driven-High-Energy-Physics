import math
import torch
import torch.nn.functional as F
from .tree import Node

def mcts(root, net, n_simulations=50, c_puct=1.4):
    """
    Runs MCTS starting from 'root' and returns the best move (0..8).
    Args:
        root: Node object representing current game state
        net: Model to predict policy and value
        n_simulations: number of MCTS simulations per move
        c_puct: exploration constant for UCB (1.4 is standard)
    
    Returns:
        best_move (int): chosen move (0..8)
    
    """

    for _ in range(n_simulations):   # run many simulations to explore the tree

        node = root      # start at root
        path = [node]    # keep track of nodes visited for backpropagation

        # 1) SELECTION — traverse the tree until a leaf node
        while not node.is_leaf() and node.state.winner() is None:
            best_move = None
            best_score = -1e9   # initialize with very low score

            # compute total visits for UCB
            total_visits = sum(child.visits for child in node.children.values()) + 1
            
            # loop over all children and compute their UCB score
            for move, child in node.children.items():
                Q = child.value / (child.visits + 1e-6)  # average value (exploitation)
                U = c_puct * math.sqrt(total_visits) / (1 + child.visits)   # exploration
                score = Q + U

                if score > best_score:   # pick the child with max UCB
                    best_score = score
                    best_move = move

            node = node.children[best_move]  # move down the tree
            path.append(node)                # add to path

        # 2) EXPANSION — if leaf node is non-terminal, expand possible moves
        winner = node.state.winner()
        if winner is None:   # game not over
        
            # convert board to tensor for model input
            board = torch.tensor([node.state.board], dtype=torch.float32)
            with torch.no_grad():   # no gradient needed, just evaluation
                logits, value = net(board)    # get policy logits and value
            value = value.item()    # scalar value

            probs = F.softmax(logits, dim=1)[0].tolist()   # convert logits to probabilities

            legal_moves = node.state.moves()  # only expand legal moves
            node.children = {}                # create empty children dict

            for m in legal_moves:
                child_state = node.state.clone()   # copy board
                child_state.play(m)                # simulate move
                node.children[m] = Node(child_state, parent=node)   # add as child

        else:
            # Terminal value from this player's perspective
            value = float(winner * node.state.current_player)

        # 3) BACKUP — update all nodes in the path
        for n in reversed(path):
            n.visits += 1     # increment visit count
            n.value += value  # add the simulation outcome
            value = -value    # flip value for opposite player

    # After search: pick most visited move
    visits = {m: c.visits for m, c in root.children.items()}
    best_move = max(visits, key=visits.get)
    return best_move

