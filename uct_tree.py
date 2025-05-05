from uct_node import UCT_Node
import random
import numpy as np  # type: ignore
import copy

class uct_tree:
    def __init__(self, player, rl_policy, board, num_simulations=1000):
        self.player = player
        self.rl_policy = rl_policy
        self.num_simulations = num_simulations
        self.exploration_weight = 1.0
        
        # Initialize the board explicitly
        self.board = board
        
        # Create the UCT_Node with the provided board
        self.root = UCT_Node(self.board, self.player)
        self.root.q_values = rl_policy.predict(self.board)

    def search(self, board):
    # Initialize the root node based on the current board state
        root = UCT_Node(board, self.player)

    # Check if the board is full, if so return None or some "game over" state
        if not board.AvailableColumns():
            print("Game over: Board is full.")
            return None  # Or handle the game over state appropriately

        # Simulate the search for a certain number of iterations
        for _ in range(self.num_simulations):  # Or any other number of iterations for the search
            node_to_expand = self.select_node(root)  # Select the node to expand
            self.expand_node(node_to_expand)  # Expand the selected node
            reward = self.rollout(node_to_expand.board, self.player)  # Simulate a random game from the expanded node
            self.backpropagate(node_to_expand, reward)  # Backpropagate the results

        # After simulation, pick the best move
        if not root.children:
            print("Warning: No children in root node, returning a random move.")
            return random.choice(board.AvailableColumns())  # Fallback to random if no children

        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move
    def select_node(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand(self.rl_policy)
            else:
                node = node.best_child()
        return node
    def expand_node(self, node):
        if not node.is_fully_expanded():
            return node.expand(self.rl_policy)
        return None
    
    def rollout(self, board, player):
        rollout_board = copy.deepcopy(board)
        turn = player

        while True:
            if rollout_board.CheckWin('R'):
                return 1 if self.player == 'R' else -1
            if rollout_board.CheckWin('Y'):
                return 1 if self.player == 'Y' else -1
            available_moves = rollout_board.AvailableColumns()
            if not available_moves:
                return 0  # Draw, if no available moves

            # Select a random move using the RL policy
            q_values = self.rl_policy.predict(rollout_board)
            best_move = np.argmax(q_values)

            # Get the row where the piece should be placed
            row = rollout_board.AvailableRowInColumn(best_move)
            if row != -1:  # Ensure the column is not full
                rollout_board.board[row][best_move] = 'R' if turn == 'R' else 'Y'

            # Switch turns between 'R' and 'Y'
            turn = 'Y' if turn == 'R' else 'R'

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.q_value += reward
            reward = -reward
            node = node.parent