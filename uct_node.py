from board import *
import random
import copy
import math

class UCT_Node:
    def __init__(self, board, player, parent=None):
        self.board = copy.deepcopy(board)
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.q_value = 0
        self.untried_moves = board.AvailableColumns()
        self.q_values = None  # Store RL policy values

    def expand(self, rl_policy):
        move = random.choice(self.untried_moves)
        new_board = copy.deepcopy(self.board)
        self.board.AvailableRowInColumn(move)
        next_player = 'Y' if self.player == 'R' else 'R'
        child_node = UCT_Node(new_board, next_player, parent=self)
        child_node.q_values = rl_policy.predict(new_board)
        self.children[move] = child_node
        self.untried_moves.remove(move)
        return child_node

    def best_child(self, c_param=1.4):
        choices = []
        for move, child in self.children.items():
            exploit = child.q_value / (child.visits + 1e-8)
            explore = math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
            score = exploit + c_param * explore
            choices.append((score, move, child))
        _, move, best = max(choices)
        return best

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.board.CheckWin('R') or self.board.CheckWin('Y') or not self.board.AvailableColumns()
