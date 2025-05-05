import numpy as np # type: ignore
from board import *

class Dummy_RL_Policy:
    def predict(self, board):
        return np.random.rand(board.cols)
