import sys
from file_reader import File_Reader
from board import Board
from uct_tree import uct_tree
from uniform_random import Uniform_Random
from dummy_rl_policy import Dummy_RL_Policy
from q_agent import QAgent
from uct_node import UCT_Node
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from dqn_agent import *

class Main:
    SYMBOLS = {'B': 1, 'R': 2}  # Example mapping for colors
    num_simulations = 100  # Default number of simulations
    game_loop = True  # Flag to control the game loop

    def main(self):
        print("Main class initialized")

        # Initialize file reader and read file content
        self.file_reader = File_Reader("test.txt")
        algorithm_type, player_color, board_data = self.file_reader.read_file()
        print(f"Algorithm Type: {algorithm_type}")
        print(f"Player Color: {player_color}")
        print(f"Board Data: {board_data}")

        # Validate board data format
        if not isinstance(board_data, list) or not all(isinstance(row, list) for row in board_data) or not all(isinstance(cell, str) for row in board_data for cell in row):
            raise TypeError("Board data is not in the correct 2D list format!")

        self.board = Board(len(board_data), len(board_data[0]))
        self.board.board = board_data
        self.board.PrintBoard()

        # Start game loop
        while self.game_loop:
            algorithm_type = input("Enter algorithm type (UR, UCT, QL, DQN, C to quit): ").strip()

            if algorithm_type not in ["UR", "UCT", "QL", "DQN", "C"]:
                print(f"Invalid input: {algorithm_type}. Please try again.")
                continue

            if algorithm_type == "C":
                print("Exiting the game.")
                break

            print(f"\n--- Running {algorithm_type} Algorithm ---")
            if algorithm_type == "UR":
                Uniform_Random.UniformRandom(player_color, self.board, output_type="verbose")

            elif algorithm_type == "UCT":
                rl_policy = Dummy_RL_Policy()
                uct = uct_tree(player_color, rl_policy, self.board, num_simulations=self.num_simulations)
                move = uct.search(self.board)
                row = self.board.AvailableRowInColumn(move)
                if row != -1:
                    self.board.board[row][move] = player_color
                print(f"UCT selected column: {move + 1}")

            elif algorithm_type == "QL":
                rl_policy = Dummy_RL_Policy()
                q_agent = QAgent()
                q_agent.TrainQLearning(player_color, 1, self.board, num_simulations=self.num_simulations, output_type="verbose")
                _, selected_column = q_agent.QLearningMove(player_color, self.board)
                row = self.board.AvailableRowInColumn(selected_column)
                if row != -1:
                    self.board.board[row][selected_column] = player_color
                print(f"Q-Learning selected column: {selected_column + 1}")

            elif algorithm_type == "DQN":
                print(f"Training DQN Agent for {self.num_simulations} episodes...")
                agent, rewards, epsilon_values = TrainDQNAgent(player_color, self.num_simulations, self.board)
                state = np.reshape(self.board.board, [1, len(self.board.board) * len(self.board.board[0])])
                action = agent.act(state)
                row = self.board.AvailableRowInColumn(action)
                player_num = self.SYMBOLS[player_color]
                if row != -1:
                    self.board.board[row][action] = player_num
                print(f"DQN selected column: {action + 1}")

                plt.plot(agent.epsilon_list)
                plt.ylabel('Epsilon')
                plt.xlabel('Episode')
                plt.title('DQN Agent Training Epsilon')
                plt.show()

            # Print updated board
            print("Updated board:")
            self.board.PrintBoard()

            # Check for win or draw
            if self.board.CheckWin(player_color):
                print(f"Game Over! {player_color} Player wins.")
                self.game_loop = False
            elif not self.board.AvailableColumns():
                print("Game Over! It's a draw.")
                self.game_loop = False

# Run the main function
if __name__ == "__main__":
    runner = Main()
    runner.main()
