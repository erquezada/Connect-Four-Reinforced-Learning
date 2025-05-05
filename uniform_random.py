from board import *  # Assuming necessary imports
import random

class Uniform_Random:
    @staticmethod
    def UniformRandom(player, board, output_type):
        opponent = 'Y' if player == 'R' else 'R'
        turn = player
        done = False

        while not done:
            if output_type.lower() == "verbose":
                print(f"Current board:")
                board.PrintBoard()

            available_columns = board.AvailableColumns()  # Get available columns

            if not available_columns:
                print("Game over: Board is full.")
                break

            selected_column = random.choice(available_columns)
            row = board.AvailableRowInColumn(selected_column)
            board.board[row][selected_column] = turn

            print(f"Move selected: {selected_column + 1}\n")

            if output_type.lower() != "none":
                print(f"Updated board:")
                board.PrintBoard()

            # Check if the current player wins
            if board.CheckWin(turn):
                print(f"Game Over! {turn} Player wins.")
                done = True
            elif not board.AvailableColumns():  # Check for a draw
                print("Game Over! It's a draw.")
                done = True
            else:
                turn = opponent if turn == player else player  # Switch turn
