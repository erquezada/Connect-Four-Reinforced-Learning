import random
import matplotlib.pyplot as plt # type: ignore
from IPython.display import clear_output
from board import *  # Assuming Board is defined in board.py

class QAgent:
    def __init__(self, Q_table=None, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        if Q_table is not None and not isinstance(Q_table, dict):
            raise TypeError("Q_table must be a dictionary or None")
        self.Q_table = Q_table if Q_table else {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def StateToKey(self, board):
        """Convert board state to a string key for Q-table lookup."""
        return ''.join(''.join(row) for row in board.board)

    def QLearningMove(self, player, board):
        state_key = self.StateToKey(board)
        available_columns = board.AvailableColumns()

        if state_key not in self.Q_table:
            self.Q_table[state_key] = {col: 0.0 for col in available_columns}

        # Select either a random or best action (epsilon-greedy)
        selected_column = random.choice(available_columns) if random.random() < self.epsilon else \
                        max(available_columns, key=lambda col: self.Q_table[state_key].get(col, 0.0))

        # Get the row where the piece should be placed in the selected column
        row = board.AvailableRowInColumn(selected_column)  # Pass only the column index
        
        if row != -1:  # If a valid row is found
            # Apply the move to the board
            next_board = board.copy()  # Create a copy of the board
            next_board.board[row][selected_column] = player  # Apply the move to the board
        else:
            next_board = board  # If no valid row, return the original board
        
        return next_board, selected_column

    def TrainQLearning(self, player, num_episode, board, num_simulations=1, output_type="verbose"):
        print("Training QLearning...")
        opponent = 'Y' if player == 'R' else 'R'
        win_rates = []

        for num_episode in range(num_simulations):
            board.reset()  # Reset board at the beginning of each simulation
            done, turn = False, player
            history, reward = [], 0  # Initialize history and reward

            while not done:
                state_key = self.StateToKey(board)
                available_columns = board.AvailableColumns()
                if not available_columns:
                    break

                # Select move based on epsilon-greedy strategy
                next_board, action = self.QLearningMove(turn, board) if turn == player else (
                    board.copy(), random.choice(available_columns))

                next_state_key = self.StateToKey(board)
                history.append((state_key, action, next_state_key, reward))  # Track state-action transitions

                board.board = next_board.board  # Update board state

                # Display board after each move (verbose mode)
                if output_type == "verbose":
                    print(f"After {turn}'s move (column {action + 1}):")
                    board.PrintBoard()

                # Check for win/loss/draw
                if board.CheckWin(player):
                    reward, done = 1, True
                elif board.CheckWin(opponent):
                    reward, done = -1, True
                elif not available_columns:  # Draw condition
                    reward, done = 0, True

                # Switch turns
                turn = opponent if turn == player else player

            # Backpropagate reward through history
            for state, action, next_state, reward in reversed(history):
                self.Q_table.setdefault(state, {})
                self.Q_table[state].setdefault(action, 0.0)
                self.Q_table.setdefault(next_state, {})

                # Update Q-values
                max_future = max(self.Q_table[next_state].values(), default=0)
                self.Q_table[state][action] += self.learning_rate * (reward + self.discount_factor * max_future - self.Q_table[state][action])
                reward *= self.discount_factor  # Discount reward

            # Print and plot win rates every 50 episodes
            if num_episode % 50 == 0:
                q_learning_win_rate = self.EvaluateAgent(player)
                win_rates.append((num_episode, q_learning_win_rate, 0.5))  # Adjust to your desired random win rate
                self.PlotLearningCurve(win_rates, live_update=True)
                print(f"Episode {num_episode}, Q-learning Win rate: {q_learning_win_rate:.2f}")

        print("Training completed.")
        self.PlotFinalResults(win_rates)
        return win_rates

    def EvaluateAgent(self, player, num_games=20):
        wins = 0
        opponent = 'Y' if player == 'R' else 'R'

        for _ in range(num_games):
            board = Board()  # Initialize a new game board
            turn = player
            done = False

            while not done:
                available_columns = board.AvailableColumns()
                if not available_columns:
                    break  # No available moves, the game ends

                if turn == player:
                    _, action = self.QLearningMove(player, board)
                    row = board.AvailableRowInColumn(action)
                    if row != -1:
                        board.board[row][action] = player
                else:
                    action = random.choice(available_columns)
                    row = board.AvailableRowInColumn(action)
                    if row != -1:
                        board.board[row][action] = opponent

                if board.CheckWin(player):  # Check if the current player wins
                    wins += 1
                    break
                elif board.CheckWin(opponent) or not board.AvailableColumns():  # Check if opponent wins or no moves left
                    break

                turn = opponent if turn == player else player  # Switch turns

        return wins / num_games  # Return win rate

    def PlotLearningCurve(self, win_rates, live_update=False):
        """Plot the learning curve of the agent's performance."""
        episodes = [entry[0] for entry in win_rates]
        q_learning_win_rates = [entry[1] for entry in win_rates]
        random_win_rates = [entry[2] for entry in win_rates]

        plt.figure(figsize=(8,5))
        plt.plot(episodes, q_learning_win_rates, marker='o', color='red', label='Q-learning Agent')
        plt.plot(episodes, random_win_rates, color='blue', label='Random Agent')
        plt.title('Win Rates over Training')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)

        if live_update:
            clear_output(wait=True)  # Ensure real-time updates in Jupyter environments
            plt.show()
        else:
            plt.show()

    def PlotFinalResults(self, win_rates):
        """Plot the final results of the training."""
        q_learning_win_rate = self.EvaluateAgent('R')  # Assuming 'R' is the player you want to evaluate
        random_win_rate = self.EvaluateAgent('Y')  # Assuming 'Y' is the opponent
        win_rates.append((len(win_rates), q_learning_win_rate, random_win_rate))
        print(f"Final Q-learning Win rate: {q_learning_win_rate:.2f}")
        print(f"Final Random Win rate: {random_win_rate:.2f}")
        self.PlotLearningCurve(win_rates, live_update=False)
