import random
from collections import deque
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from board import *  # Assumes Board, ROWS, and COLUMNS are defined in board.py

SYMBOLS = {'r': 1, 'y': 2, 'red': 1, 'yellow': 2, 'R': 1, 'Y': 2}
INT_TO_SYMBOL = {0: 'O', 1: 'R', 2: 'Y'}


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def TrainDQNAgent(player_color, EPISODES, board, make_plot=True):
    state_size = ROWS * COLUMNS
    action_size = COLUMNS
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    rewards = []
    epsilon_values = []

    player_color = player_color.lower()
    if player_color not in SYMBOLS:
        raise ValueError(f"Invalid player color: {player_color}. Valid colors: 'R', 'Y', 'red', 'yellow'.")
    player = SYMBOLS[player_color]

    for e in range(EPISODES):
        current_board = board.copy()
        total_reward = 0
        done = False

        while not done:
            state = np.reshape(current_board.StateToKey(), [1, state_size])
            action = agent.act(state)

            if current_board.board[0][action] != 'O':
                reward = -10
                done = True
            else:
                row = current_board.AvailableRowInColumn(action)
                if row == -1:
                    reward = -10
                    done = True
                else:
                    current_board.board[row][action] = INT_TO_SYMBOL[player]
                    if current_board.CheckWin(INT_TO_SYMBOL[player]):
                        reward = 10
                        done = True
                    else:
                        avail_cols = current_board.AvailableColumns()
                        if not avail_cols:
                            reward = 0
                            done = True
                        else:
                            opp_action = random.choice(avail_cols)
                            opp_row = current_board.AvailableRowInColumn(opp_action)
                            current_board.board[opp_row][opp_action] = INT_TO_SYMBOL[3 - player]
                            if current_board.CheckWin(INT_TO_SYMBOL[3 - player]):
                                reward = -10
                                done = True
                            else:
                                reward = 0

            next_state = np.reshape(current_board.StateToKey(), [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            if done:
                break

            agent.replay(batch_size)

        rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)
        print(f"Episode: {e + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    if make_plot:
        plt.plot(rewards)
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('DQN Agent Training Rewards')
        plt.show()

        plt.plot(epsilon_values)
        plt.ylabel('Epsilon')
        plt.xlabel('Episode')
        plt.title('DQN Agent Epsilon Over Time')
        plt.show()

    return agent, rewards, epsilon_values


# Example: Initialize and print model summary
if __name__ == "__main__":
    dummy_board = Board()
    agent = DQNAgent(state_size=ROWS * COLUMNS, action_size=COLUMNS)
    agent.model.summary()
