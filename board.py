ROWS = 6
COLUMNS = 7
SYMBOLS = {'o': 0, 'r': 1, 'y': 2, 'red': 1, 'yellow': 2}
INT_TO_SYMBOL = {0: 'O', 1: 'R', 2: 'Y'}
Q_table = {}

class Board:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [['O' for _ in range(cols)] for _ in range(rows)]

    def reset(self):
        """Reset the board to an empty state."""
        self.board = [['O' for _ in range(self.cols)] for _ in range(self.rows)]

    def copy(self):
        """Return a copy of the board."""
        new_board = Board(self.rows, self.cols)
        new_board.board = [row.copy() for row in self.board]  # Ensure deep copy of each row
        return new_board

    def PrintBoard(self):
        """Print the current state of the board."""
        for row in self.board:
            print('|'.join(row))
        print('-' * (self.cols * 2 - 1))

    def AvailableColumns(self):
        """Return a list of columns that have space available."""
        return [col for col in range(self.cols) if self.board[0][col] == 'O']

    def AvailableRowInColumn(self, column):
        """Find the first available row in the column."""
        for row in reversed(range(self.rows)):  # Start from the bottom row
            if self.board[row][column] == 'O':
                return row
        return -1  # If the column is full, return -1

    def CheckWin(self, player):
        """Check if the player has won."""
        # Horizontal check
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True

        # Vertical check
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True

        # Diagonal (bottom-left to top-right) check
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True

        # Diagonal (top-left to bottom-right) check
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True

        return False

    def StateToKey(self):
        return tuple(tuple(SYMBOLS[cell.lower()] for cell in row) for row in self.board)
