class File_Reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        try:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
                algorithm_type = lines[0].strip()
                player_color = lines[1].strip()
                board_lines = lines[2:8]  # 6 rows for the board
                board = [list(row.strip()) for row in board_lines]  # Keep strings
                return algorithm_type, player_color, board
        except FileNotFoundError:
            print(f"File {self.file_path} not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
