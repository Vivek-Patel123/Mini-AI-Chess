import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()

    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
                "board": 
                [['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']],
                "turn": 'white',
                }
        return state

    """
    Prints the board
    
    Args:
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        board_str = "\n"
        for i, row in enumerate(game_state["board"], start=1):
            board_str += str(6 - i) + "  " + ' '.join(piece.rjust(3) for piece in row) + "\n"
        board_str += "\n     A   B   C   D   E\n\n"
        return board_str


    """
    Check if the move is valid    
    
    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """
    def is_valid_move(self, game_state, move):
        # Check if move is in list of valid moves
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end

        if (start_row < 0 or start_row > 4) or (start_col < 0 or start_col > 4) or \
            (end_row < 0 or end_row > 4) or (end_col < 0 or end_col > 4):
                return False


        piece = game_state["board"][start_row][start_col]

        if piece == '.':
            return False

        if (game_state["turn"] == "black" and piece[0] == 'w') or (game_state["turn"] == "white" and piece[0] == 'b'): 
            return False

        if not self.tile_check(game_state, end_row, end_col):
            return False
        
        match piece:
            case 'wK' | 'bK':
                if abs(end_row - start_row) <= 1 and abs(end_col - start_col) <= 1:
                    return True
                else: return False
            
            case 'wQ' | 'bQ':
                if end_row == start_row or end_col == start_col or abs(end_row - start_row) == abs(end_col - start_col):
                    return self.is_path_clear(game_state, start, end)
                return False
            
            case 'wB' | 'bB':
                if abs(end_row - start_row) == abs(end_col - start_col):
                    return self.is_path_clear(game_state, start, end)
                return False
                    
            case 'wN' | 'bN':
                return (abs(end_row - start_row) == 2 and abs(end_col - start_col) == 1) or \
                (abs(end_row - start_row) == 1 and abs(end_col - start_col) == 2)

            case 'wp':  
                if end_col == start_col and end_row == start_row - 1:
                    if game_state["board"][end_row][end_col] == '.':
                        return True

                if abs(end_col - start_col) == 1 and end_row == start_row - 1:
                    if game_state["board"][end_row][end_col] != '.' and game_state["board"][end_row][end_col][0] == 'b':
                        return True

                return False

            case 'bp':  
                if end_col == start_col and end_row == start_row + 1:
                    if game_state["board"][end_row][end_col] == '.':
                        return True

                if abs(end_col - start_col) == 1 and end_row == start_row + 1:
                    if game_state["board"][end_row][end_col] != '.' and game_state["board"][end_row][end_col][0] == 'w':
                        return True

                return False


        return True

    """
    Check if the piece is jumping over another one or not
    Returns False if it is
    """

    def is_path_clear(self, game_state, start, end):
        start_row, start_col = start
        end_row, end_col = end
        board = game_state["board"]

        row_step = 0 if start_row == end_row else (1 if end_row > start_row else -1)
        col_step = 0 if start_col == end_col else (1 if end_col > start_col else -1)

        row, col = start_row + row_step, start_col + col_step
        while (row, col) != (end_row, end_col):
            if board[row][col] != '.': 
                return False
            row += row_step
            col += col_step

        return True


    """
    Check if the tile the player is trying to go to is already occupied by another one of his pieces
    """

    def tile_check(self, game_state, end_row, end_col):
        if game_state["turn"] == "black" and game_state["board"][end_row][end_col][0] == 'b':
            return False

        if game_state["turn"] == "white" and game_state["board"][end_row][end_col][0] == 'w':
            return False
        return True

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
    """
    def valid_moves(self, game_state):
        # Return a list of all the valid moves.
        # Implement basic move validation
        # Check for out-of-bounds, correct turn, move legality, etc
        return

    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'


        if piece == "wp" and end_row == 0:
            game_state["board"][end_row][end_col] = "wQ"
        elif piece == "bp" and end_row == 4:
            game_state["board"][end_row][end_col] = "bQ"
        else: game_state["board"][end_row][end_col] = piece

        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"

        return game_state

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        turn_count = 0
        pieces = self.check_number_pieces(self.current_game_state)
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")

        with open("gameTrace-false-5-na.txt", "w") as file:
            file.write(self.display_board(self.current_game_state) + "\n")

            while True:
                print(self.display_board(self.current_game_state))
                move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
                raw_move = move
                if move.lower() == 'exit':
                    file.write("Player: " + self.current_game_state["turn"]+ "\n")
                    file.write("Turn #" + str(turn_count//2 + 1) + "\n")
                    file.write("Move: " + raw_move + "\n")
                    print("Game exited.")
                    exit(1)

                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    continue
        
                file.write("Player: " + self.current_game_state["turn"]+ "\n")
                file.write("Turn #" + str(turn_count//2 + 1) + "\n")
                file.write("Move: " + str(raw_move) + "\n")
                self.make_move(self.current_game_state, move)
                file.write(self.display_board(self.current_game_state) + "\n")

                if self.is_game_over(self.current_game_state) != 2:
                    print(self.display_board(self.current_game_state))
                    winner = "White" if self.current_game_state["turn"] == "black" else "Black"
                    print(f"{winner} has won the game in {turn_count} turns!")
                    file.write(f"{winner} has won the game in {turn_count} turns!")
                    exit(1)

                new_pieces = self.check_number_pieces(self.current_game_state)
                if new_pieces == pieces:
                    turn_count += 1
                else: 
                    pieces = new_pieces
                    turn_count = 0

                if turn_count == 20:
                    print("It's a draw!")
                    print("Game over")
                    file.write("It's a draw!")
                    exit(1)

    def is_game_over(self, game_state):
        king_count = 0
        for row in game_state["board"]:
            for piece in row:
                if piece in {"wK", "bK"}:
                    king_count += 1
        return king_count 

    """
    Check how many pieces are still on the board
    """
    def check_number_pieces(self, game_state):
        return sum(1 for row in game_state["board"] for piece in row if piece != '.')

    

if __name__ == "__main__":
    game = MiniChess()
    game.play()