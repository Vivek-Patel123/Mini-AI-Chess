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
        valid_moves = []
        for row in range(5):
            for col in range(5):
                piece = game_state["board"][row][col]
                if piece != '.' and piece[0] == game_state["turn"][0]:  # Check if it's the current player's piece
                    # Generate all valid moves for this piece
                    for move in self.valid_moves_for_piece(game_state, (row, col)):
                        valid_moves.append(((row, col), move))
        return valid_moves

    def valid_moves_for_piece(self, game_state, start):
        # Generate valid moves for a given piece at the specified start position
        valid_moves = []
        piece = game_state["board"][start[0]][start[1]]
        for row in range(5):
            for col in range(5):
                move = ((start[0], start[1]), (row, col))
                if self.is_valid_move(game_state, move):
                    valid_moves.append((row, col))
        return valid_moves

    """
    Evaluate the game state using the chosen heuristic.

    Args:
        - game_state:   dictionary | Current game state
        - heuristic:     string   | Heuristic function to use ('e0', 'e1', 'e2')

    Returns:
        - int   | The evaluation score for the current game state
    """
    def evaluate_game_state(self, game_state, heuristic):
        if heuristic == "e0":
            return self.heuristic_e0(game_state)
        elif heuristic == "e1":
            return self.heuristic_e1(game_state)
        elif heuristic == "e2":
            return self.heuristic_e2(game_state)
        else:
            raise ValueError("Invalid heuristic")


    def heuristic_e0(self, game_state):
        piece_values = {
        "wp": 1, "wB": 3, "wN": 3, "wQ": 9, "wK": 999,
        "bp": -1, "bB": -3, "bN": -3, "bQ": -9, "bK": -999
    }

        score = 0

        for piece, value in piece_values.items():
            score += value * self.specific_piece_number(game_state, piece)

        return score

    def heuristic_e1(self, game_state):
        piece_values = {
            "wp": 1, "wB": 3, "wN": 3, "wQ": 9, "wK": 999,
            "bp": -1, "bB": -3, "bN": -3, "bQ": -9, "bK": -999
        }

        center_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]  # The center of the board
        score = 0

        # Iterate over all positions on the board
        for row in range(5):
            for col in range(5):
                piece = game_state["board"][row][col]  # Get the piece at the current position

                if piece:  # If there's a piece on the current square
                    piece_name = piece  # piece is already in the form "wp", "bQ", etc.

                    # Check if the piece exists in the piece_values dictionary
                    if piece_name in piece_values:
                        piece_score = piece_values[piece_name]
                        
                        # Enhance the score based on the piece's proximity to the center
                        if (row, col) in center_positions:
                            piece_score += 0.5  # Bonus for being in the center

                        # Further enhancement can be made here based on other factors like piece safety

                        score += piece_score  # Add the piece's score to the total score

        return score

    def heuristic_e2(self, game_state):
        piece_values = {
            "wp": 1, "wB": 3, "wN": 3, "wQ": 9, "wK": 999,
            "bp": -1, "bB": -3, "bN": -3, "bQ": -9, "bK": -999
        }

        safety_values = {
            "wp": 0.5, "wB": 0.7, "wN": 0.7, "wQ": 1, "wK": 2,
            "bp": 0.5, "bB": 0.7, "bN": 0.7, "bQ": 1, "bK": 2
        }

        center_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]  # The center of the board
        score = 0

        # Iterate over all positions on the board
        for row in range(5):
            for col in range(5):
                piece = game_state["board"][row][col]  # Get the piece at the current position

                if piece:  # If there's a piece on the current square
                    piece_name = piece  # piece is already in the form "wp", "bQ", etc.

                    # Check if the piece exists in the piece_values dictionary
                    if piece_name in piece_values:
                        piece_score = piece_values[piece_name]
                        
                        # Add piece safety bonus based on the piece type
                        if piece_name in safety_values:
                            piece_score += safety_values[piece_name]

                        # Bonus for being in the center
                        if (row, col) in center_positions:
                            piece_score += 0.5  # Bonus for being in the center

                        # Further enhancement based on material balance (the difference in material value)
                        score += piece_score

        return score
    
    def specific_piece_number(self, game_state, specific_piece):
        return sum(1 for row in game_state["board"] for piece in row if piece == specific_piece)
    
    """
        Minimax algorithm with depth-limited search and time limit.

        Args:
            - game_state:       dict  | Current game state
            - depth:            int   | Current depth of the search
            - maximizing_player: bool | True if maximizing player's turn, False otherwise
            - start_time:       float | Time when the search started
            - time_limit:       float | Maximum time allowed for the search
            - heuristic:        str   | Heuristic function to use ('e0', 'e1', 'e2')

        Returns:
            - best_move:        tuple | The best move found
            - best_value:       float | The heuristic value of the best move
            - action_time:      float | Time taken to compute the move
            - heuristic_score:  float | Heuristic score of the resulting board
            - search_score:     float | Heuristic score returned by the Minimax search
            - states_explored:  int   | Number of states explored during the search
            - branching_factor: float | Average branching factor for the current depth
            - states_explored_by_depth: dict | States explored at each depth
        """
    def minimax(self, game_state, depth, maximizing_player, start_time, time_limit, heuristic):
        start_action_time = time.time()

        # Check if time limit has been exceeded, depth reaches 0, or the game is over
        if time.time() - start_time > time_limit or depth == 0 or self.is_game_over(game_state):
            heuristic_score = self.evaluate_game_state(game_state, heuristic)
            action_time = time.time() - start_action_time
            return None, heuristic_score, action_time, heuristic_score, heuristic_score, 1, 0, {depth: 1}

        valid_moves = self.valid_moves(game_state)
        states_explored_this_depth = 0
        total_branching_factor = 0
        states_explored_by_depth = {}

        if maximizing_player:
            best_value = -math.inf
            best_move = None

            for move in valid_moves:
                new_game_state = copy.deepcopy(game_state)
                self.make_move(new_game_state, move)

                # Recurse with updated game state
                _, value, _, _, search_score, states_explored, branching_factor, depth_states = self.minimax(
                    new_game_state, depth - 1, False, start_time, time_limit, heuristic
                )

                # Update states explored per depth
                states_explored_this_depth += states_explored
                total_branching_factor += branching_factor
                states_explored_by_depth = self.update_depth_states(states_explored_by_depth, depth_states)

                if value > best_value:
                    best_value = value
                    best_move = move

            # Calculate average branching factor
            if len(valid_moves) > 0:
                branching_factor = total_branching_factor / len(valid_moves)
            else:
                branching_factor = 0

            # Calculate action time
            action_time = time.time() - start_action_time

            # Heuristic score of the resulting board after the best move
            if best_move:
                resulting_state = copy.deepcopy(game_state)
                self.make_move(resulting_state, best_move)
                heuristic_score = self.evaluate_game_state(resulting_state, heuristic)
            else:
                heuristic_score = self.evaluate_game_state(game_state, heuristic)

            return best_move, best_value, action_time, heuristic_score, best_value, states_explored_this_depth, branching_factor, states_explored_by_depth

        else:  # Minimizing player
            best_value = math.inf
            best_move = None

            for move in valid_moves:
                new_game_state = copy.deepcopy(game_state)
                self.make_move(new_game_state, move)

                # Recurse with updated game state
                _, value, _, _, search_score, states_explored, branching_factor, depth_states = self.minimax(
                    new_game_state, depth - 1, True, start_time, time_limit, heuristic
                )

                # Update states explored per depth
                states_explored_this_depth += states_explored
                total_branching_factor += branching_factor
                states_explored_by_depth = self.update_depth_states(states_explored_by_depth, depth_states)

                if value < best_value:
                    best_value = value
                    best_move = move

            # Calculate average branching factor
            if len(valid_moves) > 0:
                branching_factor = total_branching_factor / len(valid_moves)
            else:
                branching_factor = 0

            # Calculate action time
            action_time = time.time() - start_action_time

            # Heuristic score of the resulting board after the best move
            if best_move:
                resulting_state = copy.deepcopy(game_state)
                self.make_move(resulting_state, best_move)
                heuristic_score = self.evaluate_game_state(resulting_state, heuristic)
            else:
                heuristic_score = self.evaluate_game_state(game_state, heuristic)

            return best_move, best_value, action_time, heuristic_score, best_value, states_explored_this_depth, branching_factor, states_explored_by_depth

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
        draw_turn_count = 0
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
                turn_count += 1
                if new_pieces == pieces:
                    draw_turn_count += 1
                else: 
                    pieces = new_pieces
                    draw_turn_count = 0

                if draw_turn_count == 20:
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