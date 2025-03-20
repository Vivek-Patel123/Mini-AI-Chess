import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()
        self.total_states_explored = 0  # Initialize the attribute
        self.states_explored_by_depth = {i: 0 for i in range(10)}  # Example depth dictionary
        self.branching_factors = []

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
    Convert board coordinates back into string move notation.

    Args:
        - move: tuple ((start_row, start_col), (end_row, end_col))
    
    Returns:
        - string representing a move in the format "B2 B3"
    """
    def format_move(self, move):
        try:
            (start, end) = move
            start_str = f"{chr(start[1] + ord('A'))}{5 - start[0]}"
            end_str = f"{chr(end[1] + ord('A'))}{5 - end[0]}"
            return f"{start_str} {end_str}"
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
        gamemode = 0
        timeout = 0
        max_turns = 0
        alpha_beta = False
        heuristic = "e0"
        states_explored = 0
        total_nodes_expanded = 0  # Track nodes expanded for branching factor calculation

        pieces = self.check_number_pieces(self.current_game_state)
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.\n")

        while True:
            print("Enter your desired gamemode \n" + 
                "1 - Human vs Human \n" + 
                "2 - Human vs AI \n" + 
                "3 - AI vs Human\n" +
                "4 - AI vs AI")
            gamemode = input().strip()
            if gamemode in {"1", "2", "3", "4"}:
                break

        if gamemode in {"2", "3", "4"}:
            while True:
                timeout = input("Enter the value of the timeout in seconds: ").strip()
                if timeout.isdigit():
                    timeout = int(timeout)
                    break

            while True:
                max_turns = input("Enter the maximum amount of turns for the game: ").strip()
                if max_turns.isdigit():
                    max_turns = int(max_turns)
                    break

            while True:
                alpha_beta_input = input("Do you want to enable alpha-beta? Enter Yes or No: ").strip().lower()
                if alpha_beta_input in {"yes", "no"}:
                    alpha_beta = alpha_beta_input == "yes"
                    break

            while True:
                heuristic = input("Which heuristic do you want to use: e0, e1 or e2? ").strip().lower()
                if heuristic in {"e0", "e1", "e2"}:
                    break

        file_name = f"gameTrace-{alpha_beta}-{timeout}-{max_turns}.txt"
        with open(file_name, "w") as file:
            gamemode_names = {"1": "Human vs Human", "2": "Human vs AI", "3": "AI vs Human", "4": "AI vs AI"}
            file.write(f"Gamemode: {gamemode_names[gamemode]}\n")

            if gamemode in {"2", "3", "4"}:
                file.write(f"Timeout value: {timeout}\nMax turns: {max_turns}\nAlpha-beta: {alpha_beta}\nHeuristic used: {heuristic}\n")
            file.write(self.display_board(self.current_game_state) + "\n")

            while True:
                print(self.display_board(self.current_game_state))
                if gamemode == "1" or (gamemode == "2" and self.current_game_state['turn'] == "white") or (gamemode == "3" and self.current_game_state['turn'] == "black"):
                    move = input(f"{self.current_game_state['turn'].capitalize()} to move: ").strip()
                    if move.lower() == 'exit':
                        file.write(f"Player: {self.current_game_state['turn']}\nTurn #{turn_count//2 + 1}\nMove: {move}\n")
                        print("Game exited.")
                        return  # Use return instead of exit()

                    parsed_move = self.parse_input(move)
                    if not parsed_move or not self.is_valid_move(self.current_game_state, parsed_move):
                        print("Invalid move. Try again.")
                        continue

                    file.write(f"Player: {self.current_game_state['turn']}\nTurn #{turn_count//2 + 1}\nMove: {move}\n")
                    self.make_move(self.current_game_state, parsed_move)
                    file.write(self.display_board(self.current_game_state) + "\n")

                else:
                    print(f"AI ({self.current_game_state['turn']}) is thinking...")
                    start_time = time.time()

                    states_explored_by_depth = {}
                    if alpha_beta:
                        move, value, action_time, heuristic_score, alpha_beta_score, states_explored_AI, branching_factor, states_explored_by_depth = self.alpha_beta(
                            self.current_game_state, 3, False, -math.inf, math.inf, start_time, timeout, heuristic)
                    else:
                        move, value, action_time, heuristic_score, alpha_beta_score, states_explored_AI, branching_factor, states_explored_by_depth = self.minimax(
                            self.current_game_state, 3, False, start_time, timeout, heuristic)

                    states_explored += states_explored_AI
                    total_nodes_expanded += len(states_explored_by_depth)  # Count nodes that were expanded
                    total_states = sum(states_explored_by_depth.values())

                    # Ensure division by zero is avoided
                    if total_nodes_expanded > 0:
                        branching_factor = total_states / total_nodes_expanded
                    else:
                        branching_factor = 0

                    states_by_depth_percent = {depth: (count / total_states) * 100 for depth, count in states_explored_by_depth.items()} if total_states > 0 else {}

                    print(f"AI ({self.current_game_state['turn']}): {self.format_move(move)}")
                    file.write(f"AI: {self.current_game_state['turn']}\nTurn #{turn_count // 2 + 1}\nMove: {self.format_move(move)}\n")
                    file.write(f"Action Time: {action_time:.2f} sec\n")
                    file.write(f"Heuristic Score: {heuristic_score}\nSearch Score: {alpha_beta_score}\n")
                    file.write(f"Cumulative States Explored: {states_explored}\nCumulative States Explored by Depth:\n")

                    for depth, count in states_explored_by_depth.items():
                        file.write(f"  Depth {depth}: {count} states ({states_by_depth_percent.get(depth, 0):.1f}%)\n")

                    file.write(f"Average Branching Factor: {branching_factor:.2f}\n")

                    if not self.is_valid_move(self.current_game_state, move):
                        print("Invalid move by AI.")
                        print("Game over!")
                        exit(1)
                    
                    self.make_move(self.current_game_state, move)
                    file.write(self.display_board(self.current_game_state) + "\n")

                    if time.time() - start_time > timeout:
                        print("Timeout reached, game over!")
                        file.write("Game over: Timeout reached\n")
                        return

                if self.is_game_over(self.current_game_state):
                    print(self.display_board(self.current_game_state))
                    winner = "White" if self.current_game_state["turn"] == "black" else "Black"
                    print(f"{winner} has won the game in {turn_count // 2 + 1} turns!")
                    file.write(f"{winner} has won the game in {turn_count // 2 + 1} turns!\n")
                    return

                new_pieces = self.check_number_pieces(self.current_game_state)
                turn_count += 1
                if new_pieces == pieces:
                    draw_turn_count += 1
                else:
                    pieces = new_pieces
                    draw_turn_count = 0

                if draw_turn_count == 20:
                    print("It's a draw!\nGame over")
                    file.write("It's a draw!\n")
                    return

                if gamemode != "1" and turn_count >= (max_turns * 2):
                    print(self.display_board(self.current_game_state))
                    print("Maximum turns reached. Game over!")
                    file.write("Game over: Maximum turns reached\n")
                    return


    def is_game_over(self, game_state):
        king_count = 0
        for row in game_state["board"]:
            for piece in row:
                if piece in {"wK", "bK"}:
                    king_count += 1

        if king_count == 2:
            return False
        else: return True

    """
    Check how many pieces are still on the board
    """
    def check_number_pieces(self, game_state):
        return sum(1 for row in game_state["board"] for piece in row if piece != '.')

    

if __name__ == "__main__":
    game = MiniChess()
    game.play()