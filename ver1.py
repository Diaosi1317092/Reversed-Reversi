import numpy as np
import random
import time
import copy

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

# Define the board weight matrix (8×8)
WEIGHT_MATRIX = np.array([
    [1, 8, 3, 7, 7, 3, 8, 1],
    [8, 3, 2, 5, 5, 2, 3, 8],
    [3, 2, 6, 6, 6, 6, 2, 3],
    [7, 5, 6, 4, 4, 6, 5, 7],
    [7, 5, 6, 4, 4, 6, 5, 7],
    [3, 2, 6, 6, 6, 6, 2, 3],
    [8, 3, 2, 5, 5, 2, 3, 8],
    [1, 8, 3, 7, 7, 3, 8, 1]
])

# AI Class
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size  # Board size (default 8×8)
        self.color = color                      # The color the AI plays
        self.time_out = time_out                # Maximum move time (4.8 sec per move)
        self.candidate_list = []                # List to store candidate moves
        self.total_time_used = 0                # Total time used for moves
        self.move_count = 0                     # Count of moves made by the AI

    # Get the list of opponent piece positions that can be flipped if a move is made at (row, col)
    def get_flips(self, board, row, col, color):
        flips = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            temp_flips = []
            r, c = row + dr, col + dc
            while 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size:
                if board[r, c] == -color:
                    temp_flips.append((r, c))
                elif board[r, c] == color:
                    if temp_flips:
                        flips.extend(temp_flips)
                    break
                else:
                    break
                r += dr
                c += dc
        return flips

    # Generate all valid moves with their weight values
    def generate_valid_moves(self, board, color):
        moves = []
        for row in range(self.chessboard_size):
            for col in range(self.chessboard_size):
                if board[row, col] == COLOR_NONE:
                    flips = self.get_flips(board, row, col, color)
                    if flips:
                        w = WEIGHT_MATRIX[row, col]
                        moves.append(((row, col), flips, w))
        return moves

    # Simulate making a move on the board
    def apply_move(self, board, move, color):
        new_board = board.copy()
        (row, col), flips, _ = move
        new_board[row, col] = color
        for r, c in flips:
            new_board[r, c] = COLOR_NONE
        return new_board

    # Minimax search with alpha-beta pruning
    def minimax(self, board, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError

        valid_moves = self.generate_valid_moves(board, self.color if maximizing else -self.color)
        if depth == 0 or not valid_moves:
            return 0, None  # No evaluation as we're trying to occupy the highest weight

        best_move = None
        if maximizing:
            value = -float('inf')
            for move in valid_moves:
                new_board = self.apply_move(board, move, -self.color)
                score, _ = self.minimax(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
                if move[2] > value:  # Opponent tries to maximize the weight
                    value = move[2]
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move
        else:
            value = -float('inf')
            for move in valid_moves:
                new_board = self.apply_move(board, move, self.color)
                score, _ = self.minimax(new_board, depth - 1, alpha, beta, True, start_time, time_limit)
                if move[2] > value:  # AI tries to maximize the weight
                    value = move[2]
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move

    # Iterative deepening search within 4.8 seconds per move
    def iterative_deepening(self, board):
        start_time = time.time()
        time_limit = 4.8  # Max time per move
        depth = 3  # Start with depth 3
        best_move = None
        while True:
            try:
                score, move = self.minimax(board, depth, -float('inf'), float('inf'), True, start_time, time_limit)
                if move is not None:
                    best_move = move
                depth += 1
            except TimeoutError:
                break
            if time.time() - start_time > time_limit:
                break
        return best_move

    # The main function to choose a move
    def go(self, chessboard):
        self.candidate_list.clear()
        start_time = time.time()

        best_move = self.iterative_deepening(chessboard)
        if best_move is None:
            return []
        chosen_pos = best_move[0]

        valid_moves = self.generate_valid_moves(chessboard, self.color)
        for move in valid_moves:
            self.candidate_list.append(move[0])
        self.candidate_list.append(chosen_pos)

        self.total_time_used += time.time() - start_time
        self.move_count += 1

        return self.candidate_list
