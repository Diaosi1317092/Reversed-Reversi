import numpy as np
import random
import time
from numba import njit

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

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

@njit
def get_flips_numba(board, row, col, color, board_size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    flips = []
    for d in directions:
        dr, dc = d
        r, c = row + dr, col + dc
        temp = []
        while 0 <= r < board_size and 0 <= c < board_size:
            if board[r, c] == -color:
                temp.append((r, c))
            elif board[r, c] == color:
                if len(temp) > 0:
                    flips.extend(temp)
                break
            else:
                break
            r += dr
            c += dc
    return flips

@njit
def apply_move_numba(board, row, col, flips, color):
    new_board = board.copy()
    new_board[row, col] = color
    for i in range(len(flips)):
        r, c = flips[i]
        new_board[r, c] = COLOR_NONE
    return new_board

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.total_time_used = 0
        self.move_count = 0

    def generate_valid_moves(self, board, color):
        moves = []
        for row in range(self.chessboard_size):
            for col in range(self.chessboard_size):
                if board[row, col] == COLOR_NONE:
                    flips = get_flips_numba(board, row, col, color, self.chessboard_size)
                    if len(flips) > 0:
                        w = WEIGHT_MATRIX[row, col]
                        moves.append(((row, col), flips, w))
        return moves

    def apply_move(self, board, move, color):
        (row, col), flips, _ = move
        return apply_move_numba(board, row, col, np.array(flips), color)

    def minimax(self, board, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError

        valid_moves = self.generate_valid_moves(board, self.color if maximizing else -self.color)
        if depth == 0 or not valid_moves:
            return 0, None

        best_move = None
        if maximizing:
            value = -float('inf')
            for move in valid_moves:
                new_board = self.apply_move(board, move, -self.color)
                score, _ = self.minimax(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
                if move[2] > value:
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
                if move[2] > value:
                    value = move[2]
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move

    def iterative_deepening(self, board):
        start_time = time.time()
        time_limit = 4.8
        depth = 3
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
