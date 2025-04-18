import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

# don’t change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def get_flips(self, chessboard, row, col, color):
        flips = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            temp_flips = []
            r, c = row + dr, col + dc
            while 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size:
                if chessboard[r, c] == -color:
                    temp_flips.append((r, c))
                elif chessboard[r, c] == color:
                    if temp_flips:
                        flips.extend(temp_flips)
                    break
                else:
                    break
                r += dr
                c += dc
        return flips

    def go(self, chessboard):
        self.candidate_list.clear()
        
        valid_moves = []
        for row in range(self.chessboard_size):
            for col in range(self.chessboard_size):
                if chessboard[row, col] == COLOR_NONE:
                    flips = self.get_flips(chessboard, row, col, self.color)
                    if flips:
                        valid_moves.append(((row, col), len(flips)))
                        self.candidate_list.append((row, col))
        
        if not valid_moves:
            return []
        
        min_flip = min(valid_moves, key=lambda x: x[1])[1]
        best_moves = [move for move, count in valid_moves if count == min_flip]
        chosen_move = random.choice(best_moves)
        
        self.candidate_list.append(chosen_move)
        
        return self.candidate_list
