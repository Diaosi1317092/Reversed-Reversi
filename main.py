import numpy as np
import random
import time
import numba

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
], dtype=np.int64)

@numba.njit
def evaluate_board_numb(board, ai_color):
    score = 0
    rows, cols = board.shape
    son, lcor, rcor = 0, 0, 0
    lang = 0
    rang = 0
    ledg = 0
    redg = 0
    for i in range(rows):
        for j in range(cols):
            val = WEIGHT_MATRIX[i, j]
            if board[i, j] == ai_color:
                score += val
                son -= 1
                if val == 1:
                    lcor += 1
                elif val == 8:
                    lang += 1
                elif (i and not j) or (not i and j):
                    ledg += 1
            elif board[i, j] == -ai_color:
                score -= val
                son += 1
                if val == 1:
                    rcor += 1
                elif val == 8:
                    rang += 1
                elif (i and not j) or (not i and j):
                    redg += 1
    return 5*son + 10*lang - 20*rang - 1000*lcor + 20*rcor + ledg - 2*redg

@numba.njit
def get_flips_numb(board, row, col, color, board_size):
    flips = np.empty((56, 2), dtype=np.int64)
    count = 0
    directions = np.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
    for d in range(8):
        dr = directions[d, 0]
        dc = directions[d, 1]
        temp_count = 0
        i = 1
        while True:
            r = row + dr * i
            c = col + dc * i
            if r < 0 or r >= board_size or c < 0 or c >= board_size:
                break
            if board[r, c] == -color:
                temp_count += 1
            elif board[r, c] == color:
                if temp_count > 0:
                    for k in range(temp_count):
                        flips[count, 0] = row + dr * (i - temp_count + k)
                        flips[count, 1] = col + dc * (i - temp_count + k)
                        count += 1
                break
            else:
                break
            i += 1
    return flips[:count]

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.total_time_used = 0
        self.move_count = 0
        self.count = 0
        self.round = 0

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
    def update(self, color, row, col, eval, op):
        new_eval = eval
        val = WEIGHT_MATRIX[row, col]
        flg = (color == self.color)
        if op == 0:
            new_eval += -5 if flg else 5                    # son -5 +5
            if val == 1:                                    # cor -1000 +20
                new_eval += -1000 if flg else 20
            elif val == 8:                                  # ang +10 -20
                new_eval += 10 if flg else -20
            elif (row and not col) or (not row and col):    # edg +1 -2
                new_eval += 1 if flg else -2
        else:
            assert val != 1
            new_eval += -10 if flg else 10                  # son -5 +5
            if val == 8:                                    # ang +10 -20
                new_eval += 30 if flg else -30
            elif (row and not col) or (not row and col):    # edg +1 -2
                new_eval += 3 if flg else -3
        
        return new_eval
    
    def apply_move(self, board, move, color, eval):
        new_board = board.copy()
        new_eval = eval
        (row, col), flips, _ = move
        new_board[row, col] = color
        
        new_eval = self.update(color, row, col, new_eval, 0)
        
        for r, c in flips:
            new_board[r, c] = -new_board[r, c]
            new_eval = self.update(new_board[r, c], r, c, new_eval, 1)
        
        return new_board, new_eval

    def evaluate_board(self, board):
        return evaluate_board_numb(board, self.color)

    def minimax(self, board, depth, alpha, beta, maximizing, start_time, time_limit, eval):
        self.count += 1
        if self.count % 16 == 0:
            if time.time() - start_time > time_limit:
                raise TimeoutError
        if depth == 0:
            return eval, None
        
        valid_moves = self.generate_valid_moves(board, self.color if maximizing else -self.color)
        # 对 valid_moves 按照权重排序：己方回合降序排序，保证先搜索权重高的走步；对手回合升序排序。
        if valid_moves:
            if maximizing:
                valid_moves.sort(key=lambda move: move[2], reverse=True)
            else:
                valid_moves.sort(key=lambda move: move[2])
        
        best_move = None
        if maximizing:
            value = -float('inf')
            if not valid_moves:
                score, _ = self.minimax(board, depth - 1, alpha, beta, False, start_time, time_limit, eval)
                if score > value:
                    value = score
                alpha = max(alpha, value)
            for move in valid_moves:
                new_board, new_eval = self.apply_move(board, move, self.color, eval)
                score, _ = self.minimax(new_board, depth - 1, alpha, beta, False, start_time, time_limit, new_eval)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move
        else:
            value = float('inf')
            if not valid_moves:
                score, _ = self.minimax(board, depth - 1, alpha, beta, True, start_time, time_limit, eval)
                if score < value:
                    value = score
                beta = min(beta, value)
            for move in valid_moves:
                new_board, new_eval = self.apply_move(board, move, -self.color, eval)
                score, _ = self.minimax(new_board, depth - 1, alpha, beta, True, start_time, time_limit, new_eval)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def iterative_deepening(self, board):
        start_time = time.time()
        time_limit = 4.8
        depth = 5 if self.round <= 24 else 7
        best_move = None
        while True:
            eval = self.evaluate_board(board)
            try:
                self.count = 0
                score, move = self.minimax(board, depth, -float('inf'), float('inf'), True, start_time, time_limit, eval)
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
        self.round += 1
        valid_moves = self.generate_valid_moves(chessboard, self.color)
        valid_moves.sort(key=lambda move: move[2])
        for move in valid_moves:
            self.candidate_list.append(move[0])
        best_move = self.iterative_deepening(chessboard)
        if best_move is None:
            return []
        chosen_pos = best_move[0]
        self.candidate_list.append(chosen_pos)
        self.total_time_used += time.time() - start_time
        self.move_count += 1
        return self.candidate_list
