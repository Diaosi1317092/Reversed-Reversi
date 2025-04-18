import numpy as np
import copy
import pygame
import main
import ver6
from pygame.locals import QUIT, MOUSEBUTTONDOWN

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

class ReversiGame:
    dr = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def legal(self, nw):
        return not (nw[0] < 0 or nw[0] >= 8 or nw[1] < 0 or nw[1] >= 8)

    def wk(self, chessboard, nw, color):
        chessboard[nw] = color
        for i in range(8):
            to = (nw[0] + self.dr[i][0], nw[1] + self.dr[i][1])
            path = []
            while self.legal(to) and chessboard[to] == -color:
                path.append(to)
                to = (to[0] + self.dr[i][0], to[1] + self.dr[i][1])
            if self.legal(to) and chessboard[to] == color:
                for pos in path:
                    chessboard[pos] = color
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 650))  # 调整窗口大小以容纳两个棋盘
        pygame.display.set_caption("Reversi Game - Dual Boards")
        self.font = pygame.font.SysFont(None, 24)

        # 初始化两个棋盘
        self.board_left = np.zeros((8, 8), dtype=int)
        self.board_right = np.zeros((8, 8), dtype=int)
        self.board_left[3, 3], self.board_left[4, 4] = COLOR_WHITE, COLOR_WHITE
        self.board_left[3, 4], self.board_left[4, 3] = COLOR_BLACK, COLOR_BLACK
        self.board_right[3, 3], self.board_right[4, 4] = COLOR_WHITE, COLOR_WHITE
        self.board_right[3, 4], self.board_right[4, 3] = COLOR_BLACK, COLOR_BLACK

        # 初始化两个 AI
        self.ai_black = ver6.AI(8, COLOR_BLACK, 5)  # 黑执黑
        self.ai_white = main.AI(8, COLOR_WHITE, 5)  # 白执白
        self.ai_black_as_white = ver6.AI(8, COLOR_WHITE, 5)  # 黑执白
        self.ai_white_as_black = main.AI(8, COLOR_BLACK, 5)  # 白执黑

        # 游戏状态
        self.running = True
        self.game_over = False
        self.winner = None

    def draw_board(self, board, offset_x, offset_y):
        """
        绘制单个棋盘和棋子。
        """
        cell_size = 50

        # 绘制棋盘
        for i in range(8):
            for j in range(8):
                x, y = offset_x + j * cell_size, offset_y + i * cell_size
                pygame.draw.rect(self.screen, (34, 139, 34), (x, y, cell_size, cell_size))  # 蓝色棋盘
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)  # 黑色边框
                if board[i, j] == COLOR_BLACK:
                    pygame.draw.circle(self.screen, (0, 0, 0), (x + cell_size // 2, y + cell_size // 2), cell_size // 2 - 5)
                elif board[i, j] == COLOR_WHITE:
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + cell_size // 2, y + cell_size // 2), cell_size // 2 - 5)

    def play_game(self):
        """
        自动执行游戏，两个棋盘同时进行。
        """
        step_left = 0  # 左侧棋盘的回合计数
        step_right = 0  # 右侧棋盘的回合计数

        while not self.game_over:
            # 左侧棋盘：黑执黑，白执白
            if step_left % 2 == 0:  # 黑棋行动
                current_ai_left = self.ai_black
                color_left = COLOR_BLACK
            else:  # 白棋行动
                current_ai_left = self.ai_white
                color_left = COLOR_WHITE

            current_ai_left.go(self.board_left)
            if len(current_ai_left.candidate_list) > 0:
                move = current_ai_left.candidate_list[-1]
                self.wk(self.board_left, move, color_left)
            step_left += 1

            # 右侧棋盘：黑执白，白执黑
            if step_right % 2 == 0:  # 黑棋行动（执白）
                current_ai_right = self.ai_black_as_white
                color_right = COLOR_WHITE
            else:  # 白棋行动（执黑）
                current_ai_right = self.ai_white_as_black
                color_right = COLOR_BLACK

            current_ai_right.go(self.board_right)
            if len(current_ai_right.candidate_list) > 0:
                move = current_ai_right.candidate_list[-1]
                self.wk(self.board_right, move, color_right)
            step_right += 1

            # 检查游戏是否结束
            if len(self.ai_black.candidate_list) == 0 and len(self.ai_white.candidate_list) == 0 and \
               len(self.ai_black_as_white.candidate_list) == 0 and len(self.ai_white_as_black.candidate_list) == 0:
                self.game_over = True

            # 更新界面
            self.screen.fill((200, 200, 200))  # 背景颜色
            self.draw_board(self.board_left, 50, 50)  # 左棋盘
            self.draw_board(self.board_right, 650, 50)  # 右棋盘
            pygame.display.flip()
            pygame.time.delay(500)  # 等待 500 毫秒

        # 计算结果
        self.calculate_winner()

    def calculate_winner(self):
        """
        计算最终结果并确定胜者。
        """
        black_count_left = np.sum(self.board_left == COLOR_BLACK)
        white_count_left = np.sum(self.board_left == COLOR_WHITE)
        black_count_right = np.sum(self.board_right == COLOR_BLACK)
        white_count_right = np.sum(self.board_right == COLOR_WHITE)

        # 子少者胜
        total_black = black_count_left + white_count_right
        total_white = white_count_left + black_count_right

        if total_black < total_white:
            self.winner = "Black Wins!"
        elif total_white < total_black:
            self.winner = "White Wins!"
        else:
            self.winner = "It's a Tie!"

    def display_winner(self):
        """
        在界面上显示胜者。
        """
        if self.winner:
            text = self.font.render(self.winner, True, (0, 0, 0))
            self.screen.blit(text, (600 - text.get_width() // 2, 600))  # 居中显示

    def run(self):
        """
        主循环，处理事件和更新界面。
        """
        self.play_game()

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            # 绘制界面
            self.screen.fill((200, 200, 200))  # 背景颜色
            self.draw_board(self.board_left, 50, 50)  # 左棋盘
            self.draw_board(self.board_right, 650, 50)  # 右棋盘
            self.display_winner()  # 显示胜者
            pygame.display.flip()

if __name__ == "__main__":
    game = ReversiGame()
    game.run()
    pygame.quit()