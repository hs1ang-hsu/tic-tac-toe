# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import pygame
import AlphaBetaPruning.minimax_tree as ABP
pygame.init()  # 初始化
pygame.mixer.init()  # 初始化混音器


class Board:  # 棋盤
    def __init__(self):
        self.grid_lines = [[(150, 50), (150, 550)],  # vertical line
                           [(250, 50), (250, 550)],
                           [(350, 50), (350, 550)],
                           [(450, 50), (450, 550)],
                           [(50, 150), (550, 150)],  # horizontal line
                           [(50, 250), (550, 250)],
                           [(50, 350), (550, 350)],
                           [(50, 450), (550, 450)]]
        self.board_state = [[0 for x in range(5)] for y in range(5)]  # 當前盤面
        self.game_round = 0  # 第12輪結束
        self.action_count = 0  # 計算同一個人下了幾次

    def initialize(self):
        self.board_state = [[0 for x in range(5)] for y in range(5)]
        self.game_round = 0
        self.action_count = 0

    def draw_board(self, win):  # 把盤面印出來
        pygame.draw.rect(win, board_color, [50, 50, 500, 500], 0)
        for line in self.grid_lines:
            pygame.draw.line(win, grid_color, line[0], line[1])
        for y in range(5):
            for x in range(5):
                if(self.get_board_value(x, y) == 'X'):
                    win.blit(X_img, (x*100 + 50, y*100 + 50))
                elif(self.get_board_value(x, y) == 'O'):
                    win.blit(O_img, (x*100 + 50, y*100 + 50))

        pygame.draw.rect(win, scoreboard_color, [575, 100, 200, 150], 0)
        pygame.draw.rect(win, scoreboard_color, [575, 300, 200, 150], 0)
        self.print_score(win)

    def get_score(self):
        # 0~4存直行符號數 5~9存橫列符號數 10和11存對角線符號數
        X_score = [0]*12
        O_score = [0]*12
        for y in range(5):
            for x in range(5):
                if(self.get_board_value(x, y) == 'X'):
                    X_score[x] += 1
                    X_score[y+5] += 1
                    if(x == y):
                        X_score[10] += 1
                    if(x == 4-y):
                        X_score[11] += 1
                elif(self.get_board_value(x, y) == 'O'):
                    O_score[x] += 1
                    O_score[y+5] += 1
                    if(x == y):
                        O_score[10] += 1
                    if(x == 4-y):
                        O_score[11] += 1
        X = 0
        O = 0
        for i in range(12):
            if(X_score[i] >= 4):
                X += 1
            elif(O_score[i] >= 4):
                O += 1
        return O, X

    def print_score(self, win):
        O, X = self.get_score()
        display_text('O score', 675, 137, 40, win)
        display_text(str(O), 675, 212, 40, win)
        display_text('X score', 675, 337, 40, win)
        display_text(str(X), 675, 412, 40, win)

    def get_board_value(self, x, y):
        return self.board_state[y][x]

    def set_board_value(self, x, y, value):
        self.board_state[y][x] = value

    def action(self, x, y, player):  # 若有成功執行動作就回傳True，反之回傳False
        if((50 <= x < 550) and (50 <= y < 550)):
            x = (x-50) // 100
            y = (y-50) // 100
            if(self.get_board_value(x, y) == 0):
                chess.play()
                self.set_board_value(x, y, player.sign)
                return True
        return False

    def print_board(self):  # for debugging
        for x in self.board_state:
            print(x)

    # 把棋盤變成1*25的陣列，用來丟進神經網路model內
    def get_board_array(self):
        board_arr = []
        for y in range(5):
            for x in range(5):
                if(self.get_board_value(x, y) == 'X'):
                    board_arr.append(-1)
                elif(self.get_board_value(x, y) == 'O'):
                    board_arr.append(1)
                else:
                    board_arr.append(0)
        return board_arr

    def fill_with_cross(self):  # 把最後三格快速填入X
        for y in range(5):
            for x in range(5):
                if(self.get_board_value(x, y) == 0):
                    self.set_board_value(x, y, 'X')

    def end_game(self, win):  # 遊戲結算
        pygame.draw.rect(win, end_game_color, [250, 200, 300, 200], 0)
        O, X = self.get_score()
        if(X > O):
            display_text('X wins', 400, 300, 80, win)
            display_text('Again?', 400, 365, 40, win)
        elif(X < O):
            display_text('O wins', 400, 300, 80, win)
            display_text('Again?', 400, 365, 40, win)
        elif(X == O):
            display_text('ties', 400, 300, 80, win)
            display_text('Again?', 400, 365, 40, win)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(250 <= pos[0] <= 550 and 200 <= pos[1] <= 400):
                        page(next_scene)
            if event.type == pygame.QUIT:
                pygame.quit()

    def smallmenu(self, win):
        display_text('ties', 400, 300, 80, win)


class Player:
    def __init__(self, sign):
        self.sign = sign


class AI:
    def __init__(self, sign):
        self.sign = sign
        self.model = self.build_net()

        if(self.sign == 'O'):
            self.label = 1
        elif(self.sign == 'X'):
            self.label = -1

        self.model.load_weights('source\\AI_model\\tictactoeAI.h5')

    def build_net(self):
        # 兩層隱藏層，都為32層，因此神經網路為26>32>32>25
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(26,)))
        #model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(25, activation='relu'))
        adam = optimizers.Adam(lr=0.0001)
        model.compile(loss='mse', optimizer=adam)
        return model

    def get_action(self, state):
        tmp_state = state
        tmp_state.append(self.label)

        # 用神經網路預測每個位置的報酬
        q_value = self.model.predict(np.array([tmp_state]))[0]
        # 設定最小值為最小的q_value-1
        min_v = q_value[np.argmin(q_value)]-1
        for i in range(25):  # 檢查那些位置不可下棋，並避免下在這些位置
            if(tmp_state[i] != 0):
                q_value[i] = min_v
        action = np.argmax(q_value)  # 選擇報酬最高的位置
        action = [action % 5, action // 5]
        return action

class AI_ABP: # alpha-beta pruning
    def __init__(self, sign):
        self.sign = sign

    def get_action(self, state):
        action = ABP.get_next_step(state)
        return action[:]

######################################################
#                       變數                         #
######################################################

win = pygame.display.set_mode((800, 600))  # 主視窗
pygame.display.set_caption('井字遊戲')  # 視窗名稱
board = Board()  # 棋盤
next_scene = 0  # 下個場景
sign = 'O'  # AI是先手或後手
mode = 'RL'  # RL或ABP

# 色碼
board_color = (241, 130, 27)  # 棋盤顏色
grid_color = (60, 83, 127)  # 格線顏色
scoreboard_color = (255, 255, 255)  # 計分板底色
play_turn_color = (241, 130, 27)  # 標示出輪到誰了的顏色
end_game_color = (241, 84, 124)  # 結束遊戲提示的底色
menu_background_color = (60, 83, 127)  # 目錄背景顏色
menu_button_color = (241, 130, 27)  # 目錄選項顏色
main_background_color = (60, 83, 127)  # 遊戲背景顏色


# 字體
def display_text(string, x, y, size, win):
    font = pygame.font.Font("source\\font\\msjhbd.ttc", size)
    text = font.render(string, True, (60, 83, 127))
    text_rect = text.get_rect(center=(x, y))
    win.blit(text, text_rect)

# 圖片(img資料夾)
X_img = pygame.image.load("source\\img\\cross.png")
X_img.convert()
O_img = pygame.image.load("source\\img\\circle.png")
O_img.convert()

# 音效
click = pygame.mixer.Sound('source\\music\\click2.wav')
chess = pygame.mixer.Sound('source\\music\\chess.wav')

######################################################
#                     遊戲本體                       #
######################################################


# 目錄
def menu():
    global next_scene
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(250 <= pos[0] <= 550 and 125 <= pos[1] <= 275):
                        next_scene = 1
                        page(next_scene)
                        running = False
                    if(250 <= pos[0] <= 550 and 325 <= pos[1] <= 475):
                        next_scene = 4
                        page(next_scene)
                        running = False
                    if(300 <= pos[0] <= 500 and 525 <= pos[1] <= 575):
                        next_scene = 6
                        page(next_scene)
                        running = False
        win.fill(menu_background_color)
        pygame.draw.rect(win, menu_button_color, [250, 125, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [250, 325, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [300, 525, 200, 50], 0)
        display_text('單人遊戲', 400, 200, 60, win)
        display_text('雙人遊戲', 400, 400, 60, win)
        display_text('規則', 400, 550, 40, win)
        pygame.display.update()


def smallmenu():
    global next_scene
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(250 <= pos[0] <= 550 and 125 <= pos[1] <= 275):
                        next_scene = next_scene
                        page(next_scene)
                        running = False
                    if(250 <= pos[0] <= 550 and 325 <= pos[1] <= 475):
                        next_scene = 0
                        page(next_scene)
                        running = False

        win.fill(menu_background_color)
        pygame.draw.rect(win, menu_button_color, [250, 125, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [250, 325, 300, 150], 0)
        display_text('繼續遊戲', 400, 200, 60, win)
        display_text('返回目錄', 400, 400, 60, win)
        pygame.display.update()


# 兩人對戰
def main_2P():
    global board
    player1 = Player('O')
    player2 = Player('X')

    running = True
    end = False
    while running:
        win.fill(main_background_color)  # 先上背景色

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:  # 判斷按下按鈕
                if event.key == pygame.K_ESCAPE:  # 判斷按下ESC按鈕
                    next_scene = 5
                    page(next_scene)
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    if(end):  # 當遊戲結束
                        running = False
                    pos = pygame.mouse.get_pos()

                    # 先手的人
                    if(board.game_round != 11 and board.game_round % 2 == 0):
                        if(board.action(pos[0], pos[1], player1)):  # 若成功執行動作
                            board.action_count += 1
                            if(board.action_count == 2):
                                board.game_round += 1
                                board.action_count = 0
                    # 後手的人
                    elif(board.game_round != 11 and board.game_round % 2 == 1):
                        if(board.action(pos[0], pos[1], player2)):  # 若成功執行動作
                            board.action_count += 1
                            if(board.action_count == 2):
                                board.game_round += 1
                                board.action_count = 0
                    # 最後一輪
                    elif(board.game_round == 11):
                        if(board.action(pos[0], pos[1], player2)):  # 若成功執行動作
                            board.action_count += 1
                            if(board.action_count == 3):  # 結束遊戲
                                end = True
        if(board.game_round % 2 == 0):
            pygame.draw.rect(win, play_turn_color, [575, 100, 200, 150], 15)
        else:
            pygame.draw.rect(win, play_turn_color, [575, 300, 200, 150], 15)
        board.draw_board(win)
        if(end):
            board.end_game(win)
        pygame.display.flip()


# 與AI對戰
def main_1P():
    global board
    if(sign == 'X'):  # 若玩家為後手
        player1 = Player(sign)
        player2 = AI('O')

        running = True
        end = False
        while running:
            win.fill(main_background_color)  # 先上背景色

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:  # 判斷按下按鈕
                    if event.key == pygame.K_ESCAPE:  # 判斷按下ESC按鈕
                        next_scene = 5
                        page(next_scene)
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                        if(end):  # 當遊戲結束
                            running = False

                        pos = pygame.mouse.get_pos()
                        # 玩家後手才能下棋
                        if(board.game_round != 11 and board.game_round % 2 == 1):
                                # 若成功執行動作
                                if(board.action(pos[0], pos[1], player1)):
                                    board.action_count += 1
                                    if(board.action_count == 2):
                                        board.game_round += 1
                                        board.action_count = 0
                        # 最後一輪
                        elif(board.game_round == 11):
                            # 若成功執行動作
                            if(board.action(pos[0], pos[1], player1)):
                                board.action_count += 1
                                if(board.action_count == 3):  # 結束遊戲
                                    end = True
            if(board.game_round != 11 and board.game_round % 2 == 0):  # AI先手下棋
                act = player2.get_action(board.get_board_array())  # 取得下棋位置
                board.set_board_value(act[0], act[1], player2.sign)  # 執行動作
                act = player2.get_action(board.get_board_array())  # 重複一次
                board.set_board_value(act[0], act[1], player2.sign)
                board.game_round += 1
                board.action_count = 0

            if(board.game_round % 2 == 0):
                pygame.draw.rect(win, play_turn_color, [575, 100, 200, 150], 15)
            else:
                pygame.draw.rect(win, play_turn_color, [575, 300, 200, 150], 15)
            board.draw_board(win)
            if(end):
                board.end_game(win)
            pygame.display.flip()
    else:
        player1 = Player(sign)
        player2 = AI('X')

        running = True
        end = False
        while running:
            win.fill(main_background_color)  # 先上背景色

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:  # 判斷按下按鈕
                    if event.key == pygame.K_ESCAPE:  # 判斷按下ESC按鈕
                        next_scene = 5
                        page(next_scene)
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                        if(end):  # 當遊戲結束
                            running = False

                        pos = pygame.mouse.get_pos()
                        # 玩家先手下棋
                        if(board.game_round != 11 and board.game_round % 2 == 0):
                            # 若成功執行動作
                            if(board.action(pos[0], pos[1], player1)):
                                board.action_count += 1
                                if(board.action_count == 2):
                                    board.game_round += 1
                                    board.action_count = 0
            if(board.game_round != 11 and board.game_round % 2 == 1):  # AI後手下棋
                act = player2.get_action(board.get_board_array())  # 取得下棋位置
                board.set_board_value(act[0], act[1], player2.sign)  # 執行動作
                act = player2.get_action(board.get_board_array())  # 重複一次
                board.set_board_value(act[0], act[1], player2.sign)
                board.game_round += 1
                board.action_count = 0
            elif(board.game_round == 11):  # 最後一輪
                board.fill_with_cross()
                end = True

            if(board.game_round % 2 == 0):
                pygame.draw.rect(win, play_turn_color, [575, 100, 200, 150], 15)
            else:
                pygame.draw.rect(win, play_turn_color, [575, 300, 200, 150], 15)
            board.draw_board(win)
            if(end):
                board.end_game(win)
            pygame.display.flip()

def main_1P_ABP():
    global board
    if(sign == 'X'):  # 若玩家為後手
        player1 = Player(sign)
        player2 = AI_ABP('O')

        running = True
        end = False
        while running:
            win.fill(main_background_color)  # 先上背景色

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:  # 判斷按下按鈕
                    if event.key == pygame.K_ESCAPE:  # 判斷按下ESC按鈕
                        next_scene = 5
                        page(next_scene)
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                        if(end):  # 當遊戲結束
                            running = False

                        pos = pygame.mouse.get_pos()
                        # 玩家後手才能下棋
                        if(board.game_round != 11 and board.game_round % 2 == 1):
                                # 若成功執行動作
                                if(board.action(pos[0], pos[1], player1)):
                                    board.action_count += 1
                                    if(board.action_count == 2):
                                        board.game_round += 1
                                        board.action_count = 0
                        # 最後一輪
                        elif(board.game_round == 11):
                            # 若成功執行動作
                            if(board.action(pos[0], pos[1], player1)):
                                board.action_count += 1
                                if(board.action_count == 3):  # 結束遊戲
                                    end = True
            if(board.game_round != 11 and board.game_round % 2 == 0):  # AI先手下棋
                act = player2.get_action(board.board_state)  # 取得下棋位置
                board.set_board_value(act[0] % 5, act[0] // 5, player2.sign)  # 執行動作
                board.set_board_value(act[1] % 5, act[1] // 5, player2.sign)
                board.game_round += 1
                board.action_count = 0
            
            if(board.game_round % 2 == 0):
                pygame.draw.rect(win, play_turn_color, [575, 100, 200, 150], 15)
            else:
                pygame.draw.rect(win, play_turn_color, [575, 300, 200, 150], 15)
            board.draw_board(win)
            if(end):
                board.end_game(win)
            pygame.display.flip()
    else:
        player1 = Player(sign)
        player2 = AI_ABP('X')

        running = True
        end = False
        while running:
            win.fill(main_background_color)  # 先上背景色

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:  # 判斷按下按鈕
                    if event.key == pygame.K_ESCAPE:  # 判斷按下ESC按鈕
                        next_scene = 5
                        page(next_scene)
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                        if(end):  # 當遊戲結束
                            running = False

                        pos = pygame.mouse.get_pos()
                        # 玩家先手下棋
                        if(board.game_round != 11 and board.game_round % 2 == 0):
                            # 若成功執行動作
                            if(board.action(pos[0], pos[1], player1)):
                                board.action_count += 1
                                if(board.action_count == 2):
                                    board.game_round += 1
                                    board.action_count = 0
            if(board.game_round != 11 and board.game_round % 2 == 1):  # AI後手下棋
                act = player2.get_action(board.board_state)  # 取得下棋位置
                board.set_board_value(act[0] % 5, act[0] // 5, player2.sign)  # 執行動作
                board.set_board_value(act[1] % 5, act[1] // 5, player2.sign)
                board.game_round += 1
                board.action_count = 0
            elif(board.game_round == 11):  # 最後一輪
                board.fill_with_cross()
                end = True
            
            if(board.game_round % 2 == 0):
                pygame.draw.rect(win, play_turn_color, [575, 100, 200, 150], 15)
            else:
                pygame.draw.rect(win, play_turn_color, [575, 300, 200, 150], 15)
            board.draw_board(win)
            if(end):
                board.end_game(win)
            pygame.display.flip()

def mode_select():
    global next_scene, mode
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(250 <= pos[0] <= 550 and 125 <= pos[1] <= 275):
                        next_scene = 2
                        mode = 'RL'
                        page(next_scene)
                        running = False
                    if(250 <= pos[0] <= 550 and 325 <= pos[1] <= 475):
                        next_scene = 2
                        mode = 'ABP'
                        page(next_scene)
                        running = False
                    if(10 <= pos[0] <= 130 and 464 <= pos[1] <= 544):
                        next_scene = 0
                        page(next_scene)
                        running = False
        win.fill(menu_background_color)
        pygame.draw.rect(win, menu_button_color, [250, 125, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [250, 325, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [10, 464, 120, 80], 0)
        display_text('簡單', 400, 200, 60, win)
        display_text('困難', 400, 400, 60, win)
        display_text('返回', 70, 500, 40, win)
        pygame.display.update()


def order():
    global next_scene, sign
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(250 <= pos[0] <= 550 and 125 <= pos[1] <= 275):
                        next_scene = 3
                        sign = 'O'
                        page(next_scene)
                        running = False
                    if(250 <= pos[0] <= 550 and 325 <= pos[1] <= 475):
                        next_scene = 3
                        sign = 'X'
                        page(next_scene)
                        running = False
                    if(10 <= pos[0] <= 130 and 464 <= pos[1] <= 544):
                        next_scene = 1
                        page(next_scene)
                        running = False
        win.fill(menu_background_color)
        pygame.draw.rect(win, menu_button_color, [250, 125, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [250, 325, 300, 150], 0)
        pygame.draw.rect(win, menu_button_color, [10, 464, 120, 80], 0)
        display_text('先手', 400, 200, 60, win)
        display_text('後手', 400, 400, 60, win)
        display_text('返回', 70, 500, 40, win)
        pygame.display.update()


def rule():
    win.fill(menu_background_color)
    rule_img = pygame.image.load("source\\img\\rule.png")
    rule_img.convert()
    win.blit(rule_img, [125, 100])
    global next_scene
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # 當按下左鍵
                    pos = pygame.mouse.get_pos()
                    if(10 <= pos[0] <= 130 and 514 <= pos[1] <= 594):
                        next_scene = 0
                        page(next_scene)
                        running = False
        pygame.draw.rect(win, menu_button_color, [10, 514, 120, 80], 0)
        display_text('返回', 70, 550, 40, win)
        pygame.display.update()

scene_record = [0]  # 用來記錄場景切換


def page(next_scene):  # 處理場景切換的函數
    global board
    scene_record.append(next_scene)
    operating = True
    while operating:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                operating = False
                pygame.quit()
                break
            else:
                if (next_scene == 0):
                    click.play()
                    menu()
                elif(next_scene == 1):
                    click.play()
                    mode_select()
                elif(next_scene == 2):
                    click.play()
                    order()
                elif(next_scene == 3):
                    if (scene_record[-2] == 5):
                        click.play()
                    else:
                        click.play()
                        board.initialize()
                    if mode == 'RL':
                        main_1P()
                    else:
                        main_1P_ABP()
                elif(next_scene == 4):
                    if (scene_record[-2] == 5):
                        click.play()
                        main_2P()
                    else:
                        click.play()
                        board.initialize()
                        main_2P()
                elif(next_scene == 5):
                    click.play()
                    smallmenu()
                elif(next_scene == 6):
                    click.play()
                    rule()
                return page(next_scene)

    return pygame.quit()

# 執行場景
page(next_scene)
pygame.quit()
