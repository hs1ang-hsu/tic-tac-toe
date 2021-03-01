import gym
from gym import spaces


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.SIZE = 5
        # 0代表未放置 1代表圈圈 -1代表叉叉
        self.board = [[0 for i in range(self.SIZE)] for i in range(self.SIZE)]

    def step(self, action):  # action包含(x座標, y座標, 圈圈或叉叉)
        # 更新棋盤
        self.board[action[0]][action[1]] = action[2]

        # 回饋參數
        win_reward = 10
        draw_reward = 0
        lose_reward = -5
        pos_reward = 1  # 自己得分的獎勵
        neg_reward = -1  # 敵方得分的懲罰

        # 計算雙方得分
        X_score = [0]*12
        O_score = [0]*12
        for y in range(5):
            for x in range(5):
                if(self.board[x][y] == -1):
                    X_score[x] += 1
                    X_score[y+5] += 1
                    if(x == y):
                        X_score[10] += 1
                    if(x == 4-y):
                        X_score[11] += 1
                elif(self.board[x][y] == 1):
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
        
        # 檢查是否已完成棋盤，並在此給予未完成盤面時的回饋值
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if(self.board[i][j] == 0):
                    if(action[2] == 1):
                        #return self.board, pos_reward*O+neg_reward*X, False, {}
                        return self.board, 0, False, {}
                    else:
                        #return self.board, pos_reward*X+neg_reward*O, False, {}
                        return self.board, 0, False, {}
                        
        # 在此給予完成盤面後的回饋值
        if(action[2] == 1):
            if(O > X):
                return self.board, win_reward, True, {}
            if(O == X):
                return self.board, draw_reward, True, {}
            else:
                return self.board, lose_reward, True, {}
        else:
            if(X > O):
                return self.board, win_reward, True, {}
            if(X == O):
                return self.board, draw_reward, True, {}
            else:
                return self.board, lose_reward, True, {}

    def reset(self):  # 初始化盤面
        self.board = [[0 for i in range(self.SIZE)] for i in range(self.SIZE)]
        return self.board

    def render(self, mode='human'):  # 可以將盤面可視化的程式區塊，但我們不需要可視化
        return
