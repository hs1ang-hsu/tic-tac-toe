# 井井井井字遊戲

## 遊戲介紹
參照https://tioj.ck.tp.edu.tw/problems/2011

## 環境設定
執行setup.bat即可自動安裝環境所需套件。

## 程式說明
* main.py

主遊戲執行程式，遊戲相關資源（如：音檔及圖片）放置於source資料夾內。需要使用alpha-beta pruning的演算法時，會從AlphaBetaPruning資料夾中讀取相關python檔案。

* AlphaBetaPruning
  * minimax_tree.py
  
    實作minimax tree以及alpha-beta pruning演算法，並使用兩種方法進行加速。
    1. 將5x5的棋盤用二進位表示成一個50 bits以內的整數，演算法內所有操作將以位元運算代替。
    1. 使用雜湊函數紀錄已經搜尋過的盤面。
  
    執行程式輸入範例（已經過了5回合的局面）：
  ```
  X...X
  .X.O.
  ..OO.
  .O.O.
  O...X
  ```
  
  * minimax_tree.cpp
  
    minimax_tree.py的執行效率不佳，當盤面上只有8個記號時（也就是已經過了4回合了)，最差的情況大約需要跑30秒左右，因此我們嘗試以c++語言來改寫。
    改寫成c++語言後，即使是最差的情況，也都只要10秒內即可完成，但將c++語言轉為dll檔的過程不順利，最終未能成功讓main.py使用c++語言來協助加速。
  
    執行程式輸入範例（已經過了4回合的局面）：
  ```
  X...X
  .X.O.
  ..O..
  .O...
  O...X
  ```

  * data_generator.py
  
    輸入棋盤資料，自動輸出該棋盤對應的最佳策略，並儲存至json檔，以便後續機器學習使用。為了使生資料的速度提升，我們會將棋盤旋轉或鏡射得到更多棋盤。
  
    執行程式輸入範例（已經過了5回合的局面）：
  ```
  X...X
  .X.O.
  ..OO.
  .O.O.
  O...X
  ```

  * train.py

    將data_generator.py生成的資料從json檔讀出，並處理成適當的input(x), output(y)，進而當作data餵給deep neural network去做訓練。檔案裡面僅是某次調參的架構，我們只針對epoch,Dense layer個數, 每層layer的node數去做調整，並利用訓練後平均下錯幾手(0~2)做為調參的指標，最後再將表現最好的model存成.h5檔，供之後匯入使用。
  * temp.py 

    train.py主要用來調整參數，而temp.py則用來看測試單一個神經網路表現如何。
  
* gym-tictactoe-master

  * gym_tictactoe/envs/tictactoe_env.py
  
    描述遊戲的規則，以及建立reinforcement learning的回饋機制。
    在setup.bat中所執行的pip install -e gym-tictactoe-master，會將該環境建立至gym套件內。
  
  * training_agent.py
  
    Reinforcement learning的訓練程式，裡面實作了DQN演算法，並使用前述建立於gym套件內的環境進行訓練。
    一般的訓練方法是讓同一個model進行自我對弈，此外也可以讓兩個AI互相對奕，並紀錄下雙方的勝負次數。

