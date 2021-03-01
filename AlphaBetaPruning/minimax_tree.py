import numpy as np
next_board = 0
next_board_even = 0
begin_depth = 0
board_data = {}

#####################################
#     Coordinate of the board       #
#                                   #
# board[y][x] with coordinate (x,y) #
#   (0,0)|(1,0)|(2,0)|(3,0)|(4,0)   #
#   (0,1)|(1,1)|(2,1)|(3,1)|(4,1)   #
#   (0,2)|(1,2)|(2,2)|(3,2)|(4,2)   #
#   (0,3)|(1,3)|(2,3)|(3,3)|(4,3)   #
#   (0,4)|(1,4)|(2,4)|(3,4)|(4,4)   #
#                                   #
#####################################

def dict_to_json(o, level=0):
    INDENT = 3
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k,v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += dict_to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([dict_to_json(e, level+1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    
    return ret

#input: board[5][5], which is a 5x5 character matrix with 'O', 'X', or '.'.
#output: A 50 bits number
#definition: '01' stands for 'X' and '10' stands for 'O'
def set_board_to_ll(board):
    a = 0
    for y in range(5):
        for x in range(5):
            if board[y][x] == 'X':
                a += 1 << ((5*y+x)*2)
            elif board[y][x] == 'O':
                a += 1 << ((5*y+x)*2+1)
    return a

#input: board, a 50 bits number. x and y is the position on the board
#output: The sign of the position with 'O', 'X', or '.'.
def get_board_value(board, x, y):
    pos = (5*y+x)*2
    if board & (1<<pos):
        return 'X'
    elif board & (1<<(pos+1)):
        return 'O'
    else:
        return '.'

#input: b1 and b2 are two 50 bits numbers
#output: The different in 2 board, which is the move position
def compare_board(b1, b2):
    c = b1^b2
    move = []
    for i in range(50):
        if c&1:
            move.append(i//2)
        c >>= 1
    return move

#swap two position in the board
#return the result board
def bitwise_swap(board, p1, p2):
    n1 = p1 << 1
    n2 = p2 << 1
    
    b1 = (board>>n1)&3
    b2 = (board>>n2)&3
    
    x = b1^b2
    x = (x<<n1) | (x<<n2)
    
    return board^x

#retate the board for 90 degrees clockwisely
#return the result board
def rotate_board(board, move):
    #0>4>24>20 1>9>23>15
    for i in range(2):
        for j in range(i,4-i):
            a = i
            b = j
            for k in range(3):
                board = bitwise_swap(board, j+5*i, 4-a+5*b)
                c = a
                a = (4-c+5*b)//5
                b = (4-c+5*b)%5
            
    move[0] = 4-(move[0]//5)+5*(move[0]%5)
    move[1] = 4-(move[1]//5)+5*(move[1]%5)
    return board

#flip the board form left to right
#return the result board
def flip_board(board, move):
    #0>4 1>3 2>2 3>1 4>0
    for i in range(5):
        for j in range(2):
            board = bitwise_swap(board, j+5*i, 4-j+5*i)
    move[0] = 4-(move[0]%5)+5*(move[0]//5)
    move[1] = 4-(move[1]%5)+5*(move[1]//5)
    return board

#generate different symmetric board with origin board
#return a dict with the form 'board: move'
#board, a 50 bits number. move, the desired move list [a,b]
def generate_board(board, move):
    result_board = {}
    for i in range(2):
        for j in range(4):
            result_board[board] = move[:]
            board = rotate_board(board, move)
        board = flip_board(board, move)
    return result_board

def write_map(file_name, board, move):
    import json
    import os
    global board_data
    
    if os.path.isfile(file_name):
        with open(file_name, 'r', newline='') as jsfile:
            board_data = json.load(jsfile)
    else:
        with open(file_name, 'w', newline='') as jsfile:
            json.dump({}, jsfile)
    
    new_data = generate_board(board, move)
    board_data = {**board_data, **new_data}
    
    with open(file_name, 'w', encoding='utf-8') as jsfile:
        ret = dict_to_json(board_data, level=0)
        jsfile.write(ret)

#input: board, a 50 bits number.
#output: Return 1 if O wins, -1 if X wins, and 0 if game is even.
def get_result(board):
    # col:0~4, row:5~9, diag:10,11
    X_score = [0]*12
    O_score = [0]*12
    for y in range(5):
        for x in range(5):
            if(get_board_value(board, x, y) == 'X'):
                X_score[x] += 1
                X_score[y+5] += 1
                if(x == y):
                    X_score[10] += 1
                if(x == 4-y):
                    X_score[11] += 1
            elif(get_board_value(board, x, y) == 'O'):
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
    
    if X>O:
        return -1
    elif O>X:
        return 1
    else:
        return 0

#input: board, a 50 bits number.
#output: None. Print out board with 'O', 'X', and '.'.
def print_board(board):
    for y in range(5):
        for x in range(5):
            if board&1:
                sign = 'X'
            elif board&2:
                sign = 'O'
            else:
                sign = '.'
            board >>= 2
            print(sign, end='')
            if x==4:
                print('')

#minimax tree with alpha-beta pruning.
def minimax(board, depth, alpha, beta, maximizingPlayer):
    global next_board
    global next_board_even
    global begin_depth
    global board_data
    
    if depth==0:
        return get_result(board)
    elif depth==1:
        tmp = board
        for y in range(5):
            for x in range(5):
                if (not tmp&1) and (not tmp&2):
                    board += 1 << ((5*y+x)*2)
                tmp >>= 2
        return get_result(board)
    
    child = []
    tmp = board
    for y in range(5):
        for x in range(5):
            if (not tmp&1) and (not tmp&2):
                child.append(5*y+x)
            tmp >>= 2
    
    if maximizingPlayer: # O's term
        max_eval = -10
        eval = 0
        tmp = 0
        is_even = False
        for i in range(depth*2+1):
            for j in range(i+1, depth*2+1):
                tmp = board + (1 << (child[i]*2+1))
                tmp += 1 << (child[j]*2+1)
                
                eval = minimax(tmp, depth-1, alpha, beta, False)
                if depth == begin_depth:
                    if eval==0 and (not is_even):
                        next_board_even = tmp
                        is_even = True
                next_board = tmp
                
                if eval == 1:
                    return eval
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    if max_eval == 0:
                        next_board = next_board_even
                    return max_eval
        if max_eval == 0:
            next_board = next_board_even
        return max_eval
    else: # X's term
        min_eval = 10
        eval = 0
        tmp = 0
        is_even = False
        for i in range(depth*2+1):
            for j in range(i+1, depth*2+1):
                tmp = board + (1 << (child[i]*2))
                tmp += 1 << (child[j]*2)
                
                eval = minimax(tmp, depth-1, alpha, beta, True)
                if depth == begin_depth:
                    if eval==0 and not is_even:
                        next_board_even = tmp
                        is_even = True
                next_board = tmp
                
                if eval == -1:
                    return eval
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    if min_eval == 0:
                        next_board = next_board_even
                    return min_eval
        if min_eval == 0:
            next_board = next_board_even
        return min_eval

def choice_by_sign_defence(board, score): # 'score' is either 'O_score' or 'X_score'.
    # We use the case of O_score to explain the code.
    import random

    rough_choice_2 = []
    rough_choice_3 = []
    choice = []
    for i in range(12): # For the case that O has 2 or 3 signs in a line
        if score[i] == 2:
            rough_choice_2.append(i)
        elif score[i] == 3:
            rough_choice_3.append(i)
    is_finished = False
    for i in rough_choice_3:
        tmp = []
        if i < 5: # col
            for j in range(5):
                if get_board_value(board, i, j) == '.':
                    tmp.append(5*j+i)

        elif i < 10: # row
            for j in range(5):
                if get_board_value(board, j, i-5) == '.':
                    tmp.append(5*(i-5)+j)
            
        elif i == 10: #diag1
            for j in range(5):
                if get_board_value(board, j, j) == '.':
                    tmp.append(6*j)
            
        else: #diag2
            for j in range(5):
                if get_board_value(board, 4-j, j) == '.':
                    tmp.append(4*j+4)
        
        if len(tmp) == 1: # O has 3 signs and X has 1 signs
            choice.append(tmp[0])
            board += 1<<(tmp[0]*2)
            if len(choice) == 2:
                is_finished = True
        elif len(tmp) == 2: # O has 3 signs and X has no signs
            choice = tmp[:]
            is_finished = True
        
        if is_finished:
            break
    
    if not is_finished: # if not done
        for i in rough_choice_2:
            tmp = []
            if i < 5: # col
                for j in range(5):
                    if get_board_value(board, i, j) == '.':
                        tmp.append(5*j+i)

            elif i < 10: # row
                for j in range(5):
                    if get_board_value(board, j, i-5) == '.':
                        tmp.append(5*(i-5)+j)
                
            elif i == 10: #diag1
                for j in range(5):
                    if get_board_value(board, j, j) == '.':
                        tmp.append(6*j)
                
            else: #diag2
                for j in range(5):
                    if get_board_value(board, 4-j, j) == '.':
                        tmp.append(4*j+4)
            
            if len(tmp) == 2: # O has 2 signs and X has 1 signs
                a = tmp[random.randint(0,1)]
                choice.append(a)
                board += 1<<(a*2)
                if len(choice) == 2:
                    is_finished = True
            elif len(tmp) == 3: # O has 2 signs and X has no signs
                choice = random.sample(set(tmp), 2)
                is_finished = True
            
            if is_finished:
                break
    
    pick1 = [0, 4, 12, 20, 24]
    pick2 = [6, 8, 16, 18]
    pick3 = [7, 11, 13, 17]
    if len(choice) == 2:
        return choice[:]
    elif len(choice) == 1:
        tmp = []
        for pos in pick1:
            if get_board_value(board, pos%5, pos//5) == '.':
                tmp.append(pos)
        if not tmp:
            for pos in pick2:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        choice.append(tmp[random.randint(0,len(tmp)-1)])
        print(choice)
        return choice[:]
    else:
        tmp = []
        for pos in pick1:
            if get_board_value(board, pos%5, pos//5) == '.':
                tmp.append(pos)
        if len(tmp) < 2:
            for pos in pick2:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        if len(tmp) < 2:
            for pos in pick3:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        
        choice = random.sample(set(tmp), 2)
        return choice[:]

def choice_by_sign_attack(board, score): # 'score' is either 'O_score' or 'X_score'.
    # We use the case of O_score to explain the code.
    import random
    rough_choice_2 = []
    rough_choice_3 = []
    choice = []
    for i in range(12): # For the case that O has 2 or 3 signs in a line
        if score[i] == 2:
            rough_choice_2.append(i)
        elif score[i] == 3:
            rough_choice_3.append(i)
    is_finished = False
    for i in rough_choice_3:
        tmp = []
        if i < 5: # col
            for j in range(5):
                if get_board_value(board, i, j) == '.':
                    tmp.append(5*j+i)

        elif i < 10: # row
            for j in range(5):
                if get_board_value(board, j, i-5) == '.':
                    tmp.append(5*(i-5)+j)
            
        elif i == 10: #diag1
            for j in range(5):
                if get_board_value(board, j, j) == '.':
                    tmp.append(6*j)
            
        else: #diag2
            for j in range(5):
                if get_board_value(board, 4-j, j) == '.':
                    tmp.append(4*j+4)
        
        if len(tmp) == 1: # O has 3 signs and X has 1 signs
            choice.append(tmp[0])
            board += 1<<(tmp[0]*2)
            if len(choice) == 2:
                is_finished = True
        elif len(tmp) == 2: # O has 3 signs and X has no signs
            a = tmp[random.randint(0,1)]
            choice.append(a)
            board += 1<<(a*2)
            if len(choice) == 2:
                is_finished = True
        
        if is_finished:
            break
    
    if not is_finished: # if not done
        for i in rough_choice_2:
            tmp = []
            if i < 5: # col
                for j in range(5):
                    if get_board_value(board, i, j) == '.':
                        tmp.append(5*j+i)

            elif i < 10: # row
                for j in range(5):
                    if get_board_value(board, j, i-5) == '.':
                        tmp.append(5*(i-5)+j)
                
            elif i == 10: #diag1
                for j in range(5):
                    if get_board_value(board, j, j) == '.':
                        tmp.append(6*j)
                
            else: #diag2
                for j in range(5):
                    if get_board_value(board, 4-j, j) == '.':
                        tmp.append(4*j+4)
            
            if len(tmp) == 2: # O has 2 signs and X has 1 signs
                if choice:
                    a = tmp[random.randint(0,1)]
                    choice.append(a)
                else:
                    choice = tmp[:]
                is_finished = True
            elif len(tmp) == 3: # O has 2 signs and X has no signs
                choice = random.sample(set(tmp), 2)
                is_finished = True
            
            if is_finished:
                break
    
    pick1 = [0, 4, 12, 20, 24]
    pick2 = [6, 8, 16, 18]
    pick3 = [7, 11, 13, 17]
    if len(choice) == 2:
        return choice[:]
    elif len(choice) == 1:
        tmp = []
        for pos in pick1:
            if get_board_value(board, pos%5, pos//5) == '.':
                tmp.append(pos)
        if not tmp:
            for pos in pick2:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        choice.append(tmp[random.randint(0,len(tmp)-1)])
        return choice[:]
    else:
        tmp = []
        for pos in pick1:
            if get_board_value(board, pos%5, pos//5) == '.':
                tmp.append(pos)
        if len(tmp) < 2:
            for pos in pick2:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        if len(tmp) < 2:
            for pos in pick3:
                if get_board_value(board, pos%5, pos//5) == '.':
                    tmp.append(pos)
        
        choice = random.sample(set(tmp), 2)
        return choice[:]

#definition: The method for early board gaming
#input: board, a 50 bits number
#output: next_board, a 50 bits number, which takes two new steps
def heuristic_strategy(board):
    import random
    
    next_board = 0
    depth = 0 # 還有幾步可下
    tmp = board
    for y in range(5):
        for x in range(5):
            if (not tmp&1) and (not tmp&2):
                depth += 1
            tmp >>= 2
    depth = (depth-1)//2
    
    if depth == 12:
        pick = [[0,12],[4,12],[12,20],[12,24],[6,12],[8,12],[12,16],[12,18]]
        random_chart = [0,1,2,3,4,5,6,7,3,2,1,0,0,1,2,3,2,3,1]
        choice = pick[random.sample(set(random_chart),1)[0]]
        next_board = board + (1<<(choice[0]*2+1)) + (1<<(choice[1]*2+1))
    else:
        X_score = [0]*12
        O_score = [0]*12
        for y in range(5):
            for x in range(5):
                if(get_board_value(board, x, y) == 'X'):
                    X_score[x] += 1
                    X_score[y+5] += 1
                    if(x == y):
                        X_score[10] += 1
                    if(x == 4-y):
                        X_score[11] += 1
                elif(get_board_value(board, x, y) == 'O'):
                    O_score[x] += 1
                    O_score[y+5] += 1
                    if(x == y):
                        O_score[10] += 1
                    if(x == 4-y):
                        O_score[11] += 1
        
        defence_strategy = 1 # defence(stop the other from getting points):1, attack:0
        choice = []
        if depth&1: # X's term
            if defence_strategy:
                choice = choice_by_sign_defence(board, O_score)
            else:
                choice = choice_by_sign_attack(board, X_score)
            next_board = board + (1<<(choice[0]*2)) + (1<<(choice[1]*2))
        else:
            if defence_strategy:
                choice = choice_by_sign_defence(board, X_score)
            else:
                choice = choice_by_sign_attack(board, O_score)
            next_board = board + (1<<(choice[0]*2+1)) + (1<<(choice[1]*2+1))
    return next_board

#definition: The method for the board gaming after 5 moves
#input: board, a 50 bits number
#output: move, a list for the next two steps
def best_strategy(board, depth):
    global next_board
    global begin_depth
    
    result = minimax(board, depth, -10, 10, (depth+1)&1)
    return next_board

def get_next_step(state):
    global next_board
    global begin_depth
    global board_data
    
    import json
    global board_data
    
    file_name = 'AlphaBetaPruning\data.json'
    
    depth = 0
    board = set_board_to_ll(state)
    tmp = board
    for y in range(5):
        for x in range(5):
            if (not tmp&1) and (not tmp&2):
                depth += 1
            tmp >>= 2
    depth = (depth-1)//2
    begin_depth = depth
    
    if depth <= 7:
        next_board = best_strategy(board, depth)
        if depth >= 6:
            with open(file_name, 'r', newline='') as jsfile:
                board_data = json.load(jsfile)
            if board in board_data:
                return board_data[board]
    else:
        next_board = heuristic_strategy(board)
    next_move = compare_board(next_board, board)
    if 6 <= depth <= 7:
        write_map(file_name, board, next_move)
    return next_move[:]

def build_data(board, file_name='data.json'):
    import json
    import os
    global next_board
    global begin_depth
    global board_data
    
    if os.path.isfile(file_name):
        with open(file_name, 'r', newline='') as jsfile:
            board_data = json.load(jsfile)
    else:
        with open(file_name, 'w', newline='') as jsfile:
            json.dump({}, jsfile)
    
    depth = 0
    tmp = board
    for y in range(5):
        for x in range(5):
            if (not tmp&1) and (not tmp&2):
                depth += 1
            tmp >>= 2
    depth = (depth-1)//2
    begin_depth = depth
    
    if depth < 7:
        next_board = best_strategy(board, depth)
    else:
        next_board = heuristic_strategy(board)
    next_move = compare_board(next_board, board)
    print('')
    print_board(next_board)
    write_map(file_name, board, next_move)

#definition: Only for testing this code
def test(B):
    global next_board
    global begin_depth

    board = set_board_to_ll(B)
    
    depth = 0
    tmp = board
    for y in range(5):
        for x in range(5):
            if (not tmp&1) and (not tmp&2):
                depth += 1
            tmp >>= 2
    depth = (depth-1)//2
    begin_depth = depth
    result = minimax(board, depth, -10, 10, (depth-1)&1)
    
    print('\nresult:')
    print_board(next_board)
    

if __name__ == '__main__':
    B = [[x for x in input()] for y in range(5)]
    test(B)

