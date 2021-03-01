import minimax_tree

#input: board, a 50 bits number
#output: An array which has 0~24th entris as board state and 25th entry to represent whose term.
#        O: 1
#        X: -1
#        No sign: 0
def set_ll_board_to_array(board):
    state = []
    
    depth = 0
    tmp = board
    for y in range(5):
        for x in range(5):
            if tmp&1: # X: -1
                state.append(-1)
            elif tmp&2: # O: 1
                state.append(1)
            else: # .: 0 (No sign)
                state.append(0)
                depth += 1
            tmp >>= 2
    depth = (depth-1)//2
    
    if depth&1: # X's term
        state.append(1)
    else:
        state.append(-1)
    return state

if __name__ == '__main__':
    B = [[x for x in input()] for y in range(5)]
    b = minimax_tree.set_board_to_ll(B)
    
    minimax_tree.build_data(b)