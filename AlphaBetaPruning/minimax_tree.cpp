#include <cstdio>
#include <vector>
#define ll long long int
#define pb push_back
#define INF 10
#define NINF -10
using namespace std;

ll next_board;
ll next_board_even;
int begin_depth;

ll set_board_to_ll(char board[5][5]){ //.=0, X=1, O=2
	ll a = 0;
	for(int y=0; y<5; y++)
		for(int x=0; x<5; x++){
			if(board[y][x] == 'X')
				a += (ll)1 << ((5*y+x)*2);
			else if(board[y][x] == 'O')
				a += (ll)1 << ((5*y+x)*2+1);
		}
	return a;
}

char get_board_value(ll board, int x, int y){
	int pos = (5*y+x)*2;
	if(board&((ll)1<<pos))
		return 'X';
	else if(board&((ll)1<<(pos+1)))
		return 'O';
	else
		return '.';
}

// return 1 if O wins and return -1 if X wins
int get_result(ll board){
    int X_score[12] = {0};
    int O_score[12] = {0};
    for(int y=0; y<5; y++)
		for(int x=0; x<5; x++){
            if(get_board_value(board, x, y) == 'X'){
                X_score[x] += 1;
                X_score[y+5] += 1;
                if(x == y)
                    X_score[10] += 1;
                if(x == 4-y)
                    X_score[11] += 1;
            }
            else if(get_board_value(board, x, y) == 'O'){
                O_score[x] += 1;
                O_score[y+5] += 1;
                if(x == y)
                    O_score[10] += 1;
                if(x == 4-y)
                    O_score[11] += 1;
            }
        }
    int X = 0;
    int O = 0;
    for(int i=0; i<12; i++){
        if(X_score[i] >= 4)
            X += 1;
        else if(O_score[i] >= 4)
            O += 1;
    }
    // printf("scores:\nX:%d, O:%d\n", X, O);
	if(X>O)
    	return -1;
	else if(O>X)
		return 1;
	else
		return 0;
}

void print_board(ll board){
	for(int y=0; y<5; y++)
		for(int x=0; x<5; x++){
			char sign;
			if(board&1)
				sign = 'X';
			else if(board&2)
				sign = 'O';
			else
				sign = '.';
			board >>= 2;
			printf("%c", sign);
			if(x==4)
				printf("\n");
		}
}

int minimax(ll board, int depth, int alpha, int beta, bool maximizingPlayer){
	if(depth==0)
		return get_result(board);
	else if(depth==1){
		ll tmp = board;
		for(int y=0; y<5; y++)
			for(int x=0; x<5; x++){
				if((!(tmp&1)) && (!(tmp&2))) // fill in 3 X's
					board += (ll)1 << ((5*y+x)*2);
				tmp >>= 2;
			}
		//print_board(board);
		return get_result(board);
	}
	
	vector<int> child;
	ll tmp = board;
	for(int y=0; y<5; y++)
		for(int x=0; x<5; x++){
			if(!(tmp&1) && !(tmp&2))
				child.pb(5*y+x);
			tmp >>= 2;
		}
	
	if(maximizingPlayer){ // O
		int max_eval = NINF;
		int eval = 0;
		ll tmp = 0;
		bool is_even = false;
		for(int i=0; i<depth*2+1; i++){
			for(int j=i+1; j<depth*2+1; j++){
				tmp = board + ((ll)1 << (child[i]*2+1));
				tmp += (ll)1 << (child[j]*2+1);
				//printf("O: %d, %d\n", child[i], child[j]);
				//print_board(tmp);
				eval = minimax(tmp, depth-1, alpha, beta, false);
				
				if(depth==begin_depth)
					if((eval==0) && !is_even){
						next_board_even = tmp;
						is_even = true;
					}
				next_board = tmp;
				
				if(eval==1)
					return eval;
				max_eval = max_eval>eval ? max_eval:eval;
				alpha = alpha>eval ? alpha:eval;
				if(beta <= alpha){
					if(max_eval==0)
						next_board = next_board_even;
					return max_eval;
				}
			}
		}
		if(max_eval==0)
			next_board = next_board_even;
		return max_eval;
	}
	else{ // X
		int min_eval = INF;
		int eval = 0;
		ll tmp = 0;
		bool is_even = false;
		for(int i=0; i<depth*2+1; i++){
			for(int j=i+1; j<depth*2+1; j++){
				tmp = board + ((ll)1 << (child[i]*2));
				tmp += (ll)1 << (child[j]*2);
				//printf("X: %d, %d\n", child[i], child[j]);
				//print_board(tmp);
				eval = minimax(tmp, depth-1, alpha, beta, true);
				
				if(depth==begin_depth)
					if((eval==0) && (!is_even)){
						next_board_even = tmp;
						is_even = true;
					}
				next_board = tmp;
				
				if(eval==-1)
					return eval;
				
				min_eval = min_eval>eval ? eval:min_eval;
				beta = beta>eval ? eval:beta;
				if(beta <= alpha){
					if(min_eval==0)
						next_board = next_board_even;
					return min_eval;
				}
			}
		}
		if(min_eval==0)
			next_board = next_board_even;
		return min_eval;
	}
}

ll run(ll b, int depth){
	ll tmp = b;
	begin_depth = depth;
	int result = minimax(b, depth, NINF, INF, (depth+1)&1);
	return next_board;
}

int main(){
	char board[5][5];
	for(int y=0; y<5; y++)
		scanf("%s", board[y]);
	ll b = set_board_to_ll(board);
	
	int depth = 0;
	ll tmp = b;
	for(int y=0; y<5; y++)
		for(int x=0; x<5; x++){
			if(!(tmp&1) && !(tmp&2)) // unfilled
				depth += 1;
			tmp >>= 2;
		}
	depth = (depth-1)/2;
	begin_depth = depth;
	int result = minimax(b, depth, NINF, INF, (depth-1)&1);
	
	printf("result:\n");
	print_board(next_board);
	printf("\n");
	
	return 0;
}
