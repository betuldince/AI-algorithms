import copy
from host import GO
from read import readInput
from write import writeOutput
from copy import deepcopy
from board import Board

BLACK = 1
WHITE = 2
N = 5

def get_game_state():
    piece_type, previous_board, board = readInput(N)
    return piece_type, previous_board, board

class MiniMax():
    def __init__(self, current_board, previous_board, piece_type):
        self.piece_type = piece_type
        self.opponent_piece_type = 3 - piece_type  
        self.previous_board = previous_board
        self.current_board = current_board
        self.board_size = current_board.board_size
        self.opponent_pass = Board.KO(self.previous_board, self.current_board)

    def check_game_over(self, board):
        return (self.opponent_pass and Board.KO(self.current_board, board)) or (board.is_board_full())


    
    def find_next_valid_moves2(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.valid_move_check(i,j):
                        valid_moves.append((i, j))                   
        return valid_moves
    
    def valid_move_check(self, i,j):
        if self.current_board.board[i][j] == 0:
            new_board = self.current_board.create_new_board_with_move(self.piece_type, i, j)
            cleaned_new_board = new_board.clean_board_pieces(self.opponent_piece_type)
            if cleaned_new_board.get_liberty(i, j) > 0 and not Board.KO(self.previous_board, cleaned_new_board):
                return True  
        return False


    def minimax_with_move(self, depth, alpha, beta):
        board = self.current_board
        is_minimizing = self.piece_type == 1
        value = float('inf') if is_minimizing else -float('inf')
        best_move = ()   
        is_valid_move = False

        if depth == 0:
 
            return (-1, -1), board.evaluation()

        for move in self.find_next_valid_moves2():
            
            new_board = self.current_board.create_new_board_with_move(self.piece_type, move[0], move[1])
            cleaned_new_board = new_board.clean_board_pieces(self.opponent_piece_type)
            new_minimax_state = MiniMax(cleaned_new_board, self.current_board,self.opponent_piece_type)
            is_valid_move = True

            if self.check_game_over(new_minimax_state.current_board):
                
                new_value = board.evaluation()  
            else:
                new_value = new_minimax_state.minimax_with_move(depth - 1, alpha, beta)[1]

            if is_minimizing:
                if new_value < value:
                    value = new_value
                    best_move = move
                    beta = min(beta, new_value)
            else:
                if new_value > value:
                    value = new_value
                    best_move = move
                    alpha = max(alpha, new_value)

            if beta <= alpha:
                break

        if not is_valid_move:
            return (-1, -1), board.evaluation()

        return best_move, value

    def get_best_move(self, depth):
        return self.minimax_with_move(depth, -float('inf'), float('inf'))

 
 
if __name__ == '__main__':
    piece_type, previous_board, current_board = get_game_state()
    adp_depth = Board(current_board).get_adaptive_depth()
    minimax = MiniMax(Board(current_board), Board(previous_board), piece_type)
    board_pieces = minimax.current_board.get_board_pieces()
    move, value = minimax.get_best_move(depth=adp_depth)
    go = GO(5)
    print(go.n_move)

    if move == (-1, -1):
        writeOutput("PASS")
    else:
        writeOutput(move)
    '''
    num_iter = GO(5).n_move
    if piece_type == BLACK: 
        if(num_iter == 0) and minimax.valid_move_check(2,2):
            move = (2,2)
        elif(num_iter == 3) :
            if(minimax.valid_move_check(3,2)):
                move = (3,2)
            else:
                move, value = minimax.get_best_move(depth=adp_depth)
        elif(num_iter == 5):
            if(current_board[3][2] == 1):
                if(current_board[4][3] == 0) and minimax.valid_move_check(4,3):
                    move = (4,3)
                else:
                    move, value = minimax.get_best_move(depth=adp_depth)
            else:
                if(minimax.valid_move_check(2,1)):
                    move = (2,1)
                else:
                    move, value = minimax.get_best_move(depth=adp_depth)
        else:
            move, value = minimax.get_best_move(depth=adp_depth)
    else:
        move, value = minimax.get_best_move(depth=adp_depth)    

    if board_pieces <= 4:
        opening_moves = [
            (2, 2),
            (2, 3),
            (3, 2),
            (2, 1),
            (1, 2),
        ]

        valid_moves = [move for move, _ in minimax.find_next_valid_moves()]
        for opening_move in opening_moves:
            if opening_move in valid_moves:
                move = opening_move
                break
    '''

 


