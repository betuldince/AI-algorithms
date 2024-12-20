from copy import deepcopy

BLACK = 1
WHITE = 2
N = 5

#The Board functions are majorly taken from the host.py besides evaluation of board
class Board():
    def __init__(self, board):
        self.board = board
        self.board_size = len(board)
        self.liberty = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.total_pieces = None

    def clone_board(self):
        new_board = deepcopy(self.board)
        return Board(new_board)

    def get_neighbors(self, i, j):
       
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < self.board_size - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < self.board_size - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def get_allies(self, i, j):
        
        neighbors = self.get_neighbors(i, j)
        allies = []
        empty_spots = []
        ally_piece = self.board[i][j]
        for piece in neighbors:
            current_piece = self.board[piece[0]][piece[1]]
            if current_piece == ally_piece:
                allies.append(piece)
            elif current_piece == 0:
                empty_spots.append(piece)
        return allies, empty_spots

    def get_threads(self, i, j):
        neighbors = self.get_neighbors(i, j)
        threads = []
        thread_piece = 3 - self.board[i][j]
        
        for piece in neighbors:
            current_piece = self.board[piece[0]][piece[1]]
                       
            if current_piece == thread_piece:
                
                threads.append(piece)

        return threads, len(threads)


    def count_total_pieces(self):
        total_pieces = sum(row.count(BLACK) + row.count(WHITE) for row in self.board)
        return total_pieces

    def get_adaptive_depth(self):
        total_pieces = self.count_total_pieces()
        if total_pieces < 15:
            return 4
        elif total_pieces < 16:
            return 3
        else:
            return 2

    def get_all_ally_or_empty(self, i, j):
        queue = [(i, j)]
        all_allies = set()
        all_empty = set()
        all_allies.add((i, j))
        while queue:
            i, j = queue.pop(0)
            allies, empty = self.get_allies(i, j)
            for ally in allies:
                if ally not in all_allies:
                    queue.append(ally)
                    all_allies.add(ally)
            for zero in empty:
                all_empty.add(zero)
        return all_allies, all_empty

    def get_liberty(self, i, j):
        if self.liberty[i][j] == None:
            allies, empty = self.get_all_ally_or_empty(i, j)
            liberty = len(empty)
            for x, y in allies:
                self.liberty[x][y] = liberty
        return self.liberty[i][j]
    
    def cluster_liberty(self, i, j):
        cluster, _ = self.get_all_ally_or_empty(i, j)
        liberties = 0
        for stone in cluster:
            neighbors = self.get_neighbors(stone[0], stone[1])
            for neighbor in neighbors:
                if self.board[neighbor[0]][neighbor[1]] == 0:
                    liberties += 1
        return liberties

    def clean_board_pieces(self, piece_type):
        dead_pieces = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == piece_type and self.get_liberty(i, j) == 0:
                    dead_pieces.append((i, j))


        if not dead_pieces:
            return self
        else:
            new_board = self.clone_board()
            for i, j in dead_pieces:
                new_board.board[i][j] = 0
            return new_board

    def create_new_board_with_move(self, piece_type, i, j):
        new_board = self.clone_board()
        new_board.board[i][j] = piece_type
        return new_board

    def get_board_pieces(self):
        if self.total_pieces == None:
            self.total_pieces = self.count_total_pieces()
        return self.total_pieces

    def is_board_full(self):
        return self.get_board_pieces() >= N ** 2 - 1

    @staticmethod
    def KO(board1, board2):
        for i in range(board1.board_size):
            for j in range(board1.board_size):
                if board1.board[i][j] != board2.board[i][j]:
                    return False
        return True
    
    def get_square_shape_count(self, piece_type):
        square_shape = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    neighbors = self.get_neighbors(i, j)
                    if all(self.board[x][y] == piece_type for x, y in neighbors):
                        square_shape += 1
        return square_shape


    def punish_corners(self,i,j):
        corner_pos = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(2,0),(3,0),(4,0),(4,0),(4,1),(4,2),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)  ]
        position_heurr = 0
        multiply_factor = 2
        middle_pos = (2,2)
        if (i,j) in corner_pos:
            position_heurr += multiply_factor
        elif (i,j) == middle_pos :
            position_heurr += multiply_factor**3
        else:
            position_heurr += multiply_factor**2  

        return position_heurr

    
    def adaptive_factor(self):
        factor = 1 - (self.get_board_pieces() / (N ** 2))
        factor_square = factor ** 2
        return factor, factor_square

    def adaptive_factor2(self):
        #factor = (self.get_board_pieces()*10)
        factor = 50
        return factor
    
    def check_middle_coverage(self):
        corner_pos = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(2,0),(3,0),(4,0),(4,0),(4,1),(4,2),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)  ]
        counter = 0
        for i in range(N):
            for j in range(N):
                if not (i,j) in corner_pos:
                    if self.board[i][j] != 0:
                        counter += 1
        return counter




    def evaluation(self):
        result = 0        
        board_pieces = self.get_board_pieces()
        liberty_penalty_heur = 0
        liberty_heur = 0
        stone_heur = 0
        position_heurr = 0
        square_heurr = 0


        for i in range(N):
            for j in range(N):
                piece = self.board[i][j]               
                if piece == BLACK:
                    liberty = self.get_liberty(i, j)
                    liberty_heur -= liberty
                    _, thread_num = self.get_threads(i,j)
                    if liberty <= 1:
                        liberty_penalty_heur += self.adaptive_factor2()
                    #elif liberty == 2 and thread_num == 2:
                        #liberty_penalty -= 20
                    position_heurr -= self.punish_corners(i,j)                    
                    stone_heur -= 1
                    #heur_cluster1 -= (stone_heur + self.cluster_liberty(i, j))                 
                elif piece == WHITE:
                    liberty = self.get_liberty(i, j)
                    liberty_heur += liberty
                    _, thread_num = self.get_threads(i,j)
                    if liberty <= 1:
                        liberty_penalty_heur -= self.adaptive_factor2()
                    #elif liberty == 2 and thread_num == 2:
                        #liberty_penalty += 20                        
                    position_heurr += self.punish_corners(i,j)
                    stone_heur += 1
                    #heur_cluster1 += (stone_heur + self.cluster_liberty(i, j))

 
        board_coverage_factor, board_coverage_factor_squared = self.adaptive_factor()
        kill_heur_score = (stone_heur * 10) ** 3
        stone_coverage_heur = position_heurr * board_coverage_factor 
        liberty_heur_score = liberty_heur * board_coverage_factor_squared
        #cluster_diff_score = heur_cluster1 * board_coverage_factor
        liberty_penalty_score = liberty_penalty_heur * board_coverage_factor_squared
        #if the middle getting populated dont use this heuristic anymore
       #print('kill', kill_heur_score , ' stone', stone_coverage_heur, 'liberty', liberty_heur_score, 'liberty penalty',liberty_penalty_score)
        if(self.check_middle_coverage()>6):
            stone_coverage_heur = 0

        result = (
            kill_heur_score +
            stone_coverage_heur +
            liberty_heur_score +
            liberty_penalty_score                                       
        )


        
        return result
    

