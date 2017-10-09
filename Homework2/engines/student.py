from engines import Engine
from copy import deepcopy
from sys import maxint
import timeit

class StudentEngine(Engine):
    """ Game engine that you should you as skeleton code for your 
    implementation. """
    alpha_beta     = False
    cutoffDepth    = 2
    cutoffDepth_AB = 4
    node_generate  = 0
    leaves_count   = 0
    turn_count     = 0
    total_time     = 0
    branch_factors = 0
    duplicat_count = 0
    board_states   = set([])

    def get_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Wrapper function that chooses either vanilla minimax or 
        alpha-beta. """
        f = self.get_ab_minimax_move if self.alpha_beta else self.get_minimax_move
        return f(board, color, move_num, time_remaining, time_opponent)

    def duplicate_check(self, board, color):
        theState  = '3' if color == 1 else '4'
        for i in xrange(8):
            for j in xrange(8):
                if board[i][j] == 1:
                    theState += '1'
                elif board[i][j] == -1:
                    theState += '2'
                else:
                    theState += '0'
        if theState not in self.board_states:
            self.board_states.add(theState)
        else:
            self.duplicat_count += 1

    def print_helper(self, elapsed, avg_time):
        print '\nTurn Report:'
        # print 'total   # of nodes generated: (alpha_beta):', self.node_generate
        print 'average # of nodes generated per turn:', self.node_generate/self.turn_count
        print 'average # of duplicate nodes per turn:', self.duplicat_count/self.turn_count
        print 'average brancing factors for a node  :', self.branch_factors /(self.node_generate - self.leaves_count)
        # print 'runing time of the turn:      (ms)', (format(elapsed, '.2f'))
        print 'average runing time per turn (ms)    :', (format(avg_time, '.2f'))

    def get_minimax_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Skeleton code from greedy.py to get you started. """
        # Get a list of all legal moves.
        start_time = timeit.default_timer()
        moves = board.get_legal_moves(color)
        self.turn_count += 1;
        res_score = float('-inf');
        self.branch_factors += len(moves)
        for move in moves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            curscore = self.minimax_score(newboard, color*-1, 1, -1)
            if (curscore > res_score):
                res_score = curscore
                res_move  = move
        
        # print res_move
        elapsed         = (timeit.default_timer() - start_time)*1000
        self.total_time += elapsed
        avg_time = self.total_time/self.turn_count

        # self.print_helper(elapsed, avg_time)

        return res_move
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        #return max(moves, key=lambda move: self._get_cost(board, color, move))

    def minimax_score(self, board, color, curDepth, isMax):
        self.duplicate_check(board, -1*color)
        self.node_generate += 1
        if curDepth == self.cutoffDepth:
            self.leaves_count += 1
            return self.minimax_evaluation(board, color, isMax)

        legalMoves = board.get_legal_moves(color)
        self.branch_factors += len(legalMoves)
        if len(legalMoves) == 0:
            return self.minimax_evaluation(board, color, isMax)

        res_score  = float('inf') if isMax == -1 else float('-inf')
        for move in legalMoves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            curscore = self.minimax_score(newboard, color*-1, curDepth+1, isMax*-1)

            if (isMax == -1 and curscore < res_score) or (isMax == 1 and curscore > res_score):
                res_score = curscore

        return res_score

    def minimax_evaluation(self, board, color, isMax):
        weight_count     = 0.1
        weight_lmove     = 1
        weight_corner    = 10
        me_color         = color if isMax == 1 else color*-1
        op_color         = me_color*-1
        
        # coin diff
        coin_score     = board.count(me_color) - board.count(me_color*-1)
        
        # legalMove_diff
        lmove_op = len(board.get_legal_moves(op_color))
        lmove_me = len(board.get_legal_moves(me_color))
        lmove_score = lmove_me - lmove_op

        # corner diff
        corner_op      = 0
        corner_me      = 0
        if board[0][0] == me_color:
            corner_me += 1
        if board[0][7] == me_color:
            corner_me += 1
        if board[7][7] == me_color:
            corner_me += 1
        if board[7][0] == me_color:
            corner_me += 1

        if board[0][0] == op_color:
            corner_op += 1
        if board[0][7] == op_color:
            corner_op += 1
        if board[7][7] == op_color:
            corner_op += 1
        if board[7][0] == op_color:
            corner_op += 1
        corner_score   = corner_me - corner_op
       
        return weight_count*coin_score + weight_lmove*lmove_score + weight_corner*corner_score

    def get_ab_minimax_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Skeleton code from greedy.py to get you started. """
        # Get a list of all legal moves.
        start_time = timeit.default_timer()
        self.turn_count += 1;
        moves     = board.get_legal_moves(color)
        res_score = float('-inf');
        alpha     = float('-inf');
        beta      = float('inf');
        self.branch_factors += len(moves)
        for move in moves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            curscore = self.ab_minimax_score(newboard, color*-1, 1, -1, alpha, beta)
            if (curscore > res_score):
                res_score = curscore
                res_move  = move
            
            alpha = max(alpha, res_score)
            """
            if beta <= alpha:
                break
            """
        elapsed         = (timeit.default_timer() - start_time)*1000
        self.total_time += elapsed
        avg_time = self.total_time/self.turn_count

        # self.print_helper(elapsed, avg_time)
        
        return res_move
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        #return max(moves, key=lambda move: self._get_cost(board, color, move))

    def ab_minimax_score(self, board, color, curDepth, isMax, alpha, beta):
        self.duplicate_check(board, -1*color)
        self.node_generate += 1
        if curDepth == self.cutoffDepth_AB:
            return self.minimax_evaluation(board, color, isMax)

        legalMoves = board.get_legal_moves(color)
        self.branch_factors += len(legalMoves)
        if len(legalMoves) == 0:
            return self.minimax_evaluation(board, color, isMax)

        res_score  = float('inf') if isMax == -1 else float('-inf')
        for move in legalMoves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            curscore = self.ab_minimax_score(newboard, color*-1, curDepth+1, isMax*-1, alpha, beta)

            # alpha_beta core modification
            if isMax == 1:
                res_score = max(res_score, curscore)
                alpha     = max(res_score, alpha)
            else:
                res_score = min(res_score, curscore)
                beta      = min(res_score, beta)

            if beta <= alpha:
                break

        return res_score


    def _get_cost(self, board, color, move):
        """ Return the difference in number of pieces after the given move 
        is executed. """
        
        # Create a deepcopy of the board to preserve the state of the actual board
        newboard = deepcopy(board)
        newboard.execute_move(move, color)

        # Count the # of pieces of each color on the board
        num_pieces_op = len(newboard.get_squares(color*-1))
        num_pieces_me = len(newboard.get_squares(color))

        # Return the difference in number of pieces
        return num_pieces_me - num_pieces_op
        
engine = StudentEngine
