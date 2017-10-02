from engines import Engine
from copy import deepcopy
from sys import maxint

class StudentEngine(Engine):
    """ Game engine that you should you as skeleton code for your 
    implementation. """
    alpha_beta    = False
    purning_times = 0
    cutoffDepth   = 4
    node_generate = 0

    def get_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Wrapper function that chooses either vanilla minimax or 
        alpha-beta. """
        f = self.get_ab_minimax_move if self.alpha_beta else self.get_minimax_move
        return f(board, color, move_num, time_remaining, time_opponent)

    def get_minimax_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Skeleton code from greedy.py to get you started. """
        # Get a list of all legal moves.
        moves = board.get_legal_moves(color)
        res_score = float('-inf');
        for move in moves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            curscore = self.minimax_score(newboard, color*-1, 1, -1)
            if (curscore > res_score):
                res_score = curscore
                res_move  = move
        
        #print res_move
        print 'node_generate:', self.node_generate
        return res_move
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        #return max(moves, key=lambda move: self._get_cost(board, color, move))

    def minimax_score(self, board, color, curDepth, isMax):
        self.node_generate += 1
        if curDepth == self.cutoffDepth:
            return self.minimax_evaluation(board, color, isMax)

        legalMoves = board.get_legal_moves(color)
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
        coin_score     = board.count(me_color) -  board.count(me_color*-1)
        
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
        moves     = board.get_legal_moves(color)
        res_score = float('-inf');
        alpha     = float('-inf');
        beta      = float('inf');
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
                print "wtf"
                self.purning_times += 1
                break
            """
        print 'purning_times:', self.purning_times
        print 'node_generate:', self.node_generate
        return res_move
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        #return max(moves, key=lambda move: self._get_cost(board, color, move))

    def ab_minimax_score(self, board, color, curDepth, isMax, alpha, beta):
        self.node_generate += 1
        if curDepth == self.cutoffDepth:
            return self.minimax_evaluation(board, color, isMax)

        legalMoves = board.get_legal_moves(color)
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
                self.purning_times += 1
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
