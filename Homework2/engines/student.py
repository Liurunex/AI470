from engines import Engine
from copy import deepcopy
from sys import maxint

class StudentEngine(Engine):
    """ Game engine that you should you as skeleton code for your 
    implementation. """
    alpha_beta  = False

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
            curscore = self.minimax_score(newboard, color*-1, 1)
            if (curscore > res_score):
                res_score = curscore
                res_move  = move
        return res_move
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        return max(moves, key=lambda move: self._get_cost(board, color, move))

    def minimax_score(self, board, color, curDepth):
        cutoffDepth = 4
        legalMoves = board.get_legal_moves(color)
        if curDepth == cutoffDepth or len(legalMoves) == 0:
            return self.minimax_evaluation(board, color)
        
        res_score  = float('inf') if color == 1 else float('-inf')
        for move in legalMoves:
            newboard = deepcopy(board)
            newboard.execute_move(move, color)
            color_op = color*-1
            curscore = self.minimax_score(newboard, color_op, curDepth+1)

            if (color == 1 and curscore < res_score) or (color == -1 and curscore > res_score):
                res_score = curscore
        
        return res_score


    def minimax_evaluation(self, board, color):
        num_pieces_op = len(board.get_squares(color*-1))
        num_pieces_me = len(board.get_squares(color))

        return num_pieces_me - num_pieces_op

    def get_ab_minimax_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        """ Skeleton code from greedy.py to get you started. """
        # Get a list of all legal moves.
        moves = board.get_legal_moves(color)
        
        # Return the best move according to our simple utility function:
        # which move yields the largest different in number of pieces for the
        # given color vs. the opponent?
        return max(moves, key=lambda move: self._get_cost(board, color, move))

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
