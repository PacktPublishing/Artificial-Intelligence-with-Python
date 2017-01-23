# This is a variant of the Connect Four recipe given in the easyAI library

import numpy as np
from easyAI import TwoPlayersGame, Human_Player, AI_Player, \
        Negamax, SSS

class GameController(TwoPlayersGame):
    def __init__(self, players, board = None):
        # Define the players
        self.players = players

        # Define the configuration of the board
        self.board = board if (board != None) else (
            np.array([[0 for i in range(7)] for j in range(6)]))

        # Define who starts the game
        self.nplayer = 1

        # Define the positions
        self.pos_dir = np.array([[[i, 0], [0, 1]] for i in range(6)] +
                   [[[0, i], [1, 0]] for i in range(7)] +
                   [[[i, 0], [1, 1]] for i in range(1, 3)] +
                   [[[0, i], [1, 1]] for i in range(4)] +
                   [[[i, 6], [1, -1]] for i in range(1, 3)] +
                   [[[0, i], [1, -1]] for i in range(3, 7)])

    # Define possible moves
    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    # Define how to make the move
    def make_move(self, column):
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.nplayer

    # Show the current status
    def show(self):
        print('\n' + '\n'.join(
                ['0 1 2 3 4 5 6', 13 * '-'] +
                [' '.join([['.', 'O', 'X'][self.board[5 - j][i]]
                for i in range(7)]) for j in range(6)]))

    # Define what a loss_condition looks like 
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
                if self.board[pos[0], pos[1]] == self.nopponent:
                    streak += 1
                    if streak == 4:
                        return True
                else:
                    streak = 0

                pos = pos + direction

        return False

    # Check if the game is over
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()

    # Compute the score
    def scoring(self):
        return -100 if self.loss_condition() else 0

if __name__ == '__main__':
    # Define the algorithms that will be used
    algo_neg = Negamax(5)
    algo_sss = SSS(5)

    # Start the game
    game = GameController([AI_Player(algo_neg), AI_Player(algo_sss)])
    game.play()

    # Print the result
    if game.loss_condition():
        print('\nPlayer', game.nopponent, 'wins.')
    else:
        print("\nIt's a draw.")

