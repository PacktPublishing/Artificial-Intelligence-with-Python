# This is a variant of the Game of Bones recipe given in the easyAI library

from easyAI import TwoPlayersGame, id_solve, Human_Player, AI_Player
from easyAI.AI import TT

class LastCoinStanding(TwoPlayersGame):
    def __init__(self, players):
        # Define the players. Necessary parameter.
        self.players = players

        # Define who starts the game. Necessary parameter.
        self.nplayer = 1 

        # Overall number of coins in the pile 
        self.num_coins = 25

        # Define max number of coins per move 
        self.max_coins = 4 

    # Define possible moves
    def possible_moves(self): 
        return [str(x) for x in range(1, self.max_coins + 1)]
    
    # Remove coins
    def make_move(self, move): 
        self.num_coins -= int(move) 

    # Did the opponent take the last coin?
    def win(self): 
        return self.num_coins <= 0 

    # Stop the game when somebody wins 
    def is_over(self): 
        return self.win() 

    # Compute score
    def scoring(self): 
        return 100 if self.win() else 0

    # Show number of coins remaining in the pile
    def show(self): 
        print(self.num_coins, 'coins left in the pile')

if __name__ == "__main__":
    # Define the transposition table
    tt = TT()

    # Define the method
    LastCoinStanding.ttentry = lambda self: self.num_coins

    # Solve the game
    result, depth, move = id_solve(LastCoinStanding, 
            range(2, 20), win_score=100, tt=tt)
    print(result, depth, move)  

    # Start the game 
    game = LastCoinStanding([AI_Player(tt), Human_Player()])
    game.play() 

