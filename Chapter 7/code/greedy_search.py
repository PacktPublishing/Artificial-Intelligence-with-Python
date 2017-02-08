import argparse
import simpleai.search as ss 

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Creates the input string \
            using the greedy algorithm')
    parser.add_argument("--input-string", dest="input_string", required=True,
            help="Input string")
    parser.add_argument("--initial-state", dest="initial_state", required=False,
            default='', help="Starting point for the search")
    return parser

class CustomProblem(ss.SearchProblem):
    def set_target(self, target_string):
        self.target_string = target_string

    # Check the current state and take the right action
    def actions(self, cur_state):
        if len(cur_state) < len(self.target_string):
            alphabets = 'abcdefghijklmnopqrstuvwxyz'
            return list(alphabets + ' ' + alphabets.upper())
        else:
            return []

    # Concatenate state and action to get the result
    def result(self, cur_state, action):
        return cur_state + action

    # Check if goal has been achieved
    def is_goal(self, cur_state):
        return cur_state == self.target_string

    # Define the heuristic that will be used
    def heuristic(self, cur_state):
        # Compare current string with target string 
        dist = sum([1 if cur_state[i] != self.target_string[i] else 0
                    for i in range(len(cur_state))])

        # Difference between the lengths
        diff = len(self.target_string) - len(cur_state)

        return dist + diff 

if __name__=='__main__':
    args = build_arg_parser().parse_args()

    # Initialize the object
    problem = CustomProblem()

    # Set target string and initial state
    problem.set_target(args.input_string)
    problem.initial_state = args.initial_state

    # Solve the problem
    output = ss.greedy(problem)

    print('\nTarget string:', args.input_string)
    print('\nPath to the solution:')
    for item in output.path():
        print(item)
