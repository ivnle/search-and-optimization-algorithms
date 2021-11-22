# %%
import numpy as np
import matplotlib.pyplot as plt

class Node():
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self) -> str:
        return f'Node({self.state})'

def build_tree(root_node, board):
    pass

def minimax_search(board):
    pass


def build_grid_game(n_choices):
    """
    Build a game where there are two players, C and R.
    C wants to choose the column that results in the min value.
    R wants to choose the row that results in the min value.
    """
    # n_choices by n_choices grid of random integers
    grid = np.random.randint(0, 100, size=(n_choices, n_choices))
    return grid
    

if __name__ == '__main__':
    np.random.seed(1)
    board = build_grid_game(n_choices=2)
    print(board)

    # TODO think of a way to compare minimax search against a star search
  

