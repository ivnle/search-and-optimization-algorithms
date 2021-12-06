# %%
import numpy as np
import matplotlib.pyplot as plt

ACTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def construct_maze(n_rows, n_cols, pct_obstacles):
    """
    Args:
        n_rows: number of rows in maze
        n_cols: number of columns in maze
        pct_obstacles: percentage of maze that are obstacles

    Returns:
        maze: numpy array of shape (n_rows, n_cols)
        start_position: tuple of (row, col) of start position
        goal_position: tuple of (row, col) of goal position

    -1 = obstacle
    1 = start
    2 = goal
    """
    maze = np.zeros((n_rows, n_cols))
    n_obstacles = int(n_rows*n_cols*pct_obstacles)

    # pick random cell and set as start
    start_row = np.random.randint(low=0, high=n_rows)
    start_col = np.random.randint(low=0, high=n_cols)
    maze[start_row, start_col] = 1

    # pick random cell and set as goal
    goal_row = np.random.randint(0, n_rows)
    goal_col = np.random.randint(0, n_cols)
    maze[goal_row, goal_col] = 2

    for i in range(n_rows):
        for j in range(n_cols):
            if maze[i, j] != 1 and maze[i, j] != 2:
                if np.random.rand() < pct_obstacles:
                    maze[i, j] = -1

    return maze, (start_row, start_col), (goal_row, goal_col)


def manhattan_distance(goal_position):
    """
    Args:
        goal_position: tuple of (row, col) of goal position
    
    Returns:
        curried_distance: function that takes a node and returns the manhattan distance from the node to the goal
    """
    def curried_manhattan_distance(x: Node):
        return abs(x.position[0] - goal_position[0]) + abs(x.position[1] - goal_position[1])
    return curried_manhattan_distance


class Node():
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __repr__(self) -> str:
        return f'Node({self.position})'

    def __eq__(self, n) -> bool:
        return self.position == n.position


def a_star_search(maze, start_position, goal_position):
    """
    Args:
        maze: numpy array of shape (n_rows, n_cols)
        start_position: tuple of (row, col) of start position
        goal_position: tuple of (row, col) of goal position

    Returns:
        node: Node object of goal position. Use the parent of this node to trace back to start position.

    """
    manhat_dist = manhattan_distance(goal_position)

    start_node = Node(start_position)
    start_node.g = 0
    start_node.h = manhat_dist(start_node)
    start_node.f = start_node.h + start_node.g

    frontier = [start_node]  # TODO turn into priority queue
    explored = []

    while len(frontier) > 0:

        # sort frontier by manhat_dist from high to low
        frontier = sorted(frontier, key=lambda x: x.f, reverse=True)
        node = frontier.pop()

        if node.position == goal_position:
            return node

        explored.append(node)

        # get neighbors
        for action in ACTIONS.values():
            # new position
            new_position = (
                node.position[0] + action[0], node.position[1] + action[1])

            # check if valid action
            if new_position[0] < 0 or new_position[0] >= maze.shape[0]:
                continue
            if new_position[1] < 0 or new_position[1] >= maze.shape[1]:
                continue
            if maze[new_position[0], new_position[1]] == -1:
                continue

            child = Node(new_position, parent=node)
            child.g = node.g + 1
            child.h = manhat_dist(child)
            child.f = child.g + child.h

            if child not in frontier and child not in explored:
                frontier.append(child)
            elif child in frontier:
                # check if current child is better than previous child
                if child.f < frontier[frontier.index(child)].f:
                    frontier[frontier.index(child)] = child
        
    print('No path exists from start to goal.')


def get_path(node):
    path = []
    while node.parent is not None:
        path.append(node.position)
        node = node.parent
    path.append(node.position)
    path.reverse()
    return path

if __name__ == '__main__':

    #np.random.seed(1)
    np.random.seed(6)

    maze, start_position, goal_position = construct_maze(
        n_rows=10, n_cols=10, pct_obstacles=0.4)

    print(maze)
    plt.imshow(maze)

    goal_node = a_star_search(maze, start_position, goal_position)
    
    path = get_path(goal_node)

    print(f'Least cost path: {path}')

    for x, y in path:
        maze[x, y] = 2
    plt.imshow(maze)

    # TODO add visualization of how frontier and explored sets are updated over each iteration

