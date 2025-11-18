import argparse
import random
import copy
from collections import deque
import time

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process some attributes.', add_help=False)
    parser.add_argument('-r', '--rows', type=int, help='row count')
    parser.add_argument('-c', '--columns', type=int, help='column count')
    parser.add_argument('-b', '--bfs', action='store_true', help='breadth first search')
    parser.add_argument('-d', '--dfs', action='store_true', help='depth first search')
    parser.add_argument('-i', '--idfs', action='store_true', help='iterative deepening DFS')
    parser.add_argument('-h', '--bf', type=int, metavar='id_of_heuristic', help='best-first strategy')
    parser.add_argument('-a', '--astar', type=int, metavar='id_of_heuristic', help='A* strategy')
    parser.add_argument('-s', '--sma', type=int, metavar='id_of_heuristic', help='SMA* strategy')
    parser.add_argument('--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()

    # Default values if not specified on input
    print()
    if (args.rows is None):
        print(f"row count not specified - defaulting to 4")
        args.rows = 3
    if (args.columns is None):
        print(f"column count not specified - defaulting to 4")
        args.columns = 3
    print()

    # Set Values
    rows = args.rows
    columns = args.columns
    grid = generateSolvableGrid(rows, columns)
    print()

    
    # Print the grid
    print("Randomly generated grid:")
    print()
    printGrid(grid, rows, columns)

    timeStart = time.time()
    if args.bfs:
        print("Breadth first search solution:")
        print()
        result = bfs(grid, rows, columns)
        print(result)
        print()
    elif args.dfs:
        print("Depth first search solution:")
        print()
        result = dfs(grid, rows, columns)
        print(result)
        print()
    elif args.idfs:
        print("Iterative deepening DFS solution:")
        print()
        result = idfs(grid, rows, columns)
        print(result)
        print()
    elif args.bf is not None:
        print(f"Best-first search solution (heuristic {args.bf}):")
        print()
        result = best_first(grid, rows, columns, args.bf)
        print(result)
        print()
    elif args.astar is not None:
        print(f"A* search solution (heuristic {args.astar}):")
        print()
        result = astar(grid, rows, columns, args.astar)
        print(result)
        print()
    elif args.sma is not None:
        print(f"SMA* search solution (heuristic {args.sma}):")
        print()
        result = sma(grid, rows, columns, args.sma)
        print(result)
        print()
    timeEnd = time.time()
    print(f"Time taken: {timeEnd - timeStart} seconds")
    print(f"Length of solution: {len(result)}")

def printGrid(grid, rows, columns):
    for i in range(rows):
        for j in range (columns):
            if grid[i][j] < 10:
                if grid[i][j] == 0:
                    print(f"   ", end="")
                else:
                    print(f" {grid[i][j]} ", end="")
            else:
                print(grid[i][j], end=" ")
        print()
    print()

def find_zero(grid, rows, columns):
    """Find the position of the zero (empty space) in the grid."""
    for i in range(rows):
        for j in range(columns):
            if grid[i][j] == 0:
                return i, j
    raise ValueError("No zero in grid")

def generateSolvableGrid(rows, columns):
    """Generate a solvable puzzle by starting from solved state and making random moves."""
    # Start with solved grid
    grid = []
    for i in range(rows):
        row = []
        for j in range(columns):
            if i == rows - 1 and j == columns - 1:
                row.append(0)  # Empty space at bottom-right
            else:
                row.append(i * columns + j + 1)
        grid.append(row)
    
    # Make random valid moves to shuffle
    num_moves = random.randint(50, 200)  # Number of random moves to make
    return shuffleGrid(grid, rows, columns, num_moves)

def shuffleGrid(grid, rows, columns, num_moves):
    """Shuffle grid by making random valid moves. Guarantees solvability."""
    zero_i, zero_j = rows - 1, columns - 1  # Start from solved position (bottom-right)
    
    for _ in range(num_moves):
        possible_moves = []
        # L: piece from RIGHT (j+1) moves LEFT into empty space
        if zero_j < columns - 1:
            possible_moves.append(("L", zero_i, zero_j + 1))
        # R: piece from LEFT (j-1) moves RIGHT into empty space
        if zero_j > 0:
            possible_moves.append(("R", zero_i, zero_j - 1))
        # U: piece from BELOW (i+1) moves UP into empty space
        if zero_i < rows - 1:
            possible_moves.append(("U", zero_i + 1, zero_j))
        # D: piece from ABOVE (i-1) moves DOWN into empty space
        if zero_i > 0:
            possible_moves.append(("D", zero_i - 1, zero_j))
        
        move_dir, piece_i, piece_j = random.choice(possible_moves)
        # Swap in-place: move piece into empty space
        grid[zero_i][zero_j], grid[piece_i][piece_j] = grid[piece_i][piece_j], grid[zero_i][zero_j]
        zero_i, zero_j = piece_i, piece_j
    
    return grid

def isSolved(grid, rows, columns):
    for i in range(rows):
        for j in range(columns):
            expected = i * columns + j + 1
            if i == rows - 1 and j == columns - 1:
                # Last position should be 0 (empty)
                if grid[i][j] != 0:
                    return False
            else:
                if grid[i][j] != expected:
                    return False
    return True

def bfs(grid, rows, columns):
    """Breadth-first search using a queue to find the shortest solution.
    
    Note: This works well for small puzzles (2x2, 3x3) but becomes impractical
    for larger puzzles (4x4+) due to exponential state space growth.
    """
    # Queue stores (grid, path, last_move)
    queue = deque([(grid, [], "")])
    visited = set()
    
    while queue:
        current_grid, current_path, last_move = queue.popleft()
        
        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path
        
        # Convert grid to tuple for visited set
        grid_tuple = tuple(tuple(row) for row in current_grid)
        
        # Skip if already visited
        if grid_tuple in visited:
            continue
        
        # Mark as visited
        visited.add(grid_tuple)
        
        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)
        
        # Try all possible moves
        # Move notation: L/R/U/D means a PIECE moves in that direction into the empty space
        moves = []
        # L: piece from RIGHT (j+1) moves LEFT into empty space
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        # R: piece from LEFT (j-1) moves RIGHT into empty space
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        # U: piece from BELOW (i+1) moves UP into empty space
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        # D: piece from ABOVE (i-1) moves DOWN into empty space
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))
        
        # Add all valid moves to queue
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            # Move the piece into the empty space
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0
            queue.append((new_grid, current_path + [move_dir], move_dir))
    
    return "error: No solution found"

def dfs(grid, rows, columns):
    """Depth-first search using a stack to find the shortest solution.
    
    Note: This works well for small puzzles (2x2, 3x3) but becomes impractical
    for larger puzzles (4x4+) due to exponential state space growth.
    """
    # Queue stores (grid, path, last_move)
    queue = deque([(grid, [], "")])
    visited = set()
    
    while queue:
        current_grid, current_path, last_move = queue.popleft()
        
        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path
        
        # Convert grid to tuple for visited set
        grid_tuple = tuple(tuple(row) for row in current_grid)
        
        # Skip if already visited
        if grid_tuple in visited:
            continue
        
        # Mark as visited
        visited.add(grid_tuple)
        
        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)
        
        # Try all possible moves
        # Move notation: L/R/U/D means a PIECE moves in that direction into the empty space
        moves = []
        # L: piece from RIGHT (j+1) moves LEFT into empty space
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        # R: piece from LEFT (j-1) moves RIGHT into empty space
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        # U: piece from BELOW (i+1) moves UP into empty space
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        # D: piece from ABOVE (i-1) moves DOWN into empty space
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))
        
        # Add one valid move to queue
        move_dir, piece_i, piece_j = random.choice(moves)
        new_grid = copy.deepcopy(current_grid)
        # Move the piece into the empty space
        new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
        new_grid[piece_i][piece_j] = 0
        queue.append((new_grid, current_path + [move_dir], move_dir))
    
    return "error: No solution found"

def idfs(grid, rows, columns):
    """Iterative deepening DFS - not yet implemented."""
    # TODO: Implement iterative deepening DFS
    return "error: IDDFS not implemented"

def best_first(grid, rows, columns, heuristic_id):
    """Best-first search - not yet implemented."""
    # TODO: Implement best-first search with given heuristic
    return f"error: Best-first search with heuristic {heuristic_id} not implemented"

def astar(grid, rows, columns, heuristic_id):
    """A* search - not yet implemented."""
    # TODO: Implement A* search with given heuristic
    return f"error: A* search with heuristic {heuristic_id} not implemented"

def sma(grid, rows, columns, heuristic_id):
    """SMA* search - not yet implemented."""
    # TODO: Implement SMA* search with given heuristic
    return f"error: SMA* search with heuristic {heuristic_id} not implemented"

if __name__ == "__main__":
    main()