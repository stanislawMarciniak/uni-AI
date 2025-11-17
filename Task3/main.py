import argparse
import random
import copy
from collections import deque
import time

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process some attributes.')
    parser.add_argument('-r', '--rows', type=int, help='row count')
    parser.add_argument('-c', '--columns', type=int, help='column count')
    parser.add_argument('-b', '--bfs', action='store_true', help='breadth first search')
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

    # Values
    rows = args.rows
    columns = args.columns
    grid = generateSolvableGrid(rows, columns)
    print()

    
    # Print the grid
    print("Randomly generated grid:")
    print()

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

    timeStart = time.time()
    if args.bfs:
        print("Breadth first search solution:")
        print()
        result = bfs(grid, rows, columns, [], None)
        print(result)
        print()
    timeEnd = time.time()
    print(f"Time taken: {timeEnd - timeStart} seconds")

    

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
    
    # Make random valid moves to shuffle (using recursion)
    num_moves = random.randint(50, 200)  # Number of random moves to make
    return shuffleGrid(grid, rows, columns, num_moves)

def shuffleGrid(grid, rows, columns, moves_remaining):
    """Recursively shuffle grid by making random valid moves."""
    if moves_remaining == 0:
        return grid
    
    # Find zero position
    zero_i, zero_j = 0, 0
    for i in range(rows):
        for j in range(columns):
            if grid[i][j] == 0:
                zero_i, zero_j = i, j
                break
    
    # Get all possible moves
    possible_moves = []
    if zero_i > 0:
        possible_moves.append("U")  # Move zero up
    if zero_i < rows - 1:
        possible_moves.append("D")  # Move zero down
    if zero_j > 0:
        possible_moves.append("L")  # Move zero left
    if zero_j < columns - 1:
        possible_moves.append("R")  # Move zero right
    
    # Make a random move
    if possible_moves:
        move = random.choice(possible_moves)
        new_grid = copy.deepcopy(grid)
        
        if move == "U":
            new_grid[zero_i][zero_j] = new_grid[zero_i - 1][zero_j]
            new_grid[zero_i - 1][zero_j] = 0
        elif move == "D":
            new_grid[zero_i][zero_j] = new_grid[zero_i + 1][zero_j]
            new_grid[zero_i + 1][zero_j] = 0
        elif move == "L":
            new_grid[zero_i][zero_j] = new_grid[zero_i][zero_j - 1]
            new_grid[zero_i][zero_j - 1] = 0
        elif move == "R":
            new_grid[zero_i][zero_j] = new_grid[zero_i][zero_j + 1]
            new_grid[zero_i][zero_j + 1] = 0
        
        # Recursively continue shuffling
        return shuffleGrid(new_grid, rows, columns, moves_remaining - 1)
    
    return grid

def isSolvable(grid, rows, columns):
    """Check if a sliding puzzle configuration is solvable."""
    # Flatten the grid (excluding 0) and count inversions
    flat = []
    zero_row = 0
    for i in range(rows):
        for j in range(columns):
            if grid[i][j] != 0:
                flat.append(grid[i][j])
            else:
                zero_row = i
    
    # Count inversions (pairs where a larger number comes before a smaller number)
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    
    # For a puzzle to be solvable:
    # - If grid width is odd: inversions must be even
    # - If grid width is even: (inversions + distance of zero from bottom) must be even
    if columns % 2 == 1:
        return inversions % 2 == 0
    else:
        distance_from_bottom = rows - 1 - zero_row
        return (inversions + distance_from_bottom) % 2 == 0

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

def bfs(grid, rows, columns, path, visited):
    """Breadth-first search using a queue to find the shortest solution."""
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
        zero_i, zero_j = 0, 0
        for i in range(rows):
            for j in range(columns):
                if current_grid[i][j] == 0:
                    zero_i, zero_j = i, j
                    break
        
        # Try all possible moves
        moves = []
        if zero_i > 0 and last_move != "D":
            moves.append(("U", zero_i - 1, zero_j))
        if zero_i < rows - 1 and last_move != "U":
            moves.append(("D", zero_i + 1, zero_j))
        if zero_j > 0 and last_move != "R":
            moves.append(("L", zero_i, zero_j - 1))
        if zero_j < columns - 1 and last_move != "L":
            moves.append(("R", zero_i, zero_j + 1))
        
        # Add all valid moves to queue
        for move_dir, new_i, new_j in moves:
            new_grid = copy.deepcopy(current_grid)
            # Swap zero with the adjacent cell
            new_grid[zero_i][zero_j] = new_grid[new_i][new_j]
            new_grid[new_i][new_j] = 0
            queue.append((new_grid, current_path + [move_dir], move_dir))
    
    return "error: No solution found"

if __name__ == "__main__":
    main()