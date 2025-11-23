import argparse
import random
import copy
from collections import deque
import time
import heapq


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process some attributes.", add_help=False
    )
    parser.add_argument("-r", "--rows", type=int, help="row count")
    parser.add_argument("-c", "--columns", type=int, help="column count")
    parser.add_argument("-b", "--bfs", action="store_true", help="breadth first search")
    parser.add_argument("-d", "--dfs", action="store_true", help="depth first search")
    parser.add_argument(
        "-i", "--idfs", action="store_true", help="iterative deepening DFS"
    )
    parser.add_argument(
        "-h", "--bf", type=int, metavar="id_of_heuristic", help="best-first strategy"
    )
    parser.add_argument(
        "-a", "--astar", type=int, metavar="id_of_heuristic", help="A* strategy"
    )
    parser.add_argument(
        "-s", "--sma", type=int, metavar="id_of_heuristic", help="SMA* strategy"
    )
    parser.add_argument(
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    args = parser.parse_args()

    # Default values if not specified on input
    print()
    if args.rows is None:
        print(f"row count not specified - defaulting to 4")
        args.rows = 3
    if args.columns is None:
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
        for j in range(columns):
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
        grid[zero_i][zero_j], grid[piece_i][piece_j] = (
            grid[piece_i][piece_j],
            grid[zero_i][zero_j],
        )
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
    """Depth-first search

    DFS behavior:
    1. Pick a direction (move), keep going deeper until you hit a dead end
    2. When you hit a dead end (visited state or no solution), backtrack
    3. Try the next direction from the previous state
    4. Repeat until solution found or all paths exhausted

    """
    visited = set()

    def dfs_recursive(current_grid, current_path, last_move):
        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path

        # Convert grid to tuple for visited set
        grid_tuple = tuple(tuple(row) for row in current_grid)

        # Dead end: already visited this state - backtrack
        if grid_tuple in visited:
            return None

        # Mark as visited to prevent cycles
        visited.add(grid_tuple)

        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)

        # Get all legal moves from current state
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

        # Try each move one at a time, going deep on first path
        # If first path fails, backtrack and try next move
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            # Make the move
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0

            # recursively explore this path completely
            # This will keep going deeper until it hits a dead end or finds solution
            result = dfs_recursive(new_grid, current_path + [move_dir], move_dir)
            if result is not None:
                return result  # Solution found

        # BACKTRACK
        return None

    result = dfs_recursive(grid, [], "")
    if result is None:
        return "error: No solution found"
    return result


def idfs(grid, rows, columns):
    """Iterative Deepening Depth-First Search (IDDFS).

    Combines the memory efficiency of DFS with the optimality of BFS.
    Repeatedly runs DFS with increasing depth limits until solution is found.
    Guarantees shortest solution like BFS, but uses less memory.

    Classic IDDFS: Only tracks states along current path (recursion stack),
    not globally. This allows reaching same state via different branches for optimality.
    """

    def dfs_with_depth_limit(
        current_grid, current_path, last_move, depth_limit, path_states
    ):
        """DFS with a depth limit - only tracks states on current path to avoid cycles."""
        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path

        # reached the depth limit
        if len(current_path) >= depth_limit:
            return None

        # Convert grid to tuple
        grid_tuple = tuple(tuple(row) for row in current_grid)

        # heck if this state is already on the current path
        if grid_tuple in path_states:
            return None

        # Add to current path states
        path_states.add(grid_tuple)

        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)

        # Get all legal moves
        moves = []
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))

        # Try each move one at a time, going deep (but limited by depth_limit)
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0

            # explore this path
            result = dfs_with_depth_limit(
                new_grid, current_path + [move_dir], move_dir, depth_limit, path_states
            )
            if result is not None:
                return result

        # Remove from path states when backtracking (state no longer on current path)
        path_states.remove(grid_tuple)
        return None

    # IDDFS: Start with depth 0, keep increasing until solution found
    depth = 0
    while True:
        # Start with empty path states for each depth iteration
        path_states = set()
        result = dfs_with_depth_limit(grid, [], "", depth, path_states)
        if result is not None:
            return result
        depth += 1


def manhattan_distance(grid, rows, columns):
    distance = 0
    for i in range(rows):
        for j in range(columns):
            tile = grid[i][j]
            if tile != 0:  # Skip empty space
                # Calculate goal position for this tile
                goal_i = (tile - 1) // columns
                goal_j = (tile - 1) % columns
                # Add Manhattan distance
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def misplaced_tiles(grid, rows, columns):
    count = 0
    for i in range(rows):
        for j in range(columns):
            expected = i * columns + j + 1
            if i == rows - 1 and j == columns - 1:
                # Last position should be 0
                if grid[i][j] != 0:
                    count += 1
            else:
                if grid[i][j] != expected and grid[i][j] != 0:
                    count += 1
    return count


def get_heuristic(grid, rows, columns, heuristic_id):
    """Get heuristic value based on heuristic ID.

    heuristic_id:
    0 - h(n) = 0 (uninformed search)
    1 - Misplaced tiles count
    2 - Manhattan distance
    """
    if heuristic_id == 0:
        return 0  # h(n) = 0, turns A* into Dijkstra's algorithm
    elif heuristic_id == 1:
        return misplaced_tiles(grid, rows, columns)
    elif heuristic_id == 2:
        return manhattan_distance(grid, rows, columns)
    else:
        raise ValueError(f"Unknown heuristic ID: {heuristic_id}")


def best_first(grid, rows, columns, heuristic_id):
    # Priority queue stores (h_value, counter, grid, path, last_move)
    # counter breaks ties consistently
    counter = 0
    h_initial = get_heuristic(grid, rows, columns, heuristic_id)
    pq = [(h_initial, counter, grid, [], "")]
    counter += 1
    visited = set()

    while pq:
        h_value, _, current_grid, current_path, last_move = heapq.heappop(pq)

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
        moves = []
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))

        # Add all valid moves to priority queue
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0

            # Calculate heuristic for new state
            h_new = get_heuristic(new_grid, rows, columns, heuristic_id)
            heapq.heappush(
                pq, (h_new, counter, new_grid, current_path + [move_dir], move_dir)
            )
            counter += 1

    return []  # No solution found


def astar(grid, rows, columns, heuristic_id):
    # Priority queue stores (f_value, counter, g_value, grid, path, last_move)
    counter = 0
    g_initial = 0
    h_initial = get_heuristic(grid, rows, columns, heuristic_id)
    f_initial = g_initial + h_initial
    pq = [(f_initial, counter, g_initial, grid, [], "")]
    counter += 1
    visited = {}  # Maps state to best g-value seen

    while pq:
        f_value, _, g_value, current_grid, current_path, last_move = heapq.heappop(pq)

        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path

        # Convert grid to tuple for visited set
        grid_tuple = tuple(tuple(row) for row in current_grid)

        # Skip if we've seen this state with a better or equal g-value
        if grid_tuple in visited and visited[grid_tuple] <= g_value:
            continue

        # Mark as visited with current g-value
        visited[grid_tuple] = g_value

        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)

        # Try all possible moves
        moves = []
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))

        # Add all valid moves to priority queue
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0

            # Calculate g and h for new state
            g_new = g_value + 1  # Each move costs 1
            h_new = get_heuristic(new_grid, rows, columns, heuristic_id)
            f_new = g_new + h_new

            heapq.heappush(
                pq,
                (f_new, counter, g_new, new_grid, current_path + [move_dir], move_dir),
            )
            counter += 1

    return []  # No solution found


def sma(grid, rows, columns, heuristic_id, max_nodes=50000):
    # Priority queue stores (f_value, counter, g_value, grid, path, last_move)
    counter = 0
    g_initial = 0
    h_initial = get_heuristic(grid, rows, columns, heuristic_id)
    f_initial = g_initial + h_initial
    pq = [(f_initial, counter, g_initial, grid, [], "")]
    counter += 1
    visited = {}  # Maps state to best g-value seen

    while pq:
        # Memory management: if queue too large, prune worst nodes
        if len(pq) > max_nodes:
            # Keep only the best max_nodes nodes
            pq.sort()  # Sort by f-value (first element of tuple)
            pq = pq[:max_nodes]
            heapq.heapify(pq)

        f_value, _, g_value, current_grid, current_path, last_move = heapq.heappop(pq)

        # Check if solved
        if isSolved(current_grid, rows, columns):
            return current_path

        # Convert grid to tuple for visited set
        grid_tuple = tuple(tuple(row) for row in current_grid)

        # Skip if we've seen this state with a better or equal g-value
        if grid_tuple in visited and visited[grid_tuple] <= g_value:
            continue

        # Mark as visited with current g-value
        visited[grid_tuple] = g_value

        # Find zero position
        zero_i, zero_j = find_zero(current_grid, rows, columns)

        # Try all possible moves
        moves = []
        if zero_j < columns - 1 and last_move != "R":
            moves.append(("L", zero_i, zero_j + 1))
        if zero_j > 0 and last_move != "L":
            moves.append(("R", zero_i, zero_j - 1))
        if zero_i < rows - 1 and last_move != "D":
            moves.append(("U", zero_i + 1, zero_j))
        if zero_i > 0 and last_move != "U":
            moves.append(("D", zero_i - 1, zero_j))

        # Add all valid moves to priority queue
        for move_dir, piece_i, piece_j in moves:
            new_grid = copy.deepcopy(current_grid)
            new_grid[zero_i][zero_j] = new_grid[piece_i][piece_j]
            new_grid[piece_i][piece_j] = 0

            # Calculate g and h for new state
            g_new = g_value + 1
            h_new = get_heuristic(new_grid, rows, columns, heuristic_id)
            f_new = g_new + h_new

            heapq.heappush(
                pq,
                (f_new, counter, g_new, new_grid, current_path + [move_dir], move_dir),
            )
            counter += 1

    return []


if __name__ == "__main__":
    main()
