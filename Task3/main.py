import argparse
import random
import copy
from collections import deque
import time
import heapq
import signal
from contextlib import contextmanager


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process some attributes.", add_help=False
    )
    parser.add_argument("--test", action="store_true", help="run algorithm tests")
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

    if args.test:
        test_all_algorithms()
        return
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


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout on Unix systems"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Check if signal is available (Unix systems)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Fallback for Windows - no timeout
        yield


def test_all_algorithms():
    """Test all algorithms on random grids and compare performance."""

    # Configuration
    test_configs = [
        (2, 2, 10),  # 10 grids of 2x2
        (3, 3, 10),  # 10 grids of 3x3
        (4, 4, 10),  # 10 grids of 4x4
    ]

    timeout_seconds = 5

    # Algorithms to test
    algorithms = [
        ("BFS", lambda g, r, c: bfs(g, r, c)),
        ("DFS", lambda g, r, c: dfs(g, r, c)),
        ("IDFS", lambda g, r, c: idfs(g, r, c)),
        ("Best-First (h=0)", lambda g, r, c: best_first(g, r, c, 0)),
        ("Best-First (h=1)", lambda g, r, c: best_first(g, r, c, 1)),
        ("Best-First (h=2)", lambda g, r, c: best_first(g, r, c, 2)),
        ("A* (h=0)", lambda g, r, c: astar(g, r, c, 0)),
        ("A* (h=1)", lambda g, r, c: astar(g, r, c, 1)),
        ("A* (h=2)", lambda g, r, c: astar(g, r, c, 2)),
        ("SMA* (h=0)", lambda g, r, c: sma(g, r, c, 0)),
        ("SMA* (h=1)", lambda g, r, c: sma(g, r, c, 1)),
        ("SMA* (h=2)", lambda g, r, c: sma(g, r, c, 2)),
    ]

    # Generate all test grids
    print("Generating test grids...")
    all_grids = []
    for rows, cols, count in test_configs:
        for i in range(count):
            grid = generateSolvableGrid(rows, cols)
            all_grids.append((rows, cols, grid, f"{rows}x{cols}-{i+1}"))

    print(f"Generated {len(all_grids)} test grids\n")

    # Results storage
    results = {algo_name: [] for algo_name, _ in algorithms}

    # Test each algorithm on each grid
    for grid_idx, (rows, cols, grid, grid_name) in enumerate(all_grids):
        print(f"Testing grid {grid_idx + 1}/{len(all_grids)}: {grid_name}")

        for algo_name, algo_func in algorithms:
            try:
                start_time = time.time()

                # Try to run with timeout
                with time_limit(timeout_seconds):
                    result = algo_func(copy.deepcopy(grid), rows, cols)

                elapsed_time = (time.time() - start_time) * 1000  # in milliseconds

                # Check if solution found
                if result and len(result) > 0 and result != "error: No solution found":
                    results[algo_name].append(
                        {
                            "grid": grid_name,
                            "moves": len(result),
                            "time": elapsed_time,
                            "status": "solved",
                        }
                    )
                else:
                    results[algo_name].append(
                        {
                            "grid": grid_name,
                            "moves": None,
                            "time": elapsed_time,
                            "status": "no solution",
                        }
                    )

            except TimeoutException:
                results[algo_name].append(
                    {
                        "grid": grid_name,
                        "moves": None,
                        "time": timeout_seconds * 1000,  # in milliseconds
                        "status": "timeout",
                    }
                )
            except Exception as e:
                results[algo_name].append(
                    {
                        "grid": grid_name,
                        "moves": None,
                        "time": None,
                        "status": f"error: {str(e)[:20]}",
                    }
                )

        print(f"  Completed grid {grid_name}\n")

    # Print results
    print_results(results, test_configs)


def print_results(results, test_configs):
    """Print comprehensive comparison of algorithm performance."""

    print("\n" + "=" * 100)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 100)

    # Summary statistics by grid size
    for rows, cols, count in test_configs:
        grid_size = f"{rows}x{cols}"
        print(f"\n{grid_size} Grids Summary:")
        print("-" * 100)

        # Header
        print(
            f"{'Algorithm':<20} {'Solved':<10} {'Avg Moves':<12} {'Avg Time (ms)':<15} {'Timeouts':<10}"
        )
        print("-" * 100)

        for algo_name in results.keys():
            # Filter results for this grid size
            grid_results = [
                r for r in results[algo_name] if r["grid"].startswith(grid_size)
            ]

            solved_count = sum(1 for r in grid_results if r["status"] == "solved")
            timeout_count = sum(1 for r in grid_results if r["status"] == "timeout")

            # Calculate averages only for solved instances
            solved_results = [r for r in grid_results if r["status"] == "solved"]
            avg_moves = (
                sum(r["moves"] for r in solved_results) / len(solved_results)
                if solved_results
                else 0
            )
            avg_time = (
                sum(r["time"] for r in solved_results) / len(solved_results)
                if solved_results
                else 0
            )

            print(
                f"{algo_name:<20} {solved_count}/{count:<8} {avg_moves:<12.1f} {avg_time:<15.4f} {timeout_count:<10}"
            )

    # Detailed results table
    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY GRID")
    print("=" * 100)

    # Get all grid names
    grid_names = [r["grid"] for r in results[list(results.keys())[0]]]

    for grid_name in grid_names:
        print(f"\nGrid: {grid_name}")
        print("-" * 100)
        print(f"{'Algorithm':<20} {'Moves':<10} {'Time (ms)':<12} {'Status':<20}")
        print("-" * 100)

        for algo_name in results.keys():
            result = next(r for r in results[algo_name] if r["grid"] == grid_name)

            moves_str = str(result["moves"]) if result["moves"] is not None else "-"
            time_str = f"{result['time']:.4f}" if result["time"] is not None else "-"
            status_str = result["status"]

            print(f"{algo_name:<20} {moves_str:<10} {time_str:<12} {status_str:<20}")

    # Overall winner summary
    print("\n" + "=" * 100)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 100)

    print(
        f"\n{'Algorithm':<20} {'Total Solved':<15} {'Avg Moves':<15} {'Avg Time (ms)':<15}"
    )
    print("-" * 100)

    for algo_name in results.keys():
        all_results = results[algo_name]
        solved_count = sum(1 for r in all_results if r["status"] == "solved")

        solved_results = [r for r in all_results if r["status"] == "solved"]
        avg_moves = (
            sum(r["moves"] for r in solved_results) / len(solved_results)
            if solved_results
            else 0
        )
        avg_time = (
            sum(r["time"] for r in solved_results) / len(solved_results)
            if solved_results
            else 0
        )

        print(
            f"{algo_name:<20} {solved_count}/{len(all_results):<13} {avg_moves:<15.1f} {avg_time:<15.4f}"
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
