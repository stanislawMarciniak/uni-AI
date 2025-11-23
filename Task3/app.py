from flask import Flask, render_template, jsonify, request
from main import (
    generateSolvableGrid,
    bfs,
    dfs,
    idfs,
    best_first,
    astar,
    sma,
    printGrid,
    isSolved,
)
import copy
import time

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    rows = int(data.get("rows", 3))
    columns = int(data.get("columns", 3))

    grid = generateSolvableGrid(rows, columns)
    # Convert grid to list of lists for JSON
    grid_list = [[int(cell) for cell in row] for row in grid]

    return jsonify({"grid": grid_list, "rows": rows, "columns": columns})


@app.route("/solve", methods=["POST"])
def solve():
    data = request.json
    grid = data.get("grid")
    rows = int(data.get("rows"))
    columns = int(data.get("columns"))
    algorithm = data.get("algorithm")
    heuristic = int(data.get("heuristic", 0))  # Default to h=0

    # Convert grid back to list of lists
    grid_list = [[int(cell) for cell in row] for row in grid]

    # Start timing
    start_time = time.time()

    # Call the appropriate algorithm
    try:
        if algorithm == "bfs":
            solution = bfs(grid_list, rows, columns)
        elif algorithm == "dfs":
            solution = dfs(grid_list, rows, columns)
        elif algorithm == "idfs":
            solution = idfs(grid_list, rows, columns)
        elif algorithm == "best_first":
            solution = best_first(grid_list, rows, columns, heuristic)
        elif algorithm == "astar":
            solution = astar(grid_list, rows, columns, heuristic)
        elif algorithm == "sma":
            solution = sma(grid_list, rows, columns, heuristic)
        else:
            return jsonify({"error": "Invalid algorithm"}), 400

        # Calculate execution time
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        # Check if solution is valid
        if isinstance(solution, str) and "error" in solution:
            return jsonify({"error": solution}), 400

        if not solution or len(solution) == 0:
            return jsonify({"error": "No solution found"}), 400

        return jsonify(
            {
                "solution": solution,
                "moves": len(solution),
                "time": round(elapsed_time, 2),
                "algorithm": algorithm,
                "heuristic": (
                    heuristic if algorithm in ["best_first", "astar", "sma"] else None
                ),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Algorithm error: {str(e)}"}), 500


@app.route("/algorithms", methods=["GET"])
def get_algorithms():
    """Return list of available algorithms and their heuristics"""
    algorithms = {
        "uninformed": [
            {"id": "bfs", "name": "Breadth-First Search (BFS)", "hasHeuristic": False},
            {"id": "dfs", "name": "Depth-First Search (DFS)", "hasHeuristic": False},
            {
                "id": "idfs",
                "name": "Iterative Deepening DFS (IDFS)",
                "hasHeuristic": False,
            },
        ],
        "informed": [
            {"id": "best_first", "name": "Best-First Search", "hasHeuristic": True},
            {"id": "astar", "name": "A* Search", "hasHeuristic": True},
            {"id": "sma", "name": "SMA* Search", "hasHeuristic": True},
        ],
        "heuristics": [
            {
                "id": 0,
                "name": "h=0 (Uninformed)",
                "description": "No heuristic guidance",
            },
            {
                "id": 1,
                "name": "h=1 (Misplaced Tiles)",
                "description": "Count of tiles not in goal position",
            },
            {
                "id": 2,
                "name": "h=2 (Manhattan Distance)",
                "description": "Sum of distances from goal positions",
            },
        ],
    }
    return jsonify(algorithms)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
