from flask import Flask, render_template, jsonify, request
from main import generateSolvableGrid, bfs, dfs, idfs, printGrid, isSolved
import copy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    rows = int(data.get('rows', 3))
    columns = int(data.get('columns', 3))
    
    grid = generateSolvableGrid(rows, columns)
    # Convert grid to list of lists for JSON
    grid_list = [[int(cell) for cell in row] for row in grid]
    
    return jsonify({'grid': grid_list, 'rows': rows, 'columns': columns})

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    grid = data.get('grid')
    rows = int(data.get('rows'))
    columns = int(data.get('columns'))
    algorithm = data.get('algorithm')
    
    # Convert grid back to list of lists
    grid_list = [[int(cell) for cell in row] for row in grid]
    
    # Call the appropriate algorithm
    if algorithm == 'bfs':
        solution = bfs(grid_list, rows, columns)
    elif algorithm == 'dfs':
        solution = dfs(grid_list, rows, columns)
    elif algorithm == 'idfs':
        solution = idfs(grid_list, rows, columns)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    if isinstance(solution, str) and 'error' in solution:
        return jsonify({'error': solution}), 400
    
    return jsonify({'solution': solution})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

