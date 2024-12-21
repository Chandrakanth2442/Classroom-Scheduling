from flask import Flask, render_template, jsonify, request
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run/<algorithm>', methods=['POST'])
def run_algorithm(algorithm):
    scripts = {
        'aco': 'ant_colony_optimization.py',
        'genetic': 'genetic_algorithm.py',
        'simulated_annealing': 'simulated_annealing.py',
        'tabu_search': 'tabu_search.py',
         'cp': 'constraint_programming.py',
         'bb': 'branch_and_bound.py'
    }
    script = scripts.get(algorithm)

    if script:
        # Run the script in the background
        subprocess.Popen(['python3', script])

        # Return a success response
        return jsonify({'status': 'success', 'message': f'Running {script}...'})
    else:
        # Return an error response if the algorithm is not valid
        return jsonify({'status': 'error', 'message': 'Invalid algorithm!'}), 404

if __name__ == '__main__':
    app.run(debug=True)
