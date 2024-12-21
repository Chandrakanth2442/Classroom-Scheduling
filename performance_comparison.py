import time
import random
import copy
import math
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import webbrowser
import os

# Problem definitions
classes = ["Sec-A", "Sec-B", "Sec-C", "Sec-D"]
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
hours = ["h1", "h2", "h3", "h4", "h5", "h6"]

# Course Mapping: <course, room_type, lecturer, units>
course_mapping = [
    ["Cloud Computing", "Lab", "l1", 1],
    ["Artificial Intelligence", "Lab", "l1", 1],
    ["Computer Networks", "Lab", "l1", 1],
    ["Data Management", "Theory", "l1", 1],
    ["Data Structures and Algorithms", "Lab", "l2", 1],
    ["Design and Analysis of Algorithms", "Lab", "l2", 1],
    ["Linear Algebra", "Theory", "l2", 1],
    ["Computer Design", "Theory", "l2", 1],
    ["Advanced Calculus", "Theory", "l3", 1],
    ["English", "Theory", "l3", 1],
    ["Physics", "Theory", "l3", 1],
    ["Chemistry", "Theory", "l3", 1],
    ["Biology", "Theory", "l4", 1],
    ["Environmental Science", "Theory", "l4", 1],
    ["ADA", "Theory", "l4", 1],
    ["Operating Systems", "Theory", "l4", 1],
    ["Database Management Systems", "Theory", "l5", 1],
    ["Unified Modelling Language", "Theory", "l5", 1],
    ["FLAT", "Theory", "l5", 1],
    ["Software Testing Methodologies", "Theory", "l5", 1],
    ["Big Data", "Theory", "l6", 1],
    ["Mefa", "Theory", "l6", 1],
    ["Operating Systems Lab", "Lab", "l6", 1],
    ["Database Management Systems Lab", "Lab", "l6", 1],
    ["Physics Lab", "Lab", "l7", 1],
    ["Chemistry Lab", "Lab", "l7", 1],
    ["Biology Lab", "Lab", "l7", 1],
    ["Data Structures and Algorithms Lab", "Lab", "l8", 1],
]

# Room counts
rooms_count = {"theory": 50, "lab": 50}
theory_rooms = [f"T{i+1}" for i in range(rooms_count["theory"])]
lab_rooms = [f"L{i+1}" for i in range(rooms_count["lab"])]

course_indices = list(range(len(course_mapping)))
course_dict = {i: course for i, course in enumerate(course_mapping)}
course_name_to_index = {course[0]: i for i, course in enumerate(course_mapping)}
course_index_to_type = {i: course[1].lower() for i, course in enumerate(course_mapping)}
course_index_to_lecturer = {i: course[2] for i, course in enumerate(course_mapping)}

variables = []
for cls in classes:
    for day in days:
        for hour in hours:
            variables.append((cls, day, hour))

domains = {}
for var in variables:
    cls, day, hour = var
    domains[var] = course_indices.copy()

neighbors = defaultdict(set)
for var1 in variables:
    cls1, day1, hour1 = var1
    for var2 in variables:
        if var1 == var2:
            continue
        cls2, day2, hour2 = var2
        if day1 == day2 and hour1 == hour2:
            neighbors[var1].add(var2)
        if cls1 == cls2:
            if day1 == day2:
                neighbors[var1].add(var2)

def constraints(var1, val1, var2, val2):
    cls1, day1, hour1 = var1
    cls2, day2, hour2 = var2
    course1 = course_dict[val1]
    course2 = course_dict[val2]
    lecturer1 = course1[2]
    lecturer2 = course2[2]

    # Lecturers cannot teach more than one class at the same time
    if day1 == day2 and hour1 == hour2:
        if lecturer1 == lecturer2:
            return False
    # Same class constraints
    if cls1 == cls2:
        if day1 == day2:
            # Do not schedule the same course more than once per day
            if val1 == val2:
                return False
            hour_index1 = hours.index(hour1)
            hour_index2 = hours.index(hour2)
            if abs(hour_index1 - hour_index2) == 1 and val1 == val2:
                return False
    return True

def ac3(variables, domains, neighbors):
    queue = [(xi, xj) for xi in variables for xj in neighbors[xi]]
    while queue:
        xi, xj = queue.pop(0)
        if revise(xi, xj, domains, constraints):
            if not domains[xi]:
                return False
            for xk in neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(xi, xj, domains, constraints):
    revised = False
    for x in domains[xi][:]:
        if not any(constraints(xi, x, xj, y) for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

def select_unassigned_variable(variables, assignment, domains):
    # MRV heuristic
    unassigned_vars = [v for v in variables if v not in assignment]
    return min(unassigned_vars, key=lambda var: len(domains[var]))

def order_domain_values(var, domains):
    return domains[var]

def is_consistent(var, value, assignment, constraints):
    cls, day, hour = var
    course_idx = value
    course_info = course_dict[course_idx]
    room_type = course_info[1].lower()
    
    for var2 in assignment:
        val2 = assignment[var2]
        if not constraints(var, value, var2, val2):
            return False
    
    # Cumulative constraints
    count = 0
    for var2 in assignment:
        cls2, day2, hour2 = var2
        if cls2 == cls and assignment[var2] == course_idx:
            count +=1
    count +=1
    if "lab" in room_type:
        if count > 1:
            return False  # Labs scheduled more than once a week
    else:
        if count > 4:
            return False  # Theory courses more than 4 times a week
    return True

def backtrack(assignment, variables, domains, neighbors):
    if len(assignment) == len(variables):
        return assignment
    var = select_unassigned_variable(variables, assignment, domains)
    for value in order_domain_values(var, domains):
        if is_consistent(var, value, assignment, constraints):
            assignment[var] = value
            local_domains = copy.deepcopy(domains)
            local_domains[var] = [value]
            if ac3(variables, local_domains, neighbors):
                result = backtrack(assignment, variables, local_domains, neighbors)
                if result:
                    return result
            del assignment[var]
    return None

def constraint_programming():
    start_time = time.time()
    local_domains = copy.deepcopy(domains)
    if not ac3(variables, local_domains, neighbors):
        print("No solution found after initial AC-3.")
        total_time = time.time() - start_time
        return None, None, total_time, [], []
    assignment = {}
    result = backtrack(assignment, variables, local_domains, neighbors)
    total_time = time.time() - start_time
    if result:
        state = defaultdict(lambda: defaultdict(dict))
        for var in variables:
            cls, day, hour = var
            course_idx = result[var]
            course = course_dict[course_idx]
            course_name = course[0]
            room_type = course[1].lower()
            lecturer = course[2]
            units = course[3]

            # Assign a random room
            if room_type == "theory":
                room = random.choice(theory_rooms)
            else:
                room = random.choice(lab_rooms)

            state[cls][day][hour] = (course_name, room_type, lecturer, units, room)
        cost = calculate_cost(state)
        cost_progress = [cost]
        time_progress = [total_time]
        return state, cost, total_time, cost_progress, time_progress
    else:
        print("No solution found.")
        cost_progress = []
        time_progress = []
        return None, None, total_time, cost_progress, time_progress

def generate_initial_state():
    state = defaultdict(lambda: defaultdict(dict))
    for sec in classes:
        for day in days:
            for hour in hours:
                course = random.choice(course_mapping)
                course_name, room_type, lecturer, units = course

                if room_type.lower() == "theory":
                    room = random.choice(theory_rooms)
                else:
                    room = random.choice(lab_rooms)

                state[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
    return state

def calculate_cost(state):
    cost = 0
    lecturer_schedule = defaultdict(lambda: defaultdict(int))
    room_schedule = defaultdict(set)
    course_hours_per_week = defaultdict(lambda: defaultdict(int))
    course_hours_per_day = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for sec, days_dict in state.items():
        for day, hours_dict in days_dict.items():
            for hour, details in hours_dict.items():
                course_name, room_type, lecturer, units, room_number = details

                lecturer_schedule[lecturer][day] += 1
                if lecturer_schedule[lecturer][day] > 4:
                    cost += 1

                course_hours_per_week[sec][course_name] += 1
                course_hours_per_day[sec][day][course_name] += 1
                if course_hours_per_day[sec][day][course_name] > 1:
                    cost += 1

                if room_number != "NA":
                    if room_number in room_schedule[(day, hour)]:
                        cost += 1
                    else:
                        room_schedule[(day, hour)].add(room_number)
                else:
                    cost += 5

    for sec, courses in course_hours_per_week.items():
        for course, hrs in courses.items():
            if "lab" in course.lower():
                if hrs != 1:
                    cost += 1
            else:
                if hrs > 3:
                    cost += 1
    return cost

class Node:
    def __init__(self, state, cost, level):
        self.state = state
        self.cost = cost
        self.level = level
    def __lt__(self, other):
        return self.cost < other.cost

def generate_children(node):
    children = []
    level = node.level + 1

    for sec in classes:
        for day in days:
            for hour in hours:
                new_state = copy.deepcopy(node.state)
                course = random.choice(course_mapping)
                course_name, room_type, lecturer, units = course

                if room_type.lower() == "theory":
                    room = random.choice(theory_rooms)
                else:
                    room = random.choice(lab_rooms)

                new_state[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
                new_cost = calculate_cost(new_state)
                if new_cost <= node.cost:
                    children.append(Node(new_state, new_cost, level))
    return children

def branch_and_bound():
    start_time = time.time()
    initial_state = generate_initial_state()
    initial_cost = calculate_cost(initial_state)
    best_cost = initial_cost
    best_state = initial_state

    pq = PriorityQueue()
    pq.put(Node(initial_state, initial_cost, 0))

    max_iterations = 100
    iterations = 0

    cost_progress = []
    time_progress = []

    while not pq.empty() and iterations < max_iterations:
        node = pq.get()
        cost_progress.append(node.cost)
        time_progress.append(time.time() - start_time)

        if node.cost == 0:
            total_time = time.time() - start_time
            return node.state, node.cost, total_time, cost_progress, time_progress

        children = generate_children(node)
        for child in children:
            if child.cost < best_cost:
                best_cost = child.cost
                best_state = child.state
            pq.put(child)
        iterations += 1

    total_time = time.time() - start_time
    return best_state, best_cost, total_time, cost_progress, time_progress

def ant_colony_optimization():
    NUM_ANTS = 10
    NUM_ITERATIONS = 100
    EVAPORATION_RATE = 0.1
    INITIAL_PHEROMONE = 1.0

    pheromones = defaultdict(lambda: INITIAL_PHEROMONE)
    heuristics = {course[0]: 1.0 for course in course_mapping}

    best_cost = float('inf')
    best_state = None

    def choose_course(available_courses, sec, day, hour):
        weights = [pheromones[course[0]] * heuristics[course[0]] for course in available_courses]
        choice = random.choices(available_courses, weights=weights, k=1)
        return choice[0]

    def update_pheromones(ants_schedules):
        nonlocal best_cost, best_state
        for key in pheromones:
            pheromones[key] *= (1 - EVAPORATION_RATE)
        for schedule_info in ants_schedules:
            schedule = schedule_info['schedule']
            cost = schedule_info['cost']
            if cost < best_cost:
                best_cost = cost
                best_state = schedule
            for sec in classes:
                for day in days:
                    for hour in hours:
                        course = schedule[sec][day][hour]
                        pheromones[course[0]] += 1 / (cost + 1)

    def create_ant_schedule():
        schedule = defaultdict(lambda: defaultdict(dict))
        for sec in classes:
            for day in days:
                for hour in hours:
                    course = choose_course(course_mapping, sec, day, hour)
                    course_name, room_type, lecturer, units = course
                    if room_type.lower() == "theory":
                        room = random.choice(theory_rooms)
                    else:
                        room = random.choice(lab_rooms)
                    schedule[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
        return schedule

    start_time = time.time()
    for iteration in range(NUM_ITERATIONS):
        ants_schedules = []
        for _ in range(NUM_ANTS):
            schedule = create_ant_schedule()
            cost = calculate_cost(schedule)
            ants_schedules.append({'schedule': schedule, 'cost': cost})
        update_pheromones(ants_schedules)
    end_time = time.time()
    total_time = end_time - start_time

    return best_state, best_cost, total_time, [best_cost], [total_time]

def simulated_annealing():
    def generate_neighbor(state):
        neighbor = copy.deepcopy(state)
        sec = random.choice(classes)
        day = random.choice(days)
        hour = random.choice(hours)
        course = random.choice(course_mapping)
        course_name, room_type, lecturer, units = course
        if room_type.lower() == "theory":
            room = random.choice(theory_rooms)
        else:
            room = random.choice(lab_rooms)
        neighbor[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
        return neighbor

    current_state = generate_initial_state()
    current_cost = calculate_cost(current_state)
    best_state = current_state
    best_cost = current_cost
    temp = 100
    cooling_rate = 0.99
    max_iter = 100

    start_time = time.time()
    cost_progress = []
    time_progress = []

    for iteration in range(max_iter):
        if temp <= 0:
            break
        neighbor = generate_neighbor(current_state)
        neighbor_cost = calculate_cost(neighbor)
        if neighbor_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - neighbor_cost) / temp):
            current_state = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state
        temp *= cooling_rate
        cost_progress.append(current_cost)
        time_progress.append(time.time() - start_time)
    end_time = time.time()
    total_time = end_time - start_time

    return best_state, best_cost, total_time, cost_progress, time_progress

def genetic_algorithm():
    def create_population(pop_size):
        return [generate_initial_state() for _ in range(pop_size)]

    def selection(population, fitnesses, num_parents):
        selected_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:num_parents]
        return [population[i] for i in selected_indices]

    def crossover(parent1, parent2):
        child = defaultdict(lambda: defaultdict(dict))
        for sec in classes:
            for day in days:
                for hour in hours:
                    if random.random() < 0.5:
                        child[sec][day][hour] = parent1[sec][day][hour]
                    else:
                        child[sec][day][hour] = parent2[sec][day][hour]
        return child

    def mutate(state, mutation_rate=0.1):
        for sec in classes:
            for day in days:
                for hour in hours:
                    if random.random() < mutation_rate:
                        course = random.choice(course_mapping)
                        course_name, room_type, lecturer, units = course
                        if room_type.lower() == "theory":
                            room = random.choice(theory_rooms)
                        else:
                            room = random.choice(lab_rooms)
                        state[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
        return state

    pop_size = 50
    generations = 100
    num_parents = 10
    mutation_rate = 0.1
    population = create_population(pop_size)
    best_state = None
    best_cost = float('inf')

    start_time = time.time()
    cost_progress = []
    time_progress = []

    for generation in range(generations):
        fitnesses = [calculate_cost(state) for state in population]
        current_best_cost = min(fitnesses)
        current_time = time.time() - start_time
        cost_progress.append(current_best_cost)
        time_progress.append(current_time)

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_index = fitnesses.index(current_best_cost)
            best_state = population[best_index]
        if best_cost == 0:
            break
        parents = selection(population, fitnesses, num_parents)
        children = []
        while len(children) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            children.append(child)
        population = children
    end_time = time.time()
    total_time = end_time - start_time

    return best_state, best_cost, total_time, cost_progress, time_progress

def tabu_search():
    def generate_neighborhood(state, neighborhood_size=100):
        neighborhood = []
        for _ in range(neighborhood_size):
            neighbor = copy.deepcopy(state)
            sec = random.choice(list(neighbor.keys()))
            day = random.choice(list(neighbor[sec].keys()))
            hour = random.choice(list(neighbor[sec][day].keys()))
            old_course = neighbor[sec][day][hour]
            course = random.choice(course_mapping)
            course_name, room_type, lecturer, units = course
            if room_type.lower() == "theory":
                room = random.choice(theory_rooms)
            else:
                room = random.choice(lab_rooms)
            neighbor[sec][day][hour] = (course_name, room_type.lower(), lecturer, units, room)
            move = (sec, day, hour, old_course[0], course_name)
            neighborhood.append((neighbor, move))
        return neighborhood

    current_state = generate_initial_state()
    current_cost = calculate_cost(current_state)
    best_state = current_state
    best_cost = current_cost

    tabu_list = []
    tabu_list_size = 100
    max_iter = 100

    start_time = time.time()
    cost_progress = []
    time_progress = []

    for iteration in range(1, max_iter + 1):
        neighborhood = generate_neighborhood(current_state)
        candidate_moves = []
        for neighbor, move in neighborhood:
            neighbor_cost = calculate_cost(neighbor)
            candidate_moves.append((neighbor, neighbor_cost, move))

        candidate_moves.sort(key=lambda x: x[1])
        for neighbor, neighbor_cost, move in candidate_moves:
            if move not in tabu_list or neighbor_cost < best_cost:
                current_state = neighbor
                current_cost = neighbor_cost
                if neighbor_cost < best_cost:
                    best_state = neighbor
                    best_cost = neighbor_cost
                tabu_list.append(move)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)
                break

        cost_progress.append(current_cost)
        time_progress.append(time.time() - start_time)

    end_time = time.time()
    total_time = end_time - start_time

    return best_state, best_cost, total_time, cost_progress, time_progress

def generate_html(state, filename):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body { font-family: Arial, sans-serif; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #2e1a47; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
    </head>
    <body>
    <h1>Class Timetables</h1>
    """

    for class_name in classes:
        html_content += f'<h2>Timetable for {class_name}</h2>'
        data = []
        for day in days:
            row = []
            for hour in hours:
                if hour in state[class_name][day]:
                    course_info = state[class_name][day][hour]
                    if len(course_info) == 5:
                        course_name, room_type, lecturer, units, room = course_info
                        row.append(f"{course_name} ({room_type}, {lecturer}, {room})")
                    else:
                        course_name, room_type, lecturer, units = course_info
                        row.append(f"{course_name} ({room_type}, {lecturer})")
                else:
                    row.append("NaN")
            timetable = pd.DataFrame([row], index=[day], columns=hours)
            data.append(row)
        timetable = pd.DataFrame(data, index=days, columns=hours)
        html_content += timetable.to_html(classes='table', border=0)

    html_content += '</body></html>'

    with open(filename, "w") as html_file:
        html_file.write(html_content)

    print(f"Styled HTML page with timetables generated: {filename}")

def plot_costs(df):
    algorithms = df['Algorithm']
    avg_cost = df['Avg Cost']
    std_cost = df['Std Cost']
    
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, avg_cost, yerr=std_cost, capsize=5)
    plt.title('Average and Standard Deviation of Cost for Each Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_times(df):
    algorithms = df['Algorithm']
    avg_time = df['Avg Time (s)']
    std_time = df['Std Time (s)']
    
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, avg_time, yerr=std_time, capsize=5, color='orange')
    plt.title('Average and Standard Deviation of Time for Each Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_cost_vs_time(df):
    algorithms = df['Algorithm']
    avg_cost = df['Avg Cost']
    avg_time = df['Avg Time (s)']

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_time, avg_cost, color='green', s=100)

    for i, alg in enumerate(algorithms):
        plt.annotate(alg, (avg_time[i], avg_cost[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Average Cost vs Average Time for Each Algorithm')
    plt.xlabel('Average Time (s)')
    plt.ylabel('Average Cost')
    plt.grid(True)
    plt.show()

def plot_cost_vs_time_progress(cost_time_data):
    plt.figure(figsize=(10, 6))
    for alg_name, data in cost_time_data.items():
        plt.plot(data['time'], data['cost'], label=alg_name)
    plt.title('Cost vs Time Progression for Each Algorithm')
    plt.xlabel('Time (s)')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    results = []
    num_runs = 5

    algorithms = [
        ('Constraint Programming', constraint_programming, 'AC-3 and Backtracking', 'Exact'),
        ('Branch and Bound', branch_and_bound, 'Fixed Iterations', 'Exact'),
        ('Ant Colony Optimization', ant_colony_optimization, 'Fixed Iterations', 'Heuristic'),
        ('Simulated Annealing', simulated_annealing, 'Fixed Iterations', 'Heuristic'),
        ('Genetic Algorithm', genetic_algorithm, 'Fixed Iterations', 'Heuristic'),
        ('Tabu Search', tabu_search, 'Fixed Iterations', 'Heuristic'),
    ]

    cost_time_data = {}

    for alg_name, alg_func, stopping_time, nature in algorithms:
        costs = []
        times = []
        cost_runs = []
        time_runs = []
        print(f"\nRunning {alg_name}...")
        for run in range(num_runs):
            state, cost, total_time, cost_progress, time_progress = alg_func()
            if state is None:
                continue
            costs.append(cost)
            times.append(total_time)
            cost_runs.append(cost_progress)
            time_runs.append(time_progress)
            # Generate HTML for the last run of each algorithm
            generate_html(state, f"timetable_{alg_name.replace(' ', '_').lower()}.html")

        if len(costs) == 0:
            avg_cost = None
            std_cost = None
            avg_time = None
            std_time = None
        else:
            avg_cost = np.mean(costs)
            std_cost = np.std(costs)
            avg_time = np.mean(times)
            std_time = np.std(times)

        if len(cost_runs) > 0:
            # Find the minimum length of runs
            min_length = min(len(cp) for cp in cost_runs)
            truncated_cost_runs = [cp[:min_length] for cp in cost_runs]
            truncated_time_runs = [tp[:min_length] for tp in time_runs]

            avg_cost_progress = np.mean(truncated_cost_runs, axis=0)
            avg_time_progress = np.mean(truncated_time_runs, axis=0)

            cost_time_data[alg_name] = {'cost': avg_cost_progress, 'time': avg_time_progress}

        results.append({
            'Algorithm': alg_name,
            'Stopping Time': stopping_time,
            'Nature': nature,
            'Avg Cost': avg_cost,
            'Std Cost': std_cost,
            'Avg Time (s)': avg_time,
            'Std Time (s)': std_time,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    plot_costs(df)
    plot_times(df)
    plot_cost_vs_time(df)
    if cost_time_data:
        plot_cost_vs_time_progress(cost_time_data)

if __name__ == "__main__":
    main()
