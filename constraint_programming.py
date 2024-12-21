import random
import copy
import math
from collections import defaultdict
import webbrowser
import os
import matplotlib.pyplot as plt

# Problem definitions
classes = ["Sec-A", "Sec-B", "Sec-C", "Sec-D"]
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
hours = ["h1", "h2", "h3", "h4", "h5", "h6"]

# Redesigned Course Mapping: <course, room_type, lecturer, units>
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

# Define room counts dynamically
rooms = {"Theory": 50, "Lab": 50}

# Function to generate room numbers dynamically
def generate_room_numbers(room_type):
    return [f"{room_type[0].upper()}{i+1}" for i in range(rooms[room_type])]

# Theory and Lab room lists dynamically generated
theory_rooms = generate_room_numbers("Theory")  # T1 to T50
lab_rooms = generate_room_numbers("Lab")        # L1 to L50

# Map course indices to course info
course_indices = list(range(len(course_mapping)))
course_dict = {i: course for i, course in enumerate(course_mapping)}
course_name_to_index = {course[0]: i for i, course in enumerate(course_mapping)}
course_index_to_type = {i: course[1].lower() for i, course in enumerate(course_mapping)}
course_index_to_lecturer = {i: course[2] for i, course in enumerate(course_mapping)}

# Total number of rooms
total_rooms = len(theory_rooms) + len(lab_rooms)
room_indices = list(range(total_rooms))
room_index_to_type = {}
for idx, room in enumerate(theory_rooms):
    room_index_to_type[idx] = "theory"
for idx, room in enumerate(lab_rooms):
    global_idx = idx + len(theory_rooms)
    room_index_to_type[global_idx] = "lab"

# Build room index to name mapping
room_index_to_name = {}
for idx, room in enumerate(theory_rooms):
    room_index_to_name[idx] = room
for idx, room in enumerate(lab_rooms):
    global_idx = idx + len(theory_rooms)
    room_index_to_name[global_idx] = room

# Variables: (class, day, hour)
variables = []
for cls in classes:
    for day in days:
        for hour in hours:
            variables.append((cls, day, hour))

# Domains for each variable
domains = {}
for var in variables:
    cls, day, hour = var
    domains[var] = course_indices.copy()

# Neighbors: Variables that share a constraint with each variable
neighbors = defaultdict(set)

# Build neighbors
for var1 in variables:
    cls1, day1, hour1 = var1
    for var2 in variables:
        if var1 == var2:
            continue
        cls2, day2, hour2 = var2
        # Same time and room constraints
        if day1 == day2 and hour1 == hour2:
            neighbors[var1].add(var2)
        # Same class constraints
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
    type1 = course1[1].lower()
    type2 = course2[1].lower()

    # Lecturers cannot teach more than one class at the same time
    if day1 == day2 and hour1 == hour2:
        if lecturer1 == lecturer2:
            return False  # Conflict

    # Same class constraints
    if cls1 == cls2:
        # No same course in the same class and same day
        if day1 == day2:
            if val1 == val2:
                return False
            # Do not schedule the same course in consecutive time slots
            hour_index1 = hours.index(hour1)
            hour_index2 = hours.index(hour2)
            if abs(hour_index1 - hour_index2) == 1 and val1 == val2:
                return False
        
        # Prevent the same course from appearing in the same hour slot on different days for the same class
        if hour1 == hour2 and val1 == val2 and day1 != day2:
            return False

    return True

# Implement AC-3 algorithm
def ac3(variables, domains, neighbors):
    queue = [(xi, xj) for xi in variables for xj in neighbors[xi]]
    while queue:
        xi, xj = queue.pop(0)
        if revise(xi, xj, domains, constraints):
            if not domains[xi]:
                return False  # Failure due to empty domain
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

# Helper functions
def select_unassigned_variable(variables, assignment, domains):
    unassigned_vars = [v for v in variables if v not in assignment]
    return min(unassigned_vars, key=lambda var: len(domains[var]))

def order_domain_values(var, domains):
    return domains[var]

def is_consistent(var, value, assignment, constraints):
    for var2 in assignment:
        val2 = assignment[var2]
        if not constraints(var, value, var2, val2):
            return False
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

def constraint_programming_schedule():
    if not ac3(variables, domains, neighbors):
        print("No solution found after initial AC-3.")
        return None

    assignment = {}
    result = backtrack(assignment, variables, domains, neighbors)
    if result:
        print("Solution Found!")
        state = defaultdict(lambda: defaultdict(dict))
        for var in variables:
            cls, day, hour = var
            course_idx = result[var]
            course = course_dict[course_idx]
            course_name = course[0]
            room_type = course[1].lower()
            lecturer = course[2]
            units = course[3]
            room = None
            if room_type == "theory":
                room = random.choice(theory_rooms)
            else:
                room = random.choice(lab_rooms)
            state[cls][day][hour] = (course_name, room_type, lecturer, units, room)
        return state
    else:
        print("No solution found.")
        return None

def generate_html(state):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Timetables</title>
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
        html_content += '<table>'
        html_content += '<tr><th>Hour</th>' + ''.join(f'<th>{day.capitalize()}</th>' for day in days) + '</tr>'
        
        for hour in hours:
            html_content += f'<tr><td>{hour.upper()}</td>'
            for day in days:
                if hour in state[class_name][day]:
                    course_info = state[class_name][day][hour]
                    course_name, room_type, lecturer, units, room = course_info
                    
                    # Combine subject name, lecturer, and room number into a single line
                    room_display = room if room != "None" else "Unassigned"
                    subject_lecturer_room = f"{course_name} ({room_type.capitalize()}) | Lecturer: {lecturer} | Room: {room_display}"
                    
                    html_content += f'<td>{subject_lecturer_room}</td>'
                else:
                    html_content += '<td>---</td>'  # Placeholder for empty slots
            html_content += '</tr>'
        html_content += '</table>'

    html_content += """
    </body>
    </html>
    """

    filename = "schedule.html"
    with open(filename, "w") as file:
        file.write(html_content)

    path = os.path.abspath(filename)
    webbrowser.open_new_tab(f"file://{path}")

    print(f"Styled HTML page with timetables generated and opened: {filename}")


def main():
    final_state = constraint_programming_schedule()
    if final_state:
        generate_html(final_state)
    else:
        print("Failed to generate a valid schedule.")

if __name__ == "__main__":
    main()
