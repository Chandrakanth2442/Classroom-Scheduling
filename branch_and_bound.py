import random
import copy
import math
from collections import defaultdict
import pandas as pd  
from queue import PriorityQueue
from pathlib import Path
import webbrowser

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
rooms = {"Theory": 50, "Lab": 50}

def calculate_cost(state):
    cost = 0
    lecturer_schedule = defaultdict(lambda: defaultdict(int))      # lecturer_schedule[lecturer][day] = hours per day
    room_schedule = defaultdict(set)  # room_schedule[(day, hour)] = set of rooms used
    course_hours_per_week = defaultdict(lambda: defaultdict(int))  # course_hours_per_week[section][course_name] = times per week
    course_hours_per_day = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # course_hours_per_day[section][day][course_name] = times this course appears in that section on that day

    for sec, days_dict in state.items():
        for day, hours_dict in days_dict.items():
            for hour, details in hours_dict.items():
                course_name, room_type, lecturer, units, room_number = details

                # Lecturer daily teaching hours limit
                lecturer_schedule[lecturer][day] += 1
                if lecturer_schedule[lecturer][day] > 4:
                    cost += 1  # Exceeds max teaching hours per day

                # Count weekly course occurrences per section
                course_hours_per_week[sec][course_name] += 1

                # Count daily course occurrences per section
                course_hours_per_day[sec][day][course_name] += 1
                # If course appears more than once on the same day, penalize each extra occurrence
                if course_hours_per_day[sec][day][course_name] > 1:
                    cost += 1  # Additional penalty for repeating the same course in the same day

                # Room conflicts
                if room_number != "NA":
                    if room_number in room_schedule[(day, hour)]:
                        cost += 1  # Room already occupied
                    else:
                        room_schedule[(day, hour)].add(room_number)
                else:
                    cost += 5  # High penalty for unassigned room

    # Enforce course-specific weekly constraints
    for sec, courses in course_hours_per_week.items():
        for course, hrs in courses.items():
            if "lab" in course.lower():
                # Labs must appear exactly once per week per section
                if hrs != 1:
                    cost += 1
            else:
                # Theory courses should not exceed 3 times per week per section
                if hrs > 3:
                    cost += 1

    return cost


# Generate the initial state as a nested dictionary
def generate_initial_state():
    state = defaultdict(lambda: defaultdict(dict))
    for sec in classes:
        for day in days:
            for hour in hours:
                course = random.choice(course_mapping)
                course_name, room_type, lecturer, units = course
                # Assign a room based on the room type and total room count
                num_rooms = rooms[room_type]
                room = f"{room_type[0].upper()}{random.randint(1, num_rooms)}"
                state[sec][day][hour] = (course_name, room_type, lecturer, units, room)
    return state

# Branch and Bound Node
class Node:
    def __init__(self, state, cost, level):
        self.state = state
        self.cost = cost
        self.level = level  # Level in the tree to ensure limited depth

    def __lt__(self, other):
        return self.cost < other.cost  # Priority based on cost

# Generate neighbors (potential course allocations)
def generate_children(node):
    children = []
    level = node.level + 1

    for sec in classes:
        for day in days:
            for hour in hours:
                # Copy state and change one course allocation
                new_state = copy.deepcopy(node.state)

                # Randomly choose a course
                course = random.choice(course_mapping)
                course_name, room_type, lecturer, units = course

                # Assign a room based on the room type and total room count
                num_rooms = rooms[room_type]
                room = f"{room_type[0].upper()}{random.randint(1, num_rooms)}"

                new_state[sec][day][hour] = (course_name, room_type, lecturer, units, room)
                new_cost = calculate_cost(new_state)

                # Only create a child if it improves or equals the current best state
                if new_cost <= node.cost:
                    children.append(Node(new_state, new_cost, level))

    return children

# Branch and Bound solution
def branch_and_bound():
    initial_state = generate_initial_state()
    initial_cost = calculate_cost(initial_state)
    best_cost = initial_cost
    best_state = initial_state

    # Priority queue for nodes (ordered by cost)
    pq = PriorityQueue()
    pq.put(Node(initial_state, initial_cost, 0))

    max_iterations = 1000  # Limit iterations to prevent infinite loops
    iterations = 0

    while not pq.empty() and iterations < max_iterations:
        node = pq.get()

        # Stop if cost is zero (optimal)
        if node.cost == 0:
            return node.state

        # Generate child nodes
        children = generate_children(node)
        for child in children:
            if child.cost < best_cost:
                best_cost = child.cost
                best_state = child.state
            pq.put(child)

        iterations += 1

    return best_state

# Convert timetable data into a Pandas DataFrame
def create_timetable_df(state, class_name):
    data = []
    for day in days:
        row = []
        for hour in hours:
            if hour in state[class_name][day]:
                course_name, room_type, lecturer, units, room = state[class_name][day][hour]
                row.append(f"{course_name}, {room_type}:{room}")
            else:
                row.append("NaN")
        data.append(row)
    return pd.DataFrame(data, index=days, columns=hours)

# Generate HTML content for the timetable
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

    # Add timetable for each class
    for class_name in classes:
        html_content += f'<h2>Timetable for {class_name}</h2>'
        timetable = create_timetable_df(state, class_name)
        html_content += timetable.to_html(classes='table', border=0)
        html_content += "<br><br>"

    html_content += '</body></html>'

    # Save the HTML content to a file
    file_name = "timetables_styled_branch_bound.html"
    with open(file_name, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    print(f"Styled HTML page with timetables generated: {file_name}")

    # Get the absolute path to the HTML file using pathlib
    file_path = Path(file_name).resolve()
    file_url = file_path.as_uri()

    print(f"Opening HTML file at: {file_url}")

    # Open the HTML file in the default web browser
    opened = webbrowser.open(file_url)

    if not opened:
        print("Failed to open the web browser automatically. Please open the HTML file manually.")

# Run the branch and bound algorithm and generate the HTML output
final_state = branch_and_bound()
generate_html(final_state)
