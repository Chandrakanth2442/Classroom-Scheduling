import random
from collections import defaultdict
import os
import webbrowser
import pandas as pd

# Problem definitions
classes = ["Sec-A", "Sec-B", "Sec-C", "Sec-D"]
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
hours = ["h1", "h2", "h3", "h4", "h5", "h6"]

# Course Mapping: <course_name, room_type, lecturer, units>
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
theory_rooms = [f"T{i}" for i in range(1, rooms["Theory"] + 1)]
lab_rooms = [f"L{i}" for i in range(1, rooms["Lab"] + 1)]

def generate_initial_state():
    state = defaultdict(lambda: defaultdict(dict))
    room_schedule = defaultdict(set)  # Tracks occupied rooms for each (day, hour)
    for sec in classes:
        for day in days:
            for hour in hours:
                course = random.choice(course_mapping)
                course_name, room_type, lecturer, units = course
                # Assign a room based on room_type and availability
                if room_type.lower() == "theory":
                    available_rooms = set(theory_rooms) - room_schedule[(day, hour)]
                else:
                    available_rooms = set(lab_rooms) - room_schedule[(day, hour)]

                if available_rooms:
                    room_number = random.choice(list(available_rooms))
                    room_schedule[(day, hour)].add(room_number)
                else:
                    room_number = "NA"  # No available room

                state[sec][day][hour] = [course_name, room_type, lecturer, units, room_number]
    return state

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


def generate_html(state, filename="schedule.html"):
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

    for sec in classes:
        html_content += f"<h2>Timetable for {sec}</h2>"
        html_content += "<table><tr><th>Hour</th>" + "".join(f"<th>{day.capitalize()}</th>" for day in days) + "</tr>"
        
        for hour in hours:
            html_content += f"<tr><td>{hour.upper()}</td>"
            for day in days:
                details = state[sec][day][hour]
                course_name, room_type, lecturer, units, room_number = details
                room_display = room_number if room_number != "NA" else "No Room"
                html_content += f"<td>{course_name} ({room_type}, {lecturer}, {room_display})</td>"
            html_content += "</tr>"
        html_content += "</table><br/>"
    html_content += "</body></html>"

    with open(filename, "w") as file:
        file.write(html_content)

    print(f"Timetable saved as {filename}")
    file_path = os.path.abspath(filename)
    webbrowser.open(f"file://{file_path}")


# Ant Colony Optimization Code

NUM_ANTS = 10
NUM_ITERATIONS = 50
EVAPORATION_RATE = 0.1
INITIAL_PHEROMONE = 1.0

# For ACO, we treat each (class, day, hour) as a slot to fill with a course.
slots = [(sec, d, h) for sec in classes for d in days for h in hours]

# Initialize pheromones for each course with a dictionary keyed by course_name
pheromones = {course[0]: INITIAL_PHEROMONE for course in course_mapping}

def choose_course_aco(available_courses):
    # Probability based selection
    weights = [pheromones[c[0]] for c in available_courses]
    total = sum(weights)
    if total == 0:
        return random.choice(available_courses)
    probs = [w/total for w in weights]
    return random.choices(available_courses, weights=probs, k=1)[0]

def create_ant_solution():
    state = defaultdict(lambda: defaultdict(dict))
    room_schedule = defaultdict(set)
    for sec, d, h in slots:
        course = choose_course_aco(course_mapping)
        course_name, room_type, lecturer, units = course
        if room_type.lower() == "theory":
            available_rooms = set(theory_rooms) - room_schedule[(d, h)]
        else:
            available_rooms = set(lab_rooms) - room_schedule[(d, h)]

        if available_rooms:
            room_number = random.choice(list(available_rooms))
            room_schedule[(d, h)].add(room_number)
        else:
            room_number = "NA"

        state[sec][d][h] = [course_name, room_type, lecturer, units, room_number]

    return state

def update_pheromones(ants):
    # Evaporate
    for c in pheromones:
        pheromones[c] *= (1 - EVAPORATION_RATE)
        if pheromones[c] < 0.1:
            pheromones[c] = 0.1
    # Reinforce based on best ant
    best_ant = min(ants, key=lambda x: x['cost'])
    for sec, d, h in slots:
        c_name = best_ant['state'][sec][d][h][0]
        pheromones[c_name] += 1 / (1 + best_ant['cost'])

def ant_colony_optimization():
    best_state = None
    best_cost = float('inf')

    for iteration in range(NUM_ITERATIONS):
        ants = []
        for _ in range(NUM_ANTS):
            s = create_ant_solution()
            c = calculate_cost(s)
            ants.append({'state': s, 'cost': c})
            if c < best_cost:
                best_cost = c
                best_state = s
        update_pheromones(ants)

        if (iteration + 1) % 10 == 0:
            print(f"ACO Iteration {iteration+1}: Best Cost = {best_cost}")
        if best_cost == 0:
            break

    print("ACO Final Best Cost:", best_cost)
    return best_state

# Run ACO
final_state_aco = ant_colony_optimization()
generate_html(final_state_aco, "schedule_aco.html")
