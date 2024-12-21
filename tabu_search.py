import random
from collections import defaultdict
import os
import webbrowser
import pandas as pd
import copy
import math

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


# Tabu Search Code

def generate_neighborhood(state, neighborhood_size=50):
    neighborhood = []
    for _ in range(neighborhood_size):
        neighbor = copy.deepcopy(state)
        sec = random.choice(classes)
        day = random.choice(days)
        hour = random.choice(hours)
        old_course = neighbor[sec][day][hour]
        new_course = random.choice(course_mapping)
        while new_course[:4] == old_course[:4]:
            new_course = random.choice(course_mapping)
        room_type = new_course[1]
        if room_type == "Theory":
            room_number = random.choice(theory_rooms)
        else:
            room_number = random.choice(lab_rooms)
        neighbor[sec][day][hour] = new_course + [room_number]
        move = (sec, day, hour, old_course[0], new_course[0])
        neighborhood.append((neighbor, move))
    return neighborhood

def tabu_search(max_iter=100, tabu_list_size=50, neighborhood_size=50):
    current_state = generate_initial_state()
    current_cost = calculate_cost(current_state)
    best_state = copy.deepcopy(current_state)
    best_cost = current_cost
    tabu_list = []

    for iteration in range(max_iter):
        neighborhood = generate_neighborhood(current_state, neighborhood_size)
        candidate_moves = []
        for neighbor, move in neighborhood:
            neighbor_cost = calculate_cost(neighbor)
            candidate_moves.append((neighbor, neighbor_cost, move))

        candidate_moves.sort(key=lambda x: x[1])
        chosen_move = None
        for neighbor, neighbor_cost, move in candidate_moves:
            if move not in tabu_list or neighbor_cost < best_cost:
                current_state = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_state = copy.deepcopy(neighbor)
                    best_cost = current_cost
                tabu_list.append(move)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)
                chosen_move = move
                break

        if chosen_move is None:
            # No valid move found, break early
            break

        if (iteration + 1) % 10 == 0:
            print(f"Tabu Iteration {iteration + 1}: Best Cost = {best_cost}")

    print("Tabu Final Best Cost:", best_cost)
    return best_state

# Run Tabu Search
final_state_tabu = tabu_search()
generate_html(final_state_tabu, "schedule_tabu.html")
