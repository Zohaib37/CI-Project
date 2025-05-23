import traci
import os
import sys
import random
import time
from datetime import datetime

# GA Parameters
POP_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.2
NUM_CHILDREN = 10

# Traffic Light Timing Constraints
GREEN_RANGE = (10, 60)
YELLOW_RANGE = (3, 5)

# Visualization Settings
VISUALIZATION_MODE = True  # Set to False for headless mode (faster)
VISUALIZATION_DELAY = 0.05  # Seconds between steps when visualizing

def generate_individual():
    """Generate a random traffic light timing configuration"""
    g1 = random.randint(*GREEN_RANGE)
    y1 = random.randint(*YELLOW_RANGE)
    g2 = random.randint(*GREEN_RANGE)
    y2 = random.randint(*YELLOW_RANGE)
    return [g1, y1, g2, y2]

def start_sumo(gui=True, config_file=None, net_file="network.net.xml", route_file="light_traffic.rou.xml"):
    """Start a SUMO simulation"""
    if gui and VISUALIZATION_MODE:
        sumoBinary = "sumo-gui"
    else:
        sumoBinary = "sumo"
    
    if config_file:
        sumoCmd = [sumoBinary, "-c", config_file]
    else:
        sumoCmd = [
            sumoBinary, 
            "-n", net_file,
            "-r", route_file,
            "--start",  # Start the simulation immediately in GUI mode
            "--quit-on-end",  # Close GUI when simulation ends
            "--random"  # Add some randomness
        ]
    
    traci.start(sumoCmd)
    return traci.trafficlight.getIDList()[0]  # Return the traffic light ID

def apply_traffic_light_settings(tls_id, individual):
    """Apply traffic light timing settings to the simulation"""
    g1, y1, g2, y2 = individual
    
    phases = [
        traci.trafficlight.Phase(duration=g1, state="GGrr"),  # N-S Green, E-W Red
        traci.trafficlight.Phase(duration=y1, state="yyrr"),  # N-S Yellow, E-W Red
        traci.trafficlight.Phase(duration=g2, state="rrGG"),  # N-S Red, E-W Green
        traci.trafficlight.Phase(duration=y2, state="rryy")   # N-S Red, E-W Yellow
    ]
    
    new_program = traci.trafficlight.Logic("custom_program", 0, 0, phases)
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_program)

def evaluate_individual(individual, traffic_files, sim_steps=100):
    """Evaluate a traffic light configuration across multiple traffic scenarios"""
    total_waiting_time = 0
    total_vehicle_steps = 0
    
    for traffic_file in traffic_files:
        # Start a new simulation for each traffic scenario
        tls_id = start_sumo(gui=True, net_file="network.net.xml", route_file=traffic_file)
        
        # Apply the individual's traffic light settings
        apply_traffic_light_settings(tls_id, individual)
        
        # Reset metrics for this scenario
        scenario_waiting_time = 0
        scenario_vehicle_steps = 0
        
        # Run the simulation for the specified number of steps
        for step in range(sim_steps):
            traci.simulationStep()
            
            if VISUALIZATION_MODE:
                time.sleep(VISUALIZATION_DELAY)  # Add delay for visualization
                
            # Collect metrics
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                scenario_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh_id)
                
            scenario_vehicle_steps += len(vehicle_ids)
            
        # Close this traffic scenario simulation
        traci.close()
        
        # Add to total metrics
        total_waiting_time += scenario_waiting_time
        total_vehicle_steps += scenario_vehicle_steps
    
    # Calculate the fitness (lower is better)
    if total_vehicle_steps > 0:
        avg_waiting_time = total_waiting_time / total_vehicle_steps
    else:
        avg_waiting_time = float('inf')  # Penalty for no vehicles
        
    return avg_waiting_time

def mutate(individual):
    """Randomly mutate one parameter of the traffic light configuration"""
    i = random.randint(0, 3)
    if i % 2 == 0:  # Green light duration
        individual[i] = random.randint(*GREEN_RANGE)
    else:  # Yellow light duration
        individual[i] = random.randint(*YELLOW_RANGE)
    return individual

def crossover(parent1, parent2):
    """Create two children by crossing over two parents"""
    point = random.randint(1, 3)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def tournament_selection(scored_population):
    """Select a parent using tournament selection"""
    candidates = random.sample(scored_population, k=2)
    if candidates[0][1] < candidates[1][1]:  # Lower score is better
        return candidates[0][0]
    return candidates[1][0]

def visualize_best_solution(best_individual, traffic_files, sim_steps=300):
    """Run a longer visualization of the best solution"""
    print("\n=== Visualizing Best Solution ===")
    print(f"Best traffic light configuration: {best_individual}")
    
    for traffic_file in traffic_files:
        print(f"\nRunning visualization with {traffic_file}...")
        tls_id = start_sumo(gui=True, net_file="network.net.xml", route_file=traffic_file)
        apply_traffic_light_settings(tls_id, best_individual)
        
        # Run a longer simulation to properly visualize the solution
        for step in range(sim_steps):
            traci.simulationStep()
            
            if step % 50 == 0:
                print(f"  Simulation step: {step}/{sim_steps}")
                
            time.sleep(VISUALIZATION_DELAY * 2)  # Slower for better visualization
            
        traci.close()

# Setup for the GA
traffic_files = ["light_traffic.rou.xml", "heavy_traffic.rou.xml"]
population = [generate_individual() for _ in range(POP_SIZE)]

# Baseline evaluation for comparison
baseline = [42, 3, 42, 3]  # Default timing
baseline_score = evaluate_individual(baseline, traffic_files)
print(f"Baseline configuration {baseline} score: {baseline_score:.2f}")

# Run the genetic algorithm
best_individual = None
best_score = float('inf')

for gen in range(NUM_GENERATIONS):
    print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
    
    # Evaluate the current population
    scored_population = []
    for i, individual in enumerate(population):
        score = evaluate_individual(individual, traffic_files)
        scored_population.append((individual, score))
        print(f"  Individual {i+1}: {individual}, Score: {score:.2f}")
    
    # Sort by fitness (lower is better)
    scored_population.sort(key=lambda x: x[1])
    
    # Update best found solution
    if scored_population[0][1] < best_score:
        best_score = scored_population[0][1]
        best_individual = scored_population[0][0].copy()
        
    print(f"  Generation best: {scored_population[0][0]}, Score: {scored_population[0][1]:.2f}")
    print(f"  Overall best: {best_individual}, Score: {best_score:.2f}")
    
    # Create the next generation
    new_population = []
    
    # Elitism: Keep the best individual
    new_population.append(scored_population[0][0].copy())
    
    # Generate children through crossover and mutation
    children = []
    while len(children) < NUM_CHILDREN:
        parent1 = tournament_selection(scored_population)
        parent2 = tournament_selection(scored_population)
        child1, child2 = crossover(parent1, parent2)
        children.extend([child1, child2])
    
    # Apply mutation
    for i in range(len(children)):
        if random.random() < MUTATION_RATE:
            children[i] = mutate(children[i])
            
    # Add children to the population
    new_population.extend(children)
    
    # Fill the rest of the population from the best individuals
    while len(new_population) < POP_SIZE:
        new_population.append(scored_population[len(new_population) - len(children)][0].copy())
        
    # Replace the old population
    population = new_population

# Final results
print("\n=== Final Results ===")
print(f"Best traffic light configuration: {best_individual}")
print(f"Best score (avg. waiting time): {best_score:.2f} seconds")
print(f"Improvement over baseline: {(baseline_score - best_score) / baseline_score * 100:.2f}%")

# Visualize the best solution
visualize_best_solution(best_individual, traffic_files)
