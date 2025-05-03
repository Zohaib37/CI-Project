import traci
import random

POP_SIZE = 10
NUM_GENERATIONS = 20
MUTATION_RATE = 0.2
NUM_CHILDREN = 5

GREEN_RANGE = (10, 60)
YELLOW_RANGE = (3, 5)

def generate_individual():
    g1 = random.randint(*GREEN_RANGE)
    y1 = random.randint(*YELLOW_RANGE)
    g2 = random.randint(*GREEN_RANGE)
    y2 = random.randint(*YELLOW_RANGE)
    return [g1, y1, g2, y2]

def evaluate(individual):
    g1, y1, g2, y2 = individual

    # Start SUMO simulation
    sumoBinary = "sumo"
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"]

    # Start the simulation
    traci.start(sumoCmd)

    # Get the traffic light ID
    tls_id = traci.trafficlight.getIDList()[0]
    print(f"Controlling traffic light: {tls_id}")

    phases = [
        traci.trafficlight.Phase(duration=g1, state="GGrr"),  # N-S Green, E-W Red. Lasts for 42ms
        traci.trafficlight.Phase(duration=y1,  state="yyrr"),  # N-S yellow, E-W remains Red. Lasts for 3ms
        traci.trafficlight.Phase(duration=g2, state="rrGG"),  # N-S becomes red. E-W becomes green. Lasts for 42ms
        traci.trafficlight.Phase(duration=y2,  state="rryy")   # N-s remains red. E-W becomes yellow. Lasts for 3ms
    ]

    new_program = traci.trafficlight.Logic("new_program", 0, 0, phases)
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_program)

    # Track metrics
    total_waiting_time = 0
    vehicle_steps = 0

    # Simulate
    for step in range(0, 100):
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh_id)

        vehicle_steps += len(vehicle_ids)

    traci.close()

    # Calculate average waiting time
    if vehicle_steps > 0:
        avg_waiting = total_waiting_time / vehicle_steps
    else:
        avg_waiting = 0

    # print(f"Average Waiting Time: {avg_waiting:.2f} seconds")

    return avg_waiting

def mutate(ind):
    i = random.randint(0, 3)
    if i % 2 == 0:
        ind[i] = random.randint(*GREEN_RANGE)
    else:
        ind[i] = random.randint(*YELLOW_RANGE)
    return ind

def crossover(parent1, parent2):
    point = random.randint(1, 3)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def tournament_parent_selection(score):
    parents = random.sample(score, k = 2)
    if parents[0][1] < parents[1][1]:
        return parents[0][0]
    return parents[1][0]

population = [generate_individual() for _ in range(POP_SIZE)]

for gen in range(NUM_GENERATIONS):
    print(f"Generation {gen + 1}")
    
    # Evaluate population
    scored = [(ind, evaluate(ind)) for ind in population]

    # Crossover
    children = []
    while len(children) < NUM_CHILDREN:
        parent1 = tournament_parent_selection(scored)
        parent2 = tournament_parent_selection(scored)
        c1, c2 = crossover(parent1, parent2)
        children.extend([c1, c2])

    # Mutation
    for i in range(len(children)):
        if random.random() < MUTATION_RATE:
            children[i] = mutate(children[i])

    # Add children to the combined scored array
    for child in children:
        scored.append((child, evaluate(child)))

    scored.sort(key=lambda x: x[1])  # sort by fitness (lower is better)
    print(f"  Best: {scored[0][0]}, Score: {scored[0][1]:.2f}")

    # New Generation
    for i in range(POP_SIZE):
        population[i] = scored[i][0]