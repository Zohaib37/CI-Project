import traci

# Start SUMO simulation
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"]

# Start the simulation
traci.start(sumoCmd)

# Get the traffic light ID
tls_id = traci.trafficlight.getIDList()[0]
print(f"Controlling traffic light: {tls_id}")

phases = [
    traci.trafficlight.Phase(duration=42, state="GGrr"),  # N-S Green, E-W Red. Lasts for 42ms
    traci.trafficlight.Phase(duration=3,  state="yyrr"),  # N-S yellow, E-W remains Red. Lasts for 3ms
    traci.trafficlight.Phase(duration=42, state="rrGG"),  # N-S becomes red. E-W becomes green. Lasts for 42ms
    traci.trafficlight.Phase(duration=3,  state="rryy")   # N-s remains red. E-W becomes yellow. Lasts for 3ms
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

print(f"Average Waiting Time: {avg_waiting:.2f} seconds")
