import xml.etree.ElementTree as ET
from collections import defaultdict
import math
import numpy as np
import random

# STEP 1 — Extract Mobility Data

tree = ET.parse("mobility_trace.xml")
root = tree.getroot()

vehicle_data = defaultdict(dict)

for timestep in root.findall("timestep"):
    time = float(timestep.attrib["time"])
    for vehicle in timestep.findall("vehicle"):
        vid = vehicle.attrib["id"]
        x = float(vehicle.attrib["x"])
        y = float(vehicle.attrib["y"])
        speed = float(vehicle.attrib["speed"])
        angle = float(vehicle.attrib.get("angle", 0.0))
        

        
        vehicle_data[time][vid] = (x, y, speed, angle)

print("Mobility data extracted.")
v_times = sorted(vehicle_data.keys())
print(f"Total timesteps: {len(v_times)}")


# STEP 2 — RSU Configuration

RSUs = [
    {"id": "RSU1", "x": 500, "y": 500, "radius": 300},
    {"id": "RSU2", "x": 1200, "y": 800, "radius": 300}
]

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# STEP 3 — Communication Parameters

COMM_RANGE = 300
DIRECTION_THRESHOLD = 60
TX_POWER = 0.1
NOISE = 1e-9
PATH_LOSS_EXP = 3
DEADLINE = 5
TASK_LOAD = 1.0 # 0.5=Low, 1.0=Normal, 1.5=High, 2.0=Very High

def channel_gain(d):
    if d == 0:
        return 1
    return 1 / (d ** PATH_LOSS_EXP)

def calculate_rate(vx, vy, rsu):
    d = distance(vx, vy, rsu["x"], rsu["y"])
    if d > rsu["radius"]:
        return 0
    h = channel_gain(d)
    sinr = (TX_POWER * h) / NOISE
    rate = np.log2(1 + sinr)
    return rate


# STEP 4 — Adaptive AHP Implementation

# Energy Model
vehicle_energy = {} # vid -> energy
INITIAL_ENERGY = 100.0
NORMAL_ENERGY_LOSS = 0.05
CH_ENERGY_LOSS = 0.15

# Accident Detection Infrastructure
previous_speed = {}  # vid -> previous speed
stopped_duration = {}  # vid -> count of consecutive stopped timesteps
event_list = []  # list of accident events
reported_accidents = set()  # vid -> reported status (to prevent duplicates)
# Simplified thresholds for hybrid model
ACCIDENT_SPEED_THRESHOLD = 8       # Must be moving fast for Path 1
ACCIDENT_DECEL_THRESHOLD = 6       # Deceleration > 6 for Path 1
ACCIDENT_STOPPED_SPEED = 1         # Speed < 1 for Path 2
ACCIDENT_STOPPED_MIN = 4           # Increased to 4 timesteps for Path 2 (reduces false positives)
ACCIDENT_IMMEDIATE_THRESHOLD = 2    # Speed < 2 for Path 1 immediate detection
ACCIDENT_PREV_SPEED_MIN = 5       # Prev speed > 5 for Path 2 (ensures vehicle was moving)
ACCIDENT_NEIGHBOR_RADIUS = 100     # 100m radius to check neighbors
ACCIDENT_NEIGHBOR_SPEED = 5        # Neighbor avg speed < 5 m/s indicates slowdown

# Evaluation Metrics
active_clusters = {} # cluster_key -> start_time
cluster_lifetimes = []
ch_tracking = {} # cluster_key -> current_ch
total_ch_changes = 0
all_cluster_sizes = []
previous_ccs = set() # To track which cluster keys were present in the previous timestep

# Task Management
generated_tasks = []  # All tasks generated in current timestep
cluster_task_queue = {}  # CH -> list of tasks

def get_clusters(vehicles):
    """Group vehicles into clusters based on COMM_RANGE."""
    vids = list(vehicles.keys())
    adj = defaultdict(list)
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            v1, v2 = vids[i], vids[j]
            x1, y1, speed1, angle1 = vehicles[v1]
            x2, y2, speed2, angle2 = vehicles[v2]
            dist = distance(x1, y1, x2, y2)
            diff = abs(angle1 - angle2)
            dir_diff = min(diff, 360 - diff)
            if dist <= COMM_RANGE and dir_diff <= DIRECTION_THRESHOLD:
                adj[v1].append(v2)
                adj[v2].append(v1)
    
    clusters = []
    visited = set()
    for vid in vids:
        if vid not in visited:
            cluster = []
            queue = [vid]
            visited.add(vid)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)
    return clusters

def get_weights(cluster, vehicles, rsus):
    """Adaptive Weight Selection based on traffic conditions."""
    num_vehicles = len(cluster)
    avg_speed = np.mean([vehicles[vid][2] for vid in cluster])
    
    min_dist_to_rsu = float('inf')
    for vid in cluster:
        vx, vy, vspeed, vangle = vehicles[vid]
        for rsu in rsus:
            d = distance(vx, vy, rsu["x"], rsu["y"])
            if d < min_dist_to_rsu:
                min_dist_to_rsu = d
    
    # Weights Order: [Energy, DataRate, Speed, RSUProximity]
    if num_vehicles > 8: # High Traffic
        return [0.25, 0.40, 0.20, 0.15], "High Traffic"
    elif avg_speed > 15: # High Mobility
        return [0.25, 0.25, 0.35, 0.15], "High Mobility"
    elif min_dist_to_rsu < 100: # Near RSU
        return [0.20, 0.30, 0.20, 0.30], "Near RSU"
    else: # Default
        return [0.25, 0.25, 0.25, 0.25], "Default"

def normalize_safe(values):
    """Normalize values safely between 0 and 1."""
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [0.5] * len(values)
    return [(v - v_min) / (v_max - v_min) for v in values]

def detect_accidents(vehicles, time, time_idx):
    """Detect accidents using hybrid model (immediate + persistent)."""
    detected_accidents = []

    for vid, (x, y, speed, angle) in vehicles.items():
        prev_speed = previous_speed.get(vid, speed)
        speed_drop = prev_speed - speed

        # Track stopped duration (for Path 2) - only if vehicle was moving before
        if prev_speed > ACCIDENT_PREV_SPEED_MIN and speed < ACCIDENT_STOPPED_SPEED:
            stopped_duration[vid] = stopped_duration.get(vid, 0) + 1
        elif speed >= ACCIDENT_STOPPED_SPEED:
            stopped_duration[vid] = 0

        # Update previous speed
        previous_speed[vid] = speed

        # PATH 1: Immediate detection (sudden deceleration) - UNCHANGED
        accident_detected = False
        detection_path = "UNKNOWN"

        # Check PATH 1 independently
        if (prev_speed > ACCIDENT_SPEED_THRESHOLD and
            speed_drop > ACCIDENT_DECEL_THRESHOLD and
            speed < ACCIDENT_IMMEDIATE_THRESHOLD):
            accident_detected = True
            detection_path = "IMMEDIATE"

        # Check PATH 2 independently (not elif - both paths can trigger)
        if stopped_duration.get(vid, 0) >= ACCIDENT_STOPPED_MIN:
            accident_detected = True
            detection_path = "PERSISTENT"

        # Debug logging for potential accidents (first 50 timesteps)
        if time_idx < 50 and speed_drop > 4:
            print(f"  POTENTIAL Vehicle {vid}: prev={prev_speed:.1f}, "
                  f"curr={speed:.1f}, drop={speed_drop:.1f}, "
                  f"stopped={stopped_duration.get(vid, 0)}")

        # If accident detected
        if accident_detected:
            if vid not in reported_accidents:
                # Full debug logging (first 50 timesteps)
                if time < 50:
                    print(f"\n  === ACCIDENT DETECTED ({detection_path}) ===")
                    print(f"  Vehicle {vid}:")
                    print(f"  prev_speed={prev_speed:.1f}, curr_speed={speed:.1f}")
                    print(f"  speed_drop={speed_drop:.1f}, stopped={stopped_duration.get(vid, 0)}")

                print(f"\n=== ACCIDENT DETECTED at Time {time} ===")
                print(f"Vehicle {vid} stopped for {stopped_duration.get(vid, 0)} timesteps ({detection_path})")

                # Create task with priority 1
                accident_task = create_task("accident", vid, (x, y), time, 1)
                generated_tasks.append(accident_task)

                accident_event = {
                    "type": "accident",
                    "vehicle_id": vid,
                    "location": (x, y),
                    "time": time,
                    "speed": speed,
                    "stopped_duration": stopped_duration.get(vid, 0),
                    "detection_path": detection_path
                }
                detected_accidents.append(accident_event)
                event_list.append(accident_event)
                reported_accidents.add(vid)

    return detected_accidents

def create_task(task_type, vehicle_id, location, time, priority):
    """Create a task dictionary."""
    task = {
        "type": task_type,
        "vehicle_id": vehicle_id,
        "location": location,
        "time": time,
        "priority": priority
    }
    return task

print("\nStarting Improved Adaptive AHP Cluster Head Selection...\n")

# Initialize Performance Metrics
total_delay = 0
total_tasks = 0
successful_tasks = 0

for time_idx, time in enumerate(v_times):
    vehicles = vehicle_data[time]
    
    # Initialize energy for new vehicles
    for vid in vehicles:
        if vid not in vehicle_energy:
            vehicle_energy[vid] = INITIAL_ENERGY
    
    # --- STEP 6: Forced test — simulate vehicle "12" stopping at timesteps 40-45 ---
    if "12" in vehicles and 40 <= time_idx <= 45:
        x12, y12, s12, a12 = vehicles["12"]
        vehicles["12"] = (x12, y12, 0.0, a12)  # force speed to 0

    # Accident Detection
    detected_accidents = detect_accidents(vehicles, time, time_idx)

    # Traffic Task Generation (20% probability * TASK_LOAD per vehicle)
    for vid in vehicles:
        if random.random() < (0.2 * TASK_LOAD):
            x, y, speed, angle = vehicles[vid]
            traffic_task = create_task("traffic", vid, (x, y), time, 2)
            generated_tasks.append(traffic_task)
            if time_idx < 50:
                print(f"Traffic task generated for Vehicle {vid}")
    
    # Log detected accidents (limit to first 50 timesteps for readability)
    if time_idx < 50 and detected_accidents:
        print(f"\n=== ACCIDENT DETECTED at Time {time} ===")
        for accident in detected_accidents:
            vid = accident["vehicle_id"]
            x, y = accident["location"]
            print(f"Accident detected by Vehicle {vid}")
            print(f"Location: ({x:.1f}, {y:.1f})")
            print(f"Stopped for {accident['stopped_duration']} timesteps")
    
    # Cluster Formation
    clusters = get_clusters(vehicles)
    current_cluster_keys = set()
    
    if time_idx < 50:
        print(f"Time {time}: {len(clusters)} clusters formed.")
    
    timestep_chs = set()
    
    for i, cluster in enumerate(clusters):
        cluster_key = tuple(sorted(cluster))
        current_cluster_keys.add(cluster_key)
        all_cluster_sizes.append(len(cluster))
        
        # Cluster Lifetime Tracking
        if cluster_key not in active_clusters:
            active_clusters[cluster_key] = time
            
        weights, scenario = get_weights(cluster, vehicles, RSUs)
        
        # CH Candidates Filter (Speed >= 2 m/s)
        candidates = [vid for vid in cluster if vehicles[vid][2] >= 2.0]
        
        # If no moving vehicles, use the whole cluster as fallback
        voting_pool = candidates if candidates else cluster
        
        # Parameter Calculation for voting_pool
        energies = [vehicle_energy[vid] for vid in voting_pool]
        data_rates = []
        for vid in voting_pool:
            vx, vy, vspeed, vangle = vehicles[vid]
            max_rate = max([calculate_rate(vx, vy, rsu) for rsu in RSUs])
            data_rates.append(max_rate)
        speeds = [vehicles[vid][2] for vid in voting_pool]
        rsu_proximities = []
        for vid in voting_pool:
            vx, vy, _, _ = vehicles[vid]
            best_r_score = 0
            for rsu in RSUs:
                d = distance(vx, vy, rsu["x"], rsu["y"])
                # Normalized RSU score based on range
                r_score = 1 - (d / rsu["radius"])
                best_r_score = max(best_r_score, r_score)
            # Clamp between 0 and 1
            rsu_proximities.append(max(0, min(best_r_score, 1)))

        # Normalize
        n_energies = normalize_safe(energies)
        n_rates = normalize_safe(data_rates)
        
        max_s = max(speeds) if speeds else 0
        min_s = min(speeds) if speeds else 0
        if max_s == min_s:
            n_speeds = [0.5] * len(speeds)
        else:
            # Inverted Speed Scoring (Slower vehicles receive higher scores)
            n_speeds = [1 - ((s - min_s) / (max_s - min_s)) for s in speeds]
        n_proximities = normalize_safe(rsu_proximities)
        
        # Score Calculation
        scores = []
        for j in range(len(voting_pool)):
            score = (weights[0] * n_energies[j] + 
                     weights[1] * n_rates[j] + 
                     weights[2] * n_speeds[j] + 
                     weights[3] * n_proximities[j])
            scores.append(score)
            
        # CH Selection
        ch_index = np.argmax(scores)
        ch_vid = voting_pool[ch_index]
        timestep_chs.add(ch_vid)

        # Initialize task queue for this cluster head if not exists
        if ch_vid not in cluster_task_queue:
            cluster_task_queue[ch_vid] = []
        
        # CH Stability Tracking
        if cluster_key in ch_tracking:
            if ch_tracking[cluster_key] != ch_vid:
                total_ch_changes += 1
                ch_tracking[cluster_key] = ch_vid
        else:
            ch_tracking[cluster_key] = ch_vid

        # Detailed Logging (First 50 timesteps)
        if time_idx < 50 and len(cluster) > 1:
            print(f"  Cluster {i+1}: {cluster}")
            print(f"  Scenario Triggered: {scenario}")
            print(f"  Weights: {weights}")
            print(f"  Vehicle Scores:")
            for k, vid in enumerate(voting_pool):
                print(f"    Vehicle {vid} -> {scores[k]:.4f}")
            print(f"  Cluster Head: Vehicle {ch_vid}")
            print(f"  Remaining Energy: {vehicle_energy[ch_vid]:.2f}\n")
        
        # Task Assignment to Cluster Head
        # Assign tasks from this cluster to the cluster head
        for task in generated_tasks:
            task_vid = task["vehicle_id"]
            if task_vid in cluster:
                cluster_task_queue[ch_vid].append(task)
                if time_idx < 50:
                    print(f"  Task assigned: {task['type']} from Vehicle {task_vid} to CH {ch_vid}")

        # Priority Scheduling and Task Processing
        if cluster_task_queue[ch_vid]:
            # Sort tasks by priority (1=accident first, then by time)
            cluster_task_queue[ch_vid].sort(key=lambda t: (t["priority"], t["time"]))

            # --- Refined Delay Calculation ---
            queue_size = len(cluster_task_queue[ch_vid])
            processing_delay = queue_size * 0.5

            if time_idx < 50:
                print(f"  Cluster Head {ch_vid} received {queue_size} tasks")
                print(f"  Processing Delay: {processing_delay:.2f}s")

            # Process tasks in sorted order
            for task in cluster_task_queue[ch_vid]:
                task_vid = task["vehicle_id"]
                task_type = task["type"]

                # --- Metrics Calculation ---
                network_delay = random.uniform(0.5, 1.5)
                delay = processing_delay + network_delay
                
                total_delay += delay
                total_tasks += 1
                if delay <= DEADLINE:
                    successful_tasks += 1

                if time_idx < 50:
                    print(f"  Processing {task_type} (Delay: {delay:.2f}s)")
                    if task_type == "accident":
                        print("  >>> HIGH PRIORITY ALERT <<<")

                # Determine communication path
                task_x, task_y = task["location"]
                nearest_rsu = min(RSUs, key=lambda rsu: distance(task_x, task_y, rsu["x"], rsu["y"]))

                if len(cluster) == 1 or task_vid == ch_vid:
                    # Isolated vehicle or CH is task source - direct to RSU
                    if time_idx < 50:
                        print(f"  Vehicle {task_vid} -> {nearest_rsu['id']}")
                else:
                    # Via Cluster Head
                    if time_idx < 50:
                        print(f"  Vehicle {task_vid} -> CH {ch_vid} -> {nearest_rsu['id']}")

            # Clear task queue after processing
            cluster_task_queue[ch_vid].clear()

    # Cleanup finished clusters and calculate lifetime
    for deleted_ck in (previous_ccs - current_cluster_keys):
        lifetime = time - active_clusters[deleted_ck]
        cluster_lifetimes.append(lifetime)
        del active_clusters[deleted_ck]
        if deleted_ck in ch_tracking:
            del ch_tracking[deleted_ck]
            
    previous_ccs = current_cluster_keys

    # Clear generated tasks for next timestep
    generated_tasks = []

    # Energy Loss Update (apply to all vehicles present)
    for vid in vehicles:
        if vid in timestep_chs:
            vehicle_energy[vid] -= CH_ENERGY_LOSS
        else:
            vehicle_energy[vid] -= NORMAL_ENERGY_LOSS

# Final Lifetime collection for remaining clusters
for ck in active_clusters:
    lifetime = v_times[-1] - active_clusters[ck]
    cluster_lifetimes.append(lifetime)

# Final Summary
print("--------------------------------------------------")
print("Simulation Summary\n")
total_clusters = len(cluster_lifetimes)
avg_size = np.mean(all_cluster_sizes) if all_cluster_sizes else 0
avg_lifetime = np.mean(cluster_lifetimes) if cluster_lifetimes else 0
avg_energy = np.mean(list(vehicle_energy.values())) if vehicle_energy else 0

print(f"Total Clusters: {total_clusters}")
print(f"Average Cluster Size: {avg_size:.1f} vehicles")
print(f"Average Cluster Lifetime: {avg_lifetime:.1f} seconds")
print(f"Cluster Head Changes: {total_ch_changes}")
print(f"Average Remaining Energy: {avg_energy:.1f}")
print("--------------------------------------------------")

# Final Performance Metrics Computation
if total_tasks > 0:
    avg_delay = total_delay / total_tasks
    success_rate = (successful_tasks / total_tasks) * 100
else:
    avg_delay = 0
    success_rate = 0

print("\n=== PERFORMANCE METRICS ===")
print("Total Tasks:", total_tasks)
print("Average Delay:", round(avg_delay, 2))
print("Success Rate:", round(success_rate, 2), "%")
print("===========================\n")
