import xml.etree.ElementTree as ET
from collections import defaultdict
import math
import numpy as np

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

COMM_RANGE = 250
DIRECTION_THRESHOLD = 60
TX_POWER = 0.1
NOISE = 1e-9
PATH_LOSS_EXP = 3

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

# Evaluation Metrics
active_clusters = {} # cluster_key -> start_time
cluster_lifetimes = []
ch_tracking = {} # cluster_key -> current_ch
total_ch_changes = 0
all_cluster_sizes = []
previous_ccs = set() # To track which cluster keys were present in the previous timestep

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

print("\nStarting Improved Adaptive AHP Cluster Head Selection...\n")

for time_idx, time in enumerate(v_times):
    vehicles = vehicle_data[time]
    
    # Initialize energy for new vehicles
    for vid in vehicles:
        if vid not in vehicle_energy:
            vehicle_energy[vid] = INITIAL_ENERGY
    
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

    # Cleanup finished clusters and calculate lifetime
    for deleted_ck in (previous_ccs - current_cluster_keys):
        lifetime = time - active_clusters[deleted_ck]
        cluster_lifetimes.append(lifetime)
        del active_clusters[deleted_ck]
        if deleted_ck in ch_tracking:
            del ch_tracking[deleted_ck]
            
    previous_ccs = current_cluster_keys

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
