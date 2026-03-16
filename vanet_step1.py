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
        vehicle_data[time][vid] = (x, y, speed)

print("Mobility data extracted.")
print(f"Total timesteps: {len(vehicle_data)}")


# STEP 2 — RSU Configuration

RSUs = [
    {"id": "RSU1", "x": 500, "y": 500, "radius": 300},
    {"id": "RSU2", "x": 1200, "y": 800, "radius": 300}
]

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# STEP 3 — Nearest RSU Selection + Entry/Exit

print("\nServing RSU Selection + Entry/Exit Tracking...\n")

serving_rsu_log = {}        # (time, vehicle) -> rsu_id
previous_serving = {}       # vehicle -> previous rsu
entry_time_log = {}         # (vehicle, rsu) -> entry time

for time in sorted(vehicle_data.keys()):
    for vid, (vx, vy, speed) in vehicle_data[time].items():

        # Find nearest RSU
        nearest_rsu = None
        min_distance = float("inf")

        for rsu in RSUs:
            d = distance(vx, vy, rsu["x"], rsu["y"])
            if d < min_distance:
                min_distance = d
                nearest_rsu = rsu

        # Check coverage
        if min_distance <= nearest_rsu["radius"]:
            current_rsu = nearest_rsu["id"]
        else:
            current_rsu = None

        serving_rsu_log[(time, vid)] = current_rsu

        # ENTRY detection
        if vid not in previous_serving:
            previous_serving[vid] = None

        if current_rsu != previous_serving[vid]:

            # EXIT previous
            if previous_serving[vid] is not None:
                old_rsu = previous_serving[vid]
                entry_time = entry_time_log[(vid, old_rsu)]
                dwell_time = time - entry_time
                print(f"Vehicle {vid} EXITED {old_rsu} at time {time}")
                print(f"   Dwell Time: {dwell_time:.2f} sec\n")

            # ENTER new
            if current_rsu is not None:
                entry_time_log[(vid, current_rsu)] = time
                print(f"Vehicle {vid} ENTERED {current_rsu} at time {time}")
  
        previous_serving[vid] = current_rsu


# STEP 4 — Remaining Dwell Time Prediction


print("\nRemaining Dwell Time Prediction (first 10 timesteps)...\n")

for time in sorted(vehicle_data.keys())[:10]:
    for vid, (vx, vy, speed) in vehicle_data[time].items():

        current_rsu = serving_rsu_log[(time, vid)]

        if current_rsu is not None and speed > 0:
            rsu = next(r for r in RSUs if r["id"] == current_rsu)
            d = distance(vx, vy, rsu["x"], rsu["y"])

            remaining_distance = rsu["radius"] - d
            remaining_time = remaining_distance / speed

            if remaining_time > 0:
                print(f"Time {time}: Vehicle {vid}")
                print(f"   Connected to: {current_rsu}")
                print(f"   Predicted Remaining Time: {remaining_time:.2f} sec\n")


# STEP 5 — V2V Neighbor Detection (Sample)

COMM_RANGE = 250

print("\nV2V Neighbors (first 5 timesteps)...\n")

for time in sorted(vehicle_data.keys())[:5]:
    vehicles = vehicle_data[time]

    for v1 in vehicles:
        neighbors = []
        x1, y1, _ = vehicles[v1]

        for v2 in vehicles:
            if v1 != v2:
                x2, y2, _ = vehicles[v2]
                if distance(x1, y1, x2, y2) <= COMM_RANGE:
                    neighbors.append(v2)

        print(f"Time {time}: Vehicle {v1} has {len(neighbors)} neighbors")


# STEP 6 — Communication Model (Serving RSU Only)


TX_POWER = 0.1
NOISE = 1e-9
PATH_LOSS_EXP = 3

def channel_gain(d):
    if d == 0:
        return 1
    return 1 / (d ** PATH_LOSS_EXP)

print("\nCommunication Model (first 5 timesteps)...\n")

for time in sorted(vehicle_data.keys())[:5]:
    for vid, (vx, vy, _) in vehicle_data[time].items():

        current_rsu = serving_rsu_log[(time, vid)]

        if current_rsu is not None:
            rsu = next(r for r in RSUs if r["id"] == current_rsu)
            d = distance(vx, vy, rsu["x"], rsu["y"])

            h = channel_gain(d)
            sinr = (TX_POWER * h) / NOISE
            rate = np.log2(1 + sinr)

            print(f"Time {time}: {vid} -> {current_rsu}")
            print(f"   Distance: {d:.2f}")
            print(f"   SINR: {sinr:.2e}")
            print(f"   Rate: {rate:.2f} bps/Hz\n")
