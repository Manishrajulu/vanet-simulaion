# VANET Simulation with Adaptive AHP Cluster Head Selection

A Python-based **Vehicular Ad-hoc Network (VANET)** simulation that implements adaptive cluster head selection using the **Analytic Hierarchy Process (AHP)**, combined with accident detection, task scheduling, and RSU-based communication — all driven by real mobility traces generated with **SUMO**.

---

## Overview

This project simulates intelligent vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) communication in a VANET. Vehicles are grouped into clusters dynamically, and a cluster head (CH) is elected per timestep using AHP with adaptive weights based on current traffic conditions. The system also detects accident events using a hybrid detection model and routes priority tasks through the cluster head to the nearest RSU.

---

## Features

- **Adaptive AHP Cluster Head Election** — weights for energy, data rate, speed, and RSU proximity adapt to traffic scenario (high traffic, high mobility, near RSU, or default)
- **Hybrid Accident Detection** — detects accidents via sudden deceleration (Path 1) or persistent stop (Path 2)
- **Priority Task Scheduling** — accident tasks (priority 1) are processed before traffic tasks (priority 2)
- **RSU Integration** — two Road Side Units (RSUs) assist in communication routing
- **Energy Modeling** — cluster heads consume more energy per timestep than regular members
- **Performance Metrics** — reports average delay, task success rate, cluster lifetime, and CH stability

---

## Project Structure

```
vanet-simulaion/
├── vanet_adaptive_ahp.py     # Main simulation script (AHP-based CH selection + accident detection)
├── vanet_step1.py            # Step 1: Mobility data extraction from SUMO trace
├── mobility_trace.xml        # SUMO-generated vehicle mobility trace (FCD output)
├── map.osm                   # OpenStreetMap source map
├── map.net.xml               # SUMO network file (converted from OSM)
├── map.sumocfg               # SUMO simulation configuration
├── routes.rou.xml            # Vehicle routes
├── routes.rou.alt.xml        # Alternate routes
├── trips.trips.xml           # Trip definitions
├── final_output.txt          # Sample simulation output
├── test_metrics_output.txt   # Test run metrics
├── test_refined_metrics.txt  # Refined test run metrics
├── test_high_load.txt        # High load scenario test output
└── minor project.txt         # Project notes
```

---

## Requirements

- Python 3.8+
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) (for regenerating mobility traces)
- Python packages:
  ```
  numpy
  ```

Install dependencies:
```bash
pip install numpy
```

---

## How to Run

### 1. (Optional) Regenerate the mobility trace using SUMO

If you want to re-run the SUMO simulation to generate a fresh `mobility_trace.xml`:

```bash
sumo -c map.sumocfg --fcd-output mobility_trace.xml
```

### 2. Run the main simulation

```bash
python vanet_adaptive_ahp.py
```

The script will:
- Parse `mobility_trace.xml` for vehicle positions, speeds, and angles
- Form clusters at each timestep based on communication range (300 m) and direction threshold (60°)
- Elect a cluster head using adaptive AHP scoring
- Detect accidents using the hybrid model
- Schedule and process tasks via the cluster head
- Print a final performance summary

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `COMM_RANGE` | 300 m | Max V2V communication range |
| `DIRECTION_THRESHOLD` | 60° | Max angle difference for clustering |
| `TX_POWER` | 0.1 W | Vehicle transmit power |
| `PATH_LOSS_EXP` | 3 | Path loss exponent |
| `DEADLINE` | 5 s | Task processing deadline |
| `TASK_LOAD` | 1.0 | Load multiplier (0.5=Low, 1.0=Normal, 1.5=High, 2.0=Very High) |
| `INITIAL_ENERGY` | 100.0 | Starting energy per vehicle |
| `CH_ENERGY_LOSS` | 0.15 | Energy deducted per timestep for cluster heads |
| `NORMAL_ENERGY_LOSS` | 0.05 | Energy deducted per timestep for regular members |

---

## AHP Weight Scenarios

Weights are assigned to four criteria: **[Energy, Data Rate, Speed, RSU Proximity]**

| Scenario | Trigger Condition | Weights |
|---|---|---|
| High Traffic | Cluster size > 8 | [0.25, 0.40, 0.20, 0.15] |
| High Mobility | Avg speed > 15 m/s | [0.25, 0.25, 0.35, 0.15] |
| Near RSU | Closest vehicle < 100 m to RSU | [0.20, 0.30, 0.20, 0.30] |
| Default | Otherwise | [0.25, 0.25, 0.25, 0.25] |

---

## Output

At the end of the simulation, a summary is printed:

```
--------------------------------------------------
Simulation Summary

Total Clusters: ...
Average Cluster Size: ... vehicles
Average Cluster Lifetime: ... seconds
Cluster Head Changes: ...
Average Remaining Energy: ...
--------------------------------------------------

=== PERFORMANCE METRICS ===
Total Tasks: ...
Average Delay: ...
Success Rate: ... %
===========================
```

---

## Technologies Used

- **SUMO** — microscopic traffic simulation for mobility trace generation
- **Python / NumPy** — simulation logic and AHP scoring
- **OpenStreetMap** — real-world road network source

---

## Academic Context

This project was developed as part of a minor project at **SRM Institute of Science and Technology, Ramapuram**, focusing on intelligent clustering and communication protocols for VANETs.
