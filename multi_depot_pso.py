"""
Multi-Depot PSO-Based Dealer Clustering for Vehicle Delivery Optimization

This module exposes two main functions:
1. `pso_main(input_path, ...)`:
   - Reads an xlsx file from `input_path`.
   - Runs the optimization.
   - **Saves** two xlsx output files to disk.
   - Returns (dealers_df, final_summary, depots_df)

2. `pso_main_from_df(input_df, ...)`:
   - Accepts an in-memory pandas DataFrame `input_df`.
   - Runs the optimization.
   - **Does NOT save any files** to disk.
   - Returns (dealers_df, final_summary, depots_df)
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, ceil
import matplotlib.pyplot as plt
# seaborn imported only if visualization requested
from sklearn.cluster import KMeans

# ========================================================================
# CONFIGURATION
# ========================================================================
TRUCK_CONFIG = {
    'small': {'rows': 6, 'capacity': 18, 'radius': 60},
    'medium': {'rows': 10, 'capacity': 30, 'radius': 100},
    'large': {'rows': 15, 'capacity': 45, 'radius': 160}
}

# ========================================================================
# HELPERS / CORE FUNCTIONS
# ========================================================================

def load_dealer_data(path):
    """Load dealer data from provided XLSX path (non-interactive)."""
    df = pd.read_excel(path)
    # Normalise column names if needed
    df.columns = df.columns.str.strip()
    # Ensure required columns exist
    required_cols = {'Latitude', 'Longitude', 'Demand', 'City', 'Dealer_Name'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dealer file: {missing}")
    return df

def identify_depot_candidates(dealers_df, n_depots=5):
    coords = dealers_df[['Latitude', 'Longitude']].values
    kmeans = KMeans(n_clusters=n_depots, random_state=42, n_init=10)
    depot_assignments = kmeans.fit_predict(coords)

    depots = []
    for i in range(n_depots):
        cluster_mask = depot_assignments == i
        cluster_dealers = dealers_df[cluster_mask]

        depot_lat = cluster_dealers['Latitude'].mean()
        depot_lon = cluster_dealers['Longitude'].mean()
        city_counts = cluster_dealers['City'].value_counts()
        main_city = city_counts.index[0] if len(city_counts) > 0 else f"Zone_{i+1}"
        total_demand = cluster_dealers['Demand'].sum()

        depots.append({
            'depot_id': i + 1,
            'lat': depot_lat,
            'lon': depot_lon,
            'main_city': main_city,
            'dealers_count': len(cluster_dealers),
            'total_demand': total_demand
        })

    return pd.DataFrame(depots)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def assign_dealers_to_depots(dealers_df, depots_df):
    def find_nearest_depot(dealer_lat, dealer_lon):
        min_dist = float('inf')
        nearest_depot = None
        for _, depot in depots_df.iterrows():
            dist = haversine_distance(dealer_lat, dealer_lon, depot['lat'], depot['lon'])
            if dist < min_dist:
                min_dist = dist
                nearest_depot = depot['depot_id']
        return nearest_depot, min_dist

    dealers_df = dealers_df.copy()
    dealers_df['Assigned_Depot'] = 0
    dealers_df['Distance_to_Depot'] = 0.0

    for idx, dealer in dealers_df.iterrows():
        depot_id, dist = find_nearest_depot(dealer['Latitude'], dealer['Longitude'])
        dealers_df.at[idx, 'Assigned_Depot'] = depot_id
        dealers_df.at[idx, 'Distance_to_Depot'] = dist

    return dealers_df

def calculate_fitness(cluster_assignments, dealers_df, depot,
                     truck_capacity=30, max_cluster_radius=60,
                     max_dealers_per_cluster=12, min_dealers_per_cluster=3):
    clusters = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        clusters.setdefault(cluster_id, []).append(idx)

    total_cost = 0
    for cluster_id, dealer_indices in clusters.items():
        if len(dealer_indices) == 0:
            continue

        cluster_dealers = dealers_df.iloc[dealer_indices]
        centroid_lat = cluster_dealers['Latitude'].mean()
        centroid_lon = cluster_dealers['Longitude'].mean()

        distance_to_cluster = haversine_distance(depot['lat'], depot['lon'], centroid_lat, centroid_lon)
        distance_cost = 2 * distance_to_cluster * 0.8
        total_cost += distance_cost

        total_demand = cluster_dealers['Demand'].sum()
        trips_needed = ceil(total_demand / truck_capacity)
        if trips_needed > 1:
            capacity_penalty = ((trips_needed - 1) ** 2) * 400
            total_cost += capacity_penalty

        max_dist_from_centroid = 0
        for idx in dealer_indices:
            dealer = dealers_df.iloc[idx]
            dist_from_centroid = haversine_distance(centroid_lat, centroid_lon, dealer['Latitude'], dealer['Longitude'])
            max_dist_from_centroid = max(max_dist_from_centroid, dist_from_centroid)

        if max_dist_from_centroid > max_cluster_radius:
            radius_penalty = ((max_dist_from_centroid - max_cluster_radius) ** 2) * 5
            total_cost += radius_penalty

        if len(dealer_indices) > max_dealers_per_cluster:
            size_penalty = ((len(dealer_indices) - max_dealers_per_cluster) ** 2) * 800
            total_cost += size_penalty

        if len(dealer_indices) < min_dealers_per_cluster:
            tiny_penalty = ((min_dealers_per_cluster - len(dealer_indices)) ** 2) * 600
            total_cost += tiny_penalty

        utilization = (total_demand / (trips_needed * truck_capacity)) * 100
        if utilization < 60:
            total_cost += (60 - utilization) * 15
        elif utilization > 95:
            total_cost += (utilization - 95) * 10

    return total_cost

def run_pso(dealers_df, depot,
            swarm_size=40,
            iterations=150,
            truck_capacity=30,
            max_cluster_radius=60,
            max_dealers_per_cluster=12,
            min_dealers_per_cluster=3,
            verbose=False):
    n_dealers = len(dealers_df)
    if n_dealers == 0:
        return np.array([]), 0, []

    total_demand = dealers_df['Demand'].sum()
    min_clusters = max(3, n_dealers // max_dealers_per_cluster)
    max_clusters = min(n_dealers, max(min_clusters * 2, n_dealers // 5))

    if verbose:
        print(f"PSO: Dealers {n_dealers}, Demand {total_demand}, Clusters {min_clusters}-{max_clusters}")

    particles = []
    for i in range(swarm_size):
        position = np.random.randint(0, max_clusters, size=n_dealers)
        velocity = np.random.uniform(-2, 2, size=n_dealers)
        fitness = calculate_fitness(position, dealers_df, depot,
                                   truck_capacity, max_cluster_radius,
                                   max_dealers_per_cluster, min_dealers_per_cluster)
        particles.append({
            'position': position.copy(),
            'velocity': velocity.copy(),
            'fitness': fitness,
            'pbest': position.copy(),
            'pbest_fitness': fitness
        })

    gbest_idx = np.argmin([p['pbest_fitness'] for p in particles])
    gbest = particles[gbest_idx]['pbest'].copy()
    gbest_fitness = particles[gbest_idx]['pbest_fitness']
    fitness_history = [gbest_fitness]

    for iteration in range(iterations):
        w = 0.9 - (0.5 * iteration / iterations)
        c1 = 2.0
        c2 = 2.0

        for particle in particles:
            r1 = np.random.random(n_dealers)
            r2 = np.random.random(n_dealers)
            particle['velocity'] = (
                w * particle['velocity'] +
                c1 * r1 * (particle['pbest'] - particle['position']) +
                c2 * r2 * (gbest - particle['position'])
            )
            particle['position'] = particle['position'] + particle['velocity']
            particle['position'] = np.clip(np.round(particle['position']), 0, max_clusters - 1).astype(int)

            particle['fitness'] = calculate_fitness(
                particle['position'], dealers_df, depot,
                truck_capacity, max_cluster_radius, max_dealers_per_cluster,
                min_dealers_per_cluster
            )

            if particle['fitness'] < particle['pbest_fitness']:
                particle['pbest'] = particle['position'].copy()
                particle['pbest_fitness'] = particle['fitness']

            if particle['fitness'] < gbest_fitness:
                gbest = particle['position'].copy()
                gbest_fitness = particle['fitness']

        fitness_history.append(gbest_fitness)

    return gbest, gbest_fitness, fitness_history

def analyze_depot_clusters(cluster_assignments, dealers_df, depot,
                           depot_id, truck_capacity=30):
    clusters = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        clusters.setdefault(cluster_id, []).append(idx)

    active_clusters = {i: indices for i, (cid, indices) in enumerate(clusters.items()) if len(indices) > 0}
    cluster_details = []
    total_trips = 0

    for cluster_num, dealer_indices in active_clusters.items():
        cluster_dealers = dealers_df.iloc[dealer_indices]
        total_demand = cluster_dealers['Demand'].sum()
        trips_needed = ceil(total_demand / truck_capacity)
        total_trips += trips_needed

        centroid_lat = cluster_dealers['Latitude'].mean()
        centroid_lon = cluster_dealers['Longitude'].mean()
        distance_from_depot = haversine_distance(depot['lat'], depot['lon'], centroid_lat, centroid_lon)

        max_radius = max([
            haversine_distance(centroid_lat, centroid_lon,
                               dealers_df.iloc[idx]['Latitude'],
                               dealers_df.iloc[idx]['Longitude'])
            for idx in dealer_indices
        ]) if len(dealer_indices) > 0 else 0.0

        utilization = (total_demand / (trips_needed * truck_capacity)) * 100 if trips_needed > 0 else 0.0

        cluster_details.append({
            'Depot_ID': depot_id,
            'Cluster': cluster_num + 1,
            'Dealers': len(dealer_indices),
            'Demand': total_demand,
            'Trips': trips_needed,
            'Distance_km': distance_from_depot,
            'Radius_km': max_radius,
            'Utilization_%': utilization
        })

    return pd.DataFrame(cluster_details)

def visualize_multi_depot_results(dealers_df, depots_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax1 = axes[0]
    depot_colors = plt.cm.Set3(np.linspace(0, 1, len(depots_df)))
    for i, depot_id in enumerate(depots_df['depot_id']):
        depot_dealers = dealers_df[dealers_df['Assigned_Depot'] == depot_id]
        depot_info = depots_df[depots_df['depot_id'] == depot_id].iloc[0]
        ax1.scatter(depot_dealers['Longitude'], depot_dealers['Latitude'], c=[depot_colors[i]], s=50, alpha=0.6,
                    label=f"Depot {depot_id}: {depot_info['main_city']}", edgecolors='black', linewidths=0.3)
        ax1.scatter(depot_info['lon'], depot_info['lat'], c=[depot_colors[i]], s=600, marker='X',
                   edgecolors='black', linewidths=2, zorder=100)
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude'); ax1.set_title('Multi-Depot Regional Assignment')
    ax1.legend(loc='best', fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    unique_clusters = dealers_df['Final_Cluster_ID'].unique()
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    for i, cluster_id in enumerate(sorted(unique_clusters)):
        cluster_dealers = dealers_df[dealers_df['Final_Cluster_ID'] == cluster_id]
        ax2.scatter(cluster_dealers['Longitude'], cluster_dealers['Latitude'],
                    c=[cluster_colors[i % len(cluster_colors)]], s=50, alpha=0.6, edgecolors='black', linewidths=0.3)
    for _, depot in depots_df.iterrows():
        ax2.scatter(depot['lon'], depot['lat'], c='red', s=600, marker='X', edgecolors='black', linewidths=2, zorder=100)
    ax2.set_xlabel('Longitude'); ax2.set_ylabel('Latitude'); ax2.set_title(f'Final Clustering ({len(unique_clusters)} clusters)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ========================================================================
# INTERNAL PIPELINE (SHARED LOGIC)
# ========================================================================

def _run_pso_pipeline(dealers_df,
                      scenario='medium',
                      n_depots=7,
                      swarm_size=40,
                      iterations=150,
                      visualize=False,
                      verbose=False):
    """
    Internal function containing the core PSO logic.
    Accepts a DataFrame and returns (dealers_df, final_summary, depots_df)
    """
    # Config from scenario
    if scenario not in TRUCK_CONFIG:
        raise ValueError("scenario must be one of: " + ", ".join(TRUCK_CONFIG.keys()))
    config = TRUCK_CONFIG[scenario]
    TRUCK_CAPACITY = config['capacity']
    # These were hardcoded in the original pso_main, so we keep them here
    MAX_CLUSTER_RADIUS = 60
    MAX_DEALERS_PER_CLUSTER = 12
    MIN_DEALERS_PER_CLUSTER = 3

    # 1) Identify depots
    depots_df = identify_depot_candidates(dealers_df, n_depots=n_depots)

    # 2) Assign dealers to depots
    dealers_df = assign_dealers_to_depots(dealers_df, depots_df)

    # Ensure columns exist for final cluster id
    dealers_df['Final_Cluster_ID'] = -1
    all_cluster_results = []
    cluster_offset = 0

    # 3) Run PSO per depot
    for depot_id in depots_df['depot_id']:
        depot_dealers = dealers_df[dealers_df['Assigned_Depot'] == depot_id].copy()
        if len(depot_dealers) == 0:
            continue
        depot_info = depots_df[depots_df['depot_id'] == depot_id].iloc[0]
        depot = {'lat': depot_info['lat'], 'lon': depot_info['lon']}
        depot_dealers_reset = depot_dealers.reset_index(drop=True)

        best_solution, best_fitness, fitness_history = run_pso(
            dealers_df=depot_dealers_reset,
            depot=depot,
            swarm_size=swarm_size,
            iterations=iterations,
            truck_capacity=TRUCK_CAPACITY,
            max_cluster_radius=MAX_CLUSTER_RADIUS,
            max_dealers_per_cluster=MAX_DEALERS_PER_CLUSTER,
            min_dealers_per_cluster=MIN_DEALERS_PER_CLUSTER,
            verbose=verbose
        )

        adjusted_clusters = best_solution + cluster_offset
        depot_dealers_reset['Local_Cluster_ID'] = best_solution
        depot_dealers_reset['Final_Cluster_ID'] = adjusted_clusters

        cluster_summary = analyze_depot_clusters(
            best_solution,
            depot_dealers_reset,
            depot,
            depot_id,
            TRUCK_CAPACITY
        )
        all_cluster_results.append(cluster_summary)

        # update global final ids
        for orig_idx, reset_idx in zip(depot_dealers.index, depot_dealers_reset.index):
            dealers_df.at[orig_idx, 'Final_Cluster_ID'] = depot_dealers_reset.at[reset_idx, 'Final_Cluster_ID']

        cluster_offset = int(adjusted_clusters.max()) + 1 if len(adjusted_clusters) > 0 else cluster_offset

    final_summary = pd.concat(all_cluster_results, ignore_index=True) if len(all_cluster_results) > 0 else pd.DataFrame()

    if verbose:
        print(f"PSO Complete. Total Depots: {n_depots}, Total Clusters: {len(final_summary)}, Total Dealers: {len(dealers_df)}")

    # Optionally visualize (return figure)
    fig = None
    if visualize and len(depots_df) > 0:
        fig = visualize_multi_depot_results(dealers_df, depots_df)
        if visualize:
            plt.show()

    # Return dataframes and optionally the figure
    return dealers_df, final_summary, depots_df if fig is None else (dealers_df, final_summary, depots_df, fig)


# ========================================================================
# TOP-LEVEL ENTRY 1: From File Path (Original)
# ========================================================================
def pso_main(input_path,
             scenario='medium',
             n_depots=7,
             swarm_size=40,
             iterations=150,
             visualize=False,
             verbose=False):
    """
    Runs the full pipeline from a local XLSX input file, SAVES results to disk,
    and returns: (dealers_df_with_assignments, final_summary_df, depots_df)

    - input_path: path to uploaded dealers xlsx
    - scenario: 'small'|'medium'|'large' selects truck capacity
    - visualize: if True, returns and shows matplotlib figure (not appropriate in Streamlit)
    """
    # Load dealers from file
    dealers_df = load_dealer_data(input_path)

    # Run the core logic
    result = _run_pso_pipeline(
        dealers_df, scenario, n_depots, swarm_size, iterations, visualize, verbose
    )

    # Unpack results to save them
    if visualize:
        dealers_df_out, final_summary_out, depots_df_out, fig = result
    else:
        dealers_df_out, final_summary_out, depots_df_out = result

    # --- Save results to disk (Original behavior) ---
    output_filename = f'multi_depot_clustered_{scenario}_truck.xlsx'
    summary_filename = f'cluster_summary_{scenario}_truck.xlsx'
    dealers_df_out.to_excel(output_filename, index=False)
    final_summary_out.to_excel(summary_filename, index=False)

    if verbose:
        print("Saved:", output_filename, summary_filename)

    # Return dataframes and optionally the figure
    return result

# ========================================================================
# TOP-LEVEL ENTRY 2: From DataFrame (For Streamlit)
# ========================================================================
def pso_main_from_df(input_df,
                     scenario='medium',
                     n_depots=7,
                     swarm_size=40,
                     iterations=150,
                     visualize=False,
                     verbose=False):
    """
    Runs the full pipeline from an in-memory DataFrame and returns:
    (dealers_df_with_assignments, final_summary_df, depots_df)

    This version DOES NOT save any files to disk.

    - input_df: pandas DataFrame with dealer data
    - scenario: 'small'|'medium'|'large' selects truck capacity
    - visualize: if True, returns and shows matplotlib figure (not appropriate in Streamlit)
    """
    # --- Pre-processing check (from load_dealer_data) ---
    dealers_df = input_df.copy()
    dealers_df.columns = dealers_df.columns.str.strip()
    required_cols = {'Latitude', 'Longitude', 'Demand', 'City', 'Dealer_Name'}
    missing = required_cols - set(dealers_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dealer DataFrame: {missing}")

    # --- Run the core logic ---
    result = _run_pso_pipeline(
        dealers_df, scenario, n_depots, swarm_size, iterations, visualize, verbose
    )

    # --- This version DOES NOT save files ---

    # Return dataframes and optionally the figure
    return result


# If run directly, allow testing from command line (keeps original behavior minimal)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Running pso_main from file: {path}")
        pso_main(path, visualize=True, verbose=True)
    else:
        print("This script is intended to be imported.")
        print("To test, run: python multi_depot_pso.py <path_to_your_dealer_file.xlsx>")