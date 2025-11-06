import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import Popup, FeatureGroup
from folium.plugins import FeatureGroupSubGroup
import os
import random
import io  # For in-memory byte stream
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
import streamlit.components.v1 as components

# --- PSO Optimization Script ---
# This assumes 'multi_depot_pso.py' is in the same directory
# and has the 'pso_main_from_df' function.
try:
    from multi_depot_pso import pso_main_from_df
except ImportError:
    st.error("Fatal Error: Could not import `pso_main_from_df` from `multi_depot_pso.py`.")
    st.info("Please ensure `multi_depot_pso.py` exists and contains a function `pso_main_from_df(input_df, ...)`.")


    # Create a dummy function so the app can load without crashing
    def pso_main_from_df(*args, **kwargs):
        st.warning("Using dummy PSO function. Optimization will not run.")
        results_df = pd.DataFrame(
            columns=['Assigned_Depot', 'Final_Cluster_ID', 'Latitude', 'Longitude', 'City', 'Demand', 'Dealer_Name'])
        summary_df = pd.DataFrame()
        depots_df = pd.DataFrame()
        return results_df, summary_df, depots_df

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Multi-Depot Route Optimizer",
    layout="wide"
)


# -----------------------------
# Helper function
# -----------------------------
def random_color():
    """Generates a random hex color."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# -----------------------------
# Core Map Plotting Function
# -----------------------------
def plot_all_cities_streamlit(dealers_df):
    """
    Plots all trucks city-by-city on one map with nested filters.
    Adapted for Streamlit with st.progress and st.warning.
    """

    if dealers_df.empty:
        st.warning("⚠️ Dealer data is empty. Cannot create map.")
        return folium.Map()

    # Handle potential empty data after filtering
    if dealers_df['Latitude'].empty or dealers_df['Longitude'].empty:
        st.error("Cannot generate map: No valid latitude or longitude data found.")
        return folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Default to India

    avg_lat = dealers_df['Latitude'].mean()
    avg_lon = dealers_df['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7, tiles="CartoDB Positron")

    depot_added = set()

    # Top-level group for Depots
    depot_fg = FeatureGroup(name="📍 All Depots", show=True).add_to(m)

    all_cities = sorted(dealers_df['City'].unique())

    # --- Streamlit Progress Bar Setup ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_cities = len(all_cities)

    for i, city in enumerate(all_cities):
        # Update progress
        progress_percentage = (i + 1) / total_cities
        progress_bar.progress(progress_percentage)
        status_text.info(f"🏙️ Processing City {i + 1}/{total_cities}: {city}")

        city_dealers = dealers_df[dealers_df['City'] == city].copy()
        if city_dealers.empty:
            continue

        # Top-level FeatureGroup for each city
        city_fg = FeatureGroup(name=f"🏙️ {city}", show=True).add_to(m)

        trucks = sorted(city_dealers['Cluster_ID'].unique())

        for truck_id in trucks:
            truck_group = city_dealers[city_dealers['Cluster_ID'] == truck_id].copy()
            if truck_group.empty:
                continue

            depot_id = truck_group['Depot_ID'].iloc[0]

            # Using dealer centroid as depot location
            depot_lat = truck_group['Latitude'].mean()
            depot_lon = truck_group['Longitude'].mean()
            color = random_color()

            # Add Depot marker
            if depot_id not in depot_added:
                folium.Marker(
                    location=[depot_lat, depot_lon],
                    popup=Popup(f"<b>Depot {depot_id}</b><br>(Est. Location)", max_width=300),
                    icon=folium.Icon(color="red", icon="industry", prefix="fa")
                ).add_to(depot_fg)
                depot_added.add(depot_id)

            # Graph and Route Calculation
            all_points = truck_group[['Latitude', 'Longitude']].values.tolist() + [[depot_lat, depot_lon]]
            lats, lons = zip(*all_points)
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            max_dist = max([geodesic((center_lat, center_lon), (lat, lon)).meters for lat, lon in all_points])
            radius = min(max(max_dist * 1.5, 1000), 50000)  # Min 1km, Max 50km radius

            try:
                G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type='drive')

                if not G.nodes:
                    continue
                if not nx.is_strongly_connected(G):
                    largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
                    G = G.subgraph(largest_scc_nodes).copy()

            except Exception as e:
                continue

            # Map coordinates to nearest nodes
            try:
                truck_group['nearest_node'] = truck_group.apply(
                    lambda row: ox.distance.nearest_nodes(G, X=row['Longitude'], Y=row['Latitude']), axis=1
                )
                depot_node = ox.distance.nearest_nodes(G, X=depot_lon, Y=depot_lat)
            except Exception as e:
                continue

            # Build route
            route_nodes = [depot_node] + truck_group['nearest_node'].tolist() + [depot_node]
            route_coords = []
            total_distance_meters = 0

            for j in range(len(route_nodes) - 1):
                try:
                    if not G.has_node(route_nodes[j]) or not G.has_node(route_nodes[j + 1]):
                        print(f"  - Skipping segment (node missing) for Truck {truck_id}")  # Console log for debug
                        continue

                    segment = nx.shortest_path(G, route_nodes[j], route_nodes[j + 1], weight='length')
                    segment_length = nx.path_weight(G, segment, weight='length')
                    total_distance_meters += segment_length
                    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in segment]
                    route_coords.extend(coords)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    print(f"  - Skipping segment (no path) for Truck {truck_id}")  # Console log for debug
                    continue

            # Calculate metrics for popup
            total_kms = total_distance_meters / 1000
            total_units = truck_group['Demand'].sum()
            dealer_names_list = truck_group['Dealer_Name'].unique().tolist()
            dealer_names = ", ".join([str(d) for d in dealer_names_list])

            # Cost calculation
            mileage_kmpl = 9
            fuel_cost_rs_per_l = 104
            total_cost = (total_kms / mileage_kmpl) * fuel_cost_rs_per_l

            # Create Popup HTML
            route_popup_html = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4 style="margin:0 0 5px 0; padding-bottom:5px; border-bottom:1px solid #ccc;">🚚 Truck {truck_id}</h4>
                <ul style="list-style-type:none; padding:0; margin:0;">
                    <li><b>Total Distance:</b> {total_kms:.2f} km</li>
                    <li><b>Total Demand:</b> {total_units} units</li>
                    <li><b>Estimated Cost:</b> ₹{total_cost:.2f}</li>
                    <li style="margin-top:5px;"><b>Dealers Served:</b><br>{dealer_names}</li>
                </ul>
            </div>
            """

            # Draw route
            if route_coords:
                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=Popup(route_popup_html, max_width=350),
                    tooltip=f"Truck {truck_id} from Depot {depot_id} ({city})"
                ).add_to(city_fg)

            # Dealer markers
            for _, dealer in truck_group.iterrows():
                popup_html = f"""
                <b>Dealer:</b> {dealer['Dealer_Name']}<br>
                <b>City:</b> {dealer['City']}<br>
                <b>Depot:</b> {depot_id}<br>
                <b>Truck/Cluster:</b> {truck_id}<br>
                <b>Demand:</b> {dealer.get('Demand', 'N/A')} units
                """
                folium.CircleMarker(
                    location=[dealer['Latitude'], dealer['Longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=Popup(popup_html, max_width=300)
                ).add_to(city_fg)

    # Clean up progress bar
    progress_bar.empty()
    status_text.success("✅ All-cities map processed!")

    # Add LayerControl
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🚚 Multi-Depot Truck Route Optimizer")

# --- 1. File Upload ---
st.header("1. Upload Dealer Data")
st.markdown("Upload the Excel file (`.xlsx`) containing all dealer locations, demand, and city information.")

uploaded_file = st.file_uploader("Upload Dealer File", type=["xlsx"], label_visibility="collapsed")

if uploaded_file:

    st.info(f"Processing uploaded file: `{uploaded_file.name}`")
    try:
        input_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()  # Stop execution if file is invalid

    # --- 2. Run Optimization ---
    st.header("2. Generate Map")
    if st.button("🚀 Generate Route Map"):

        results_df = None

        try:
            # --- Run PSO (with hardcoded parameters) ---
            with st.spinner("⚙️ Running PSO Optimization... (This may take several minutes)"):

                results_df, summary_df, depots_df = pso_main_from_df(
                    input_df=input_df,
                    scenario="medium",  # Hardcoded
                    n_depots=7,  # Hardcoded
                    swarm_size=40,  # Hardcoded
                    iterations=150,  # Hardcoded
                    visualize=False,
                    verbose=False,
                )

            st.success("✅ PSO Optimization Complete!")

            # --- Data Processing ---
            dealers_df = results_df.rename(
                columns={'Assigned_Depot': 'Depot_ID', 'Final_Cluster_ID': 'Cluster_ID'}
            )
            dealers_df.columns = dealers_df.columns.str.strip()

            # --- Generate Map ---
            with st.spinner("🗺️ Building Folium map... (Downloading road networks)"):
                folium_map = plot_all_cities_streamlit(dealers_df)

            st.success("✅ Map Generated!")

            # --- 3. Display & Download Map ---
            st.header("3. View and Download Map")

            # Display map from memory
            html_data = folium_map._repr_html_()
            components.html(html_data, height=700, scrolling=True)

            # Provide download from memory
            map_data_io = io.BytesIO()
            folium_map.save(map_data_io, close_file=False)

            st.download_button(
                label="⬇️ Download Route Map (HTML)",
                data=map_data_io.getvalue(),
                file_name="multi_depot_route_map.html",
                mime="text/html",
            )

            # Optionally display summary tables
            with st.expander("Show Optimization Summary"):
                st.dataframe(summary_df)
            with st.expander("Show Depot Details"):
                st.dataframe(depots_df)


        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            if results_df is not None:
                st.subheader("PSO Results (Debug)")
                st.dataframe(results_df.head())

else:

    st.info("Please upload an Excel file to begin.")
