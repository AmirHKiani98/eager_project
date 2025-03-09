import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import geopandas as gpd
from shapely.wkt import loads
from pyproj import Proj, Transformer
from tqdm import tqdm




def lat_lon_to_axis(lat, lon, axis_direction=(1, 0), crs="EPSG:3857"):
    """
    Convert latitude and longitude to a one-dimensional value in meters along an axis.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        axis_direction (tuple): Direction vector of the axis (x, y).
        crs (str): CRS for projecting to meters (default: EPSG:3857).

    Returns:
        float: One-dimensional projection value in meters.
    """
    
    # Define the transformer for projecting lat/lon to the desired CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    # Project the latitude and longitude to meters
    x, y = transformer.transform(lon, lat)
    
    # Normalize the axis direction vector
    axis_vector = np.array(axis_direction)
    axis_unit_vector = axis_vector / np.linalg.norm(axis_vector)
    
    # Compute the projection of the point onto the axis
    projection_value = np.dot([x, y], axis_unit_vector)
    return projection_value

def find_traffic_light_id(row, segments_gdf):
    """
    Find the nearest segment ID to a given row in a DataFrame.

    Args:
        row (pd.Series): Row in a DataFrame with columns 'lat' and 'lon'.
        segments_gdf (gpd.GeoDataFrame): GeoDataFrame with segment geometries.

    Returns:
        int: ID of the nearest segment.
    """
    # Create a GeoDataFrame for the point
    point = gpd.GeoDataFrame(
        geometry=[loads(f"POINT ({row['lat']} {row['lon']})")], crs="EPSG:4326" # Make sure lat and lon is given completely right! Lat first then lon
    )
    
    # Reproject both point and segments to a projected CRS for accurate distance calculations
    projected_crs = "EPSG:4326"  # Common projected CRS for distance calculations
    point_proj = point.to_crs(projected_crs)
    segments_gdf_proj = segments_gdf.to_crs(projected_crs)
    max_distance = float("inf")
    nearest_link_id = None
    for index, row in segments_gdf_proj.iterrows():
        distance = row.geometry.distance(point_proj.geometry[0])
        if distance < max_distance:
            max_distance = distance
            nearest_link_id = index
    return nearest_link_id




# loading the main dataframe
main_df = pd.read_csv("20181024_d1_0830_0900_traffic_segmented.csv")
# loading the traffic light states
with open("20181024_d1_0830_0900_traffic_light_states.json") as f:
    traffic_info = json.load(f)
# loading the segments
segments_gdf = gpd.read_file("20181024_d1_0830_0900_traffic_lights_segments.csv")
# Convert to GeoDataFrame
if "geometry" in segments_gdf.columns:
    # Convert 'geometry' to shapely objects if needed
    segments_gdf["geometry"] = segments_gdf["geometry"].apply(loads)
    # Create GeoDataFrame and set the CRS to WGS 84 (latitude/longitude)
    segments_gdf = gpd.GeoDataFrame(segments_gdf, geometry="geometry", crs="EPSG:4326")
else:
    raise ValueError("The DataFrame does not have a 'geometry' column.")

# Reproject to a projected CRS (replace EPSG code with appropriate UTM zone)
segments_gdf = segments_gdf.to_crs("EPSG:25832")  # Example for UTM Zone 32N

# Calculate lengths
segments_gdf["length"] = segments_gdf.geometry.length
segments_gdf = segments_gdf.to_crs("EPSG:4326")  # Example for UTM Zone 32N

lon_one_axis_trajectory = []
lat_one_axis_trajectory = []
for i, row in segments_gdf.iterrows():
    lat_one_axis_trajectory.append(lat_lon_to_axis(row.geometry.coords.xy[0][1], row.geometry.coords.xy[0][0], crs="EPSG:4326"))
    lon_one_axis_trajectory.append(lat_lon_to_axis(row.geometry.coords.xy[1][1], row.geometry.coords.xy[1][0], crs="EPSG:4326"))

segments_gdf["lon_one_axis_trajectory"] = lon_one_axis_trajectory
segments_gdf["lat_one_axis_trajectory"] = lat_one_axis_trajectory

# Calculate total length
total_length = segments_gdf["length"].sum()

# loading the traffic lights
traffic_lights_df = pd.read_csv("traffic_lights.csv")
# find traffic light segment id
traffic_lights_df["nearest_link_id"] = traffic_lights_df.apply(find_traffic_light_id, axis=1, segments_gdf=segments_gdf)
traffic_lights_df = traffic_lights_df.iloc[:5, :]
# loading the traffic light states
with open("traffic_info_dict.pkl", "rb") as f:
    traffic_lights_dict_states = pickle.load(f)


# loading the main dataframe
main_df = pd.read_csv("20181024_d1_0830_0900_traffic_segmented.csv")
# loading the traffic light states
with open("20181024_d1_0830_0900_traffic_light_states.json") as f:
    traffic_info = json.load(f)
# loading the segments
segments_gdf = gpd.read_file("20181024_d1_0830_0900_traffic_lights_segments.csv")
# Convert to GeoDataFrame
if "geometry" in segments_gdf.columns:
    # Convert 'geometry' to shapely objects if needed
    segments_gdf["geometry"] = segments_gdf["geometry"].apply(loads)
    # Create GeoDataFrame and set the CRS to WGS 84 (latitude/longitude)
    segments_gdf = gpd.GeoDataFrame(segments_gdf, geometry="geometry", crs="EPSG:4326")
else:
    raise ValueError("The DataFrame does not have a 'geometry' column.")

# Reproject to a projected CRS (replace EPSG code with appropriate UTM zone)
segments_gdf = segments_gdf.to_crs("EPSG:25832")  # Example for UTM Zone 32N

# Calculate lengths
segments_gdf["length"] = segments_gdf.geometry.length
segments_gdf = segments_gdf.to_crs("EPSG:4326")  # Example for UTM Zone 32N

lon_one_axis_trajectory = []
lat_one_axis_trajectory = []
for i, row in segments_gdf.iterrows():
    lat_one_axis_trajectory.append(lat_lon_to_axis(row.geometry.coords.xy[0][1], row.geometry.coords.xy[0][0], crs="EPSG:4326"))
    lon_one_axis_trajectory.append(lat_lon_to_axis(row.geometry.coords.xy[1][1], row.geometry.coords.xy[1][0], crs="EPSG:4326"))

segments_gdf["lon_one_axis_trajectory"] = lon_one_axis_trajectory
segments_gdf["lat_one_axis_trajectory"] = lat_one_axis_trajectory

# Calculate total length
total_length = segments_gdf["length"].sum()

# loading the traffic lights
traffic_lights_df = pd.read_csv("traffic_lights.csv")
# find traffic light segment id
traffic_lights_df["nearest_link_id"] = traffic_lights_df.apply(find_traffic_light_id, axis=1, segments_gdf=segments_gdf)
traffic_lights_df = traffic_lights_df.iloc[:5, :]
# loading the traffic light states
with open("traffic_info_dict.pkl", "rb") as f:
    traffic_lights_dict_states = pickle.load(f)


from CTM_classic import CTMParameters, initialize_density, update_cell_status
from sklearn.preprocessing import normalize


DISTANCE_THRESHOLD = 0.0001

# Initialize CTM parameters
ctm_params = CTMParameters()
ctm_params.segment_length = segments_gdf["length"].mean()
ctm_params.cell_length= 15

main_df_truncated = main_df[main_df["link_distance"] < DISTANCE_THRESHOLD].copy()  # Ensure it's a copy

# Normalize trajectory
normalized_trajectory = (
    main_df_truncated["one_axis_trajectory"] - main_df_truncated["one_axis_trajectory"].min()
)
normalized_trajectory = normalized_trajectory / normalized_trajectory.max() * segments_gdf["length"].sum()

# Assign to the DataFrame safely
main_df_truncated.loc[:, "normalized_trajectory"] = normalized_trajectory


from shapely.geometry import LineString
# Initialize densities for each segment
segment_densities_predicted = {idx: initialize_density(ctm_params, segments_gdf.iloc[idx]["length"]) for idx in segments_gdf.index}

def divide_linestring(linestring, n):
    """
    Divides a LINESTRING into n equal parts.

    :param linestring: A Shapely LineString object
    :param n: Number of equal parts to divide into
    :return: A list of LineString objects representing the segments
    """
    if not isinstance(linestring, LineString):
        raise ValueError("Input must be a Shapely LineString")

    # Get the total length of the LineString
    total_length = linestring.length

    # Calculate the step for division
    distances = np.linspace(0, total_length, n + 1)

    # Create parts
    parts = []
    for i in range(len(distances) - 1):
        start_point = linestring.interpolate(distances[i])
        end_point = linestring.interpolate(distances[i + 1])
        parts.append(LineString([start_point, end_point]))

    return gpd.GeoSeries(parts)

cells_linestring = []
for link_id, segment in segments_gdf.iterrows():
    # Divide the segment into cells
    cells = divide_linestring(segment["geometry"], len(segment_densities_predicted[link_id])) #‌ TODO check if the length of the cells is correct
    cells_linestring.append(cells)

# Flatten the cells into a new column for each segment
segments_gdf["cells_linestring"] = cells_linestring

# If you want to ensure individual cells are separate rows, you can explode the column
segments_gdf_exploded = segments_gdf.explode("cells_linestring").reset_index().rename(columns={"index": "link_id"})


# Group by `link_id` and collect `cells_linestring` into lists
grouped = segments_gdf_exploded.groupby("link_id")["cells_linestring"].apply(list).reset_index()

# Expand the lists into separate columns
expanded = pd.DataFrame(grouped["cells_linestring"].tolist(), index=grouped["link_id"])

# Add the other columns you want to retain (e.g., `geometry`, `length`, etc.)
metadata = segments_gdf_exploded.drop_duplicates(subset="link_id").set_index("link_id")[["geometry", "length", "lon_one_axis_trajectory", "lat_one_axis_trajectory"]]
segments_gdf_exploded = metadata.join(expanded)

# Rename the columns for clarity (optional)
segments_gdf_exploded.columns = list(metadata.columns) + [f"cell_{i+1}" for i in range(expanded.shape[1])]

# Reset index for a clean DataFrame
segments_gdf_exploded = segments_gdf_exploded.reset_index()
for column in segments_gdf_exploded.columns:
    if column.startswith("cell_"):
        segments_gdf_exploded[column] = gpd.GeoSeries(segments_gdf_exploded[column])

import pandas as pd

# Read the CSV file
main_df_truncated_with_cell = pd.read_csv("20181024_d1_0830_0900_segmented_oneaxistrajectory_cell.csv")

# Group by the desired columns and aggregate the vehicle IDs into a frozenset
grouped_with_veh_ids = main_df_truncated_with_cell.groupby(["link_id", "time", "closest_cell"]).agg({
    "veh_id": lambda x: frozenset(x)  # Use frozenset instead of set
}).reset_index()

# Merge the new column back into the original dataframe
main_df_truncated_with_cell = main_df_truncated_with_cell.merge(grouped_with_veh_ids, on=["link_id", "time", "closest_cell"], suffixes=('', '_list'))

# Rename the new column for clarity
main_df_truncated_with_cell.rename(columns={"veh_id_list": "veh_id_list"}, inplace=True)

# Display the shapes
duplicate_dropped = main_df_truncated_with_cell.drop_duplicates(subset=["link_id", "time", "closest_cell", "veh_id_list"])[["veh_id_list", "link_id", "time", "closest_cell"]]

# Filter the DataFrame for closest_cell == "cell_1"
cell_1 = duplicate_dropped[duplicate_dropped["closest_cell"] == "cell_1"]
_dict = {}

# Process each link_id group separately
for link_id in cell_1["link_id"].unique():
    # Filter the DataFrame for the current link_id
    link_df = cell_1[cell_1["link_id"] == link_id].copy()

    # Compute the deletion list as the set difference between the current and previous rows
    link_df["veh_id_list_deletion"] = [
        curr - prev if isinstance(curr, frozenset) and isinstance(prev, frozenset) else frozenset()
        for prev, curr in zip([frozenset()] * len(link_df), link_df["veh_id_list"].values)
    ]

    link_df["inflow"] = link_df["veh_id_list_deletion"].apply(len)
    _dict[link_id] = link_df[["inflow", "time"]]

inflow = pd.DataFrame({})
for index, value in _dict.items():
    value["link_id"] = index
    inflow = pd.concat([inflow, value])
inflow.to_csv("20181024_d1_0830_0900_inflow.csv")

speeds = main_df_truncated_with_cell['speed']


import numpy as np
free_flow_threshold = 25 
free_flow_speeds = speeds[speeds > free_flow_threshold]

mean_speed = np.mean(free_flow_speeds)
percentile_85 = np.percentile(free_flow_speeds, 85)



percentile_85_m_s = percentile_85/3.6
ctm_params.free_flow_speed = percentile_85_m_s


grouped = duplicate_dropped.groupby(["closest_cell", "link_id", "time"])
vehicle_count = grouped.size().reset_index(name="vehicle_count")
vehicle_count["density"] = vehicle_count["vehicle_count"] / ctm_params.cell_length


def get_density_for_time(link_id, time, vehicle_count):
    t0 = time
    density_t0 = vehicle_count[(vehicle_count["link_id"] == link_id) & (vehicle_count["time"] == t0)]
    density_t0_initialized = segment_densities_predicted[link_id].copy()
    for index, row in density_t0.iterrows():
        density_t0_initialized[int(row["closest_cell"].split("_")[-1])-1] = row["density"]


    t1 = t0 + ctm_params.time_step


    density_t1 = vehicle_count[(vehicle_count["link_id"] == link_id) & (vehicle_count["time"] == t1)]
    density_t1_initialized = segment_densities_predicted[link_id].copy()
    for index, row in density_t1.iterrows():
        density_t1_initialized[int(row["closest_cell"].split("_")[-1])-1] = row["density"]
    density_t1_initialized = np.array(density_t1_initialized)
    density_t0_initialized = np.array(density_t0_initialized)
    # print("density_density_t0_initialized", density_t0_initialized, "density_t1_initialized", density_t1_initialized)
    return density_t0_initialized, density_t1_initialized
actual_values_dict = {}
predicted_values_dict = {}
rmses_dict = {}
times_dict = {}
predict_mean_dict = {}
actual_mean_dict = {}


for link_id in vehicle_count["link_id"].unique():
    rmses = []
    times = []
    predict_mean = []
    actual_mean = []
    predicted_values = []
    actual_values = []
    for unique_time in vehicle_count[vehicle_count["link_id"] == link_id]["time"].unique(): 
        density_t0_initialized, density_t1_initialized = get_density_for_time(link_id, unique_time, vehicle_count)
        inflow_dt = inflow[(inflow["time"] == unique_time) & (inflow["link_id"] == link_id)]
        if inflow_dt.empty:
            inflow_dt = 0
        else:
            inflow_dt = inflow_dt["inflow"].values[0]
        predicted_den = np.array(update_cell_status(unique_time, link_id, density_t0_initialized, ctm_params, inflow_dt, traffic_lights_df, traffic_lights_dict_states))
        predict_mean.append(predicted_den[:].mean())
        predicted_value = predicted_den[:]
        actual_value = density_t1_initialized[:]

        rmse = np.sqrt(np.mean((predicted_value - actual_value)**2))
        predicted_values.append(predicted_value)
        actual_values.append(actual_value)
        actual_mean.append(density_t0_initialized[:].mean())
        rmses.append(rmse)
        times.append(unique_time)
    actual_values_dict[link_id] = actual_values
    predicted_values_dict[link_id] = predicted_values
    rmses_dict[link_id] = rmses
    times_dict[link_id] = times
    predict_mean_dict[link_id] = predict_mean
    actual_mean_dict[link_id] = actual_mean

from collections import defaultdict

all_rmses = defaultdict(lambda : defaultdict(list))
for link_id in vehicle_count["link_id"].unique():
    times = times_dict[link_id]
    rmses = rmses_dict[link_id]
    predict_mean = predict_mean_dict[link_id]
    actual_mean = actual_mean_dict[link_id]
    
    # plt.scatter(times[:], rmses[:], label="RMSE", s=0.5)
    # plt.scatter(times[:], actual_mean[:], label="Actual Mean", s=55, alpha=0.25)
    # plt.scatter(times[:], predict_mean[:], label="Predicted Mean", s=0.5, color="purple", alpha=1)
    # plt.xlabel("Time (s)")
    # plt.ylabel("RMSE")
    # plt.title("RMSE over time")
    # plt.hlines(np.mean(rmses), times[0], times[-1], label="Mean RMSE", color="red")
    # plt.hlines(vehicle_count[vehicle_count["link_id"] == link_id].density.mean(), times[0], times[-1], label="Mean Density", color="green")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # print("For link with id: ", link_id, " the RMSE is: ", np.mean(rmses))


from collections import defaultdict

heatmap_data = defaultdict(lambda : defaultdict(list))
min_i = 0
for link_id in times_dict.keys():
    for time_index in range(len(times_dict[link_id])):
        time = times_dict[link_id][time_index]
        a = actual_values_dict[link_id][time_index]
        p = predicted_values_dict[link_id][time_index]
        e2 = (a-p)**2
        heatmap_data[link_id][time] = e2



