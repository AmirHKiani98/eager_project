#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import geopandas as gpd
from shapely.wkt import loads
from pyproj import Proj, Transformer


# In[2]:


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
    nearest_segment_id = None
    for index, row in segments_gdf_proj.iterrows():
        distance = row.geometry.distance(point_proj.geometry[0])
        if distance < max_distance:
            max_distance = distance
            nearest_segment_id = index
    return nearest_segment_id


# In[ ]:


# loading the main dataframe
main_df = pd.read_csv("20181024_d1_0830_0900_segmented_oneaxistrajectory_traffic.csv")
# loading the traffic light states
with open("20181024_d1_0830_0900_traffic_light_states.json") as f:
    traffic_info = json.load(f)
# loading the segments
segments_gdf = gpd.read_file("20181024_d1_0830_0900_segments.csv")
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
traffic_lights_df["nearest_segment_id"] = traffic_lights_df.apply(find_traffic_light_id, axis=1, segments_gdf=segments_gdf)
traffic_lights_df = traffic_lights_df.iloc[:5, :]
# loading the traffic light states
with open("traffic_info_dict.pkl", "rb") as f:
    traffic_lights_dict_states = pickle.load(f)


# In[ ]:





# In[4]:


plt.plot(segments_gdf["lon_one_axis_trajectory"], segments_gdf["lat_one_axis_trajectory"], "-", label="Road Segment")
plt.plot(traffic_lights_df["lon"], traffic_lights_df["lat"], "ro", label="Traffic Light")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Light Locations")
plt.legend()
plt.ticklabel_format(useOffset=False)
plt.grid()
plt.show()


# # Define the constants for truncating

# In[5]:


DISTANCE_THRESHOLD = 5


# # Normalize the one axis trajectory

# In[13]:


from CTM_classic import CTMParameters, initialize_density, update_cell_status
from helper import veh_to_density
from sklearn.preprocessing import normalize

# Initialize CTM parameters
ctm_params = CTMParameters()
ctm_params.segment_length = segments_gdf["length"].mean()

# Truncate points that are too far from the main corridor line
main_df_truncated = main_df[main_df["temp_distance"] < DISTANCE_THRESHOLD].copy()  # Ensure it's a copy
# main_df_truncated["one_axis_trajectory"] = main_df_truncated.apply(lambda row: lat_lon_to_axis(row["lat"], row["lon"], crs="EPSG:4326"), axis=1)

# Normalize trajectory
normalized_trajectory = (
    main_df_truncated["one_axis_trajectory"] - main_df_truncated["one_axis_trajectory"].min()
)
normalized_trajectory = normalized_trajectory / normalized_trajectory.max() * segments_gdf["length"].sum()

# Assign to the DataFrame safely
main_df_truncated.loc[:, "normalized_trajectory"] = normalized_trajectory


# In[14]:


main_df_truncated


# # Define the constants for CTM

# In[15]:


CELLS_NUM_PER_SEGMENT = 2


# In[21]:


# Initialize densities for each segment
segment_densities_predicted = {idx: initialize_density(ctm_params) for idx in segments_gdf.index}


# In[ ]:


from CTM_classic import update_cell_status

def simulate_corridor(segments_gdf, segment_densities_predicted, ctm_params, traffic_lights_df, traffic_lights_dict_states, total_time):
    # Initialize the inflow for the first segment
    entry_flow = 0  # Assuming no inflow at the start of the corridor

    # Simulate for a total time
    for t in np.arange(0, total_time, ctm_params.time_step):
        prev_outflow = entry_flow  # Outflow from the previous segment
        for segment_id in segments_gdf.index:
            # Get the current segment's density
            densities = segment_densities_predicted[segment_id]

            # Update densities for the segment
            new_densities = update_cell_status(
                t,
                segment_id,
                densities,
                ctm_params,
                prev_outflow,
                traffic_lights_df,
                traffic_lights_dict_states,
            )

            # Calculate the outflow from the last cell of the current segment
            last_cell_outflow = min(
                ctm_params.max_flow(),
                ctm_params.free_flow_speed * new_densities[-1] * ctm_params.time_step,
            )

            # Update the segment densities
            segment_densities_predicted[segment_id] = new_densities

            # Pass the outflow to the next segment
            prev_outflow = last_cell_outflow

        # Optionally: Store results or visualize densities at each time step
        print(f"Time {t}: {segment_densities_predicted}")

# Run the simulation
simulate_corridor(segments_gdf, segment_densities_predicted, ctm_params, traffic_lights_df, traffic_lights_dict_states, total_time=10)


# In[ ]:


ctm_params.cell_length


# In[ ]:





# In[28]:


traffic_lights_df


# In[ ]:




