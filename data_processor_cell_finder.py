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



DISTANCE_THRESHOLD = 0.0001


from CTM_classic import CTMParameters, initialize_density, update_cell_status
from sklearn.preprocessing import normalize

# Initialize CTM parameters
ctm_params = CTMParameters()
ctm_params.segment_length = segments_gdf["length"].mean()
ctm_params.cell_length= 15

# Truncate points that are too far from the main corridor line
main_df_truncated = main_df[main_df["link_distance"] < DISTANCE_THRESHOLD].copy()  # Ensure it's a copy
# main_df_truncated["one_axis_trajectory"] = main_df_truncated.apply(lambda row: lat_lon_to_axis(row["lat"], row["lon"], crs="EPSG:4326"), axis=1)

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
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm  # Import tqdm for progress

def find_closest_cell(lat, lon, link_id, segments_gdf_exploded):
    """
    Find the nearest segment ID to a given point.
    
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        segments_gdf_exploded (GeoDataFrame): GeoDataFrame with segment geometries and cell data.
    
    Returns:
        str: ID of the nearest cell (e.g., 'cell_1').
    """
    # Create a GeoDataFrame for the point
    point = gpd.GeoDataFrame(
        geometry=[Point(lat, lon)], crs="EPSG:4326"
    )

    # Reproject both point and segments to a projected CRS for accurate distance calculations
    projected_crs = "EPSG:3395"  # Consider using a projected CRS suitable for distance measurements
    point_proj = point.to_crs(projected_crs)

    max_distance = float("inf")
    nearest_cell_id = None

    # Iterate through the segments and calculate distances
    for index, row in segments_gdf_exploded.iloc[[link_id]].iterrows():
        cell_columns = row.filter(regex=r'^cell_\d')  # Use raw string to avoid escape sequence warning
        if not cell_columns.empty:
            for cell_id, cell_value in cell_columns.items():
                if cell_value is None and int(cell_id.split("_")[1]) > len(segment_densities_predicted[index]):  # Skip empty cells
                    
                    continue
                # Assuming the cell_value corresponds to a geometry, e.g., a LineString or Polygon
                cell_geometry = segments_gdf_exploded.at[index, cell_id]
                # Convert to GeoSeries and reproject
                if isinstance(cell_geometry, gpd.GeoSeries):
                    cell_geometry = cell_geometry.to_crs(projected_crs)
                else:
                    cell_geometry = gpd.GeoSeries([cell_geometry], crs=segments_gdf_exploded.crs).to_crs(projected_crs).iloc[0]
                if cell_geometry and cell_geometry.is_valid:  # Ensure the geometry is valid
                    # Calculate the distance from the point to the cell geometry
                    distance = point_proj.distance(cell_geometry)

                    # Check if distance is scalar (just in case it's returned as a Series)
                    if isinstance(distance, (float, int)):  # Ensure it's a scalar
                        if distance < max_distance:
                            max_distance = distance
                            nearest_cell_id = cell_id  # Capture the ID of the nearest cell
                    else:
                        # Handle the case where distance is a Series or other unexpected result
                        distance = distance.item()  # Convert to scalar if it's a Series
                        if distance < max_distance:
                            max_distance = distance
                            nearest_cell_id = cell_id

    return nearest_cell_id

main_df_truncated_with_cell = main_df_truncated.copy()

main_df_truncated_with_cell["closest_cell"] = [
    find_closest_cell(row["lat"], row["lon"], int(row["link_id"]), segments_gdf_exploded) 
    for _, row in tqdm(main_df_truncated_with_cell.iterrows(), total=len(main_df_truncated_with_cell), desc="Processing Rows")
]

# print(f"The nearest cell IDs are: {main_df_truncated_with_cell['closest_cell']}")

main_df_truncated_with_cell.to_csv("20181024_d1_0830_0900_segmented_oneaxistrajectory_cell.csv")






