import pandas as pd
import numpy as np
from tqdm import tqdm




file_path = "20181024_d1_0830_0900.csv" # Is the same as 20181024_d1_0830_0900.csv





import pandas as pd
def chunk_list(lst, chunk_size=6):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def parse_vehicle_data(file_path):
    return_dict = {}
    with open(file_path) as f:
        i = 0
        for line in f:
            i+=1
            split = line.replace(" ", "").strip().split(";")
            track_id = split[0]
            split = line.replace(" ", "").strip().split(";")[4:]
            
            if i != 1:
                chunk = chunk_list(split)
            
                if len(chunk[-1]) == 1 and chunk[-1][0] == "":
                    chunk = chunk[:-1] 
                if track_id in return_dict:
                    print("whaaat?")
                return_dict[track_id] = pd.DataFrame(chunk, columns=["lat", "lon", "speed", "lon_acc", "lat_acc", "time"])
                
    return return_dict
            
print("Info dict processing started.")
info_dict = parse_vehicle_data(file_path)
print("Info dict processed.")




print("Making the main df.")
main_df = pd.DataFrame()
for key, df in info_dict.items():
    df["veh_id"] = key
    main_df = pd.concat([main_df, df], axis=0)
print("The main df is produced.")


from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd

# Define the coordinates
traffic_lights_coordinates = pd.read_csv("traffic_lights.csv")

# Create a list of lines and IDs for each pair of consecutive coordinates
lines = []

for index in traffic_lights_coordinates.index:
    if index == traffic_lights_coordinates.index[-1]:
        break
    lines.append(LineString([traffic_lights_coordinates.iloc[index].values, traffic_lights_coordinates.iloc[index + 1].values]))

# Create a pandas DataFrame
segments_traffic_lights = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
segments_traffic_lights.to_csv("traffic_lights_segments.csv", index=False)


import geopandas as gpd
from shapely.geometry import Point
from IPython.display import display, HTML, clear_output

i = 0

def find_segment_for_point(segments_gdf, point: Point, length_of_df=None):
    global i
    """
    Finds the segment(s) to which a point belongs and displays a live-updating progress bar.

    Args:
        segments_gdf (GeoDataFrame): GeoDataFrame containing segment geometries.
        point (Point): A Shapely Point object representing the point of interest.

    Returns:
        Index of the segment in segments_gdf that contains the point, or None if no match.
    """
    # # Perform spatial join to find intersecting segments
    # joined_gdf = gpd.sjoin(point_gdf, segments_gdf, how='inner', predicate='intersects')
    temp_distance = float("inf")
    segment_index = None
    for index, row in segments_gdf.iterrows():
        distance = row["geometry"].distance(point)
        if distance < temp_distance:
            temp_distance = distance
            segment_index = index
    if length_of_df is not None:
        # Update the progress bar
        i += 1
        
    
    # Return the matching segment index
    return segment_index, temp_distance



from pyproj import Proj, Transformer
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from IPython.display import display, HTML, clear_output


i = 0

def lat_lon_to_axis(lat, lon, axis_direction=(1, 0), crs="EPSG:3857", length_of_df=len(main_df)):
    global i
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
    # Update the progress bar
    if length_of_df is not None:
        i += 1
    return projection_value



import os
from shapely.geometry import Point
import csv
i = 0 # Global variable for find_segment_for_point


name_without_ext, _ = os.path.splitext(file_path)


if os.path.exists(name_without_ext + "_traffic_segmented_saved_index.txt"):
    index = int(open(name_without_ext + "_traffic_segmented_saved_index.txt", "r").read().strip())
else:
    index = 0

with open(name_without_ext + "_traffic_segmented_saved_index.txt", "w") as f_2:
    try:
        with open(name_without_ext + "_traffic_segmented.csv", "+a") as f:
            writer_created = False
            with tqdm(total=len(main_df.iloc[index:]), desc="Processing Data") as pbar:
                for index, row in main_df.iloc[index:].iterrows():
                    segment_index, distance = find_segment_for_point(segments_traffic_lights, Point(row['lat'], row['lon']))
                    main_df.at[index, "link_id"] = segment_index
                    row["link_id"] = segment_index
                    main_df.at[index, "link_distance"] = distance
                    row["link_distance"] = distance
                    one_axis_trajectory = lat_lon_to_axis(row['lat'], row['lon'], axis_direction=(1, 0))
                    main_df.at[index, "one_axis_trajectory"] = one_axis_trajectory
                    row["one_axis_trajectory"] = one_axis_trajectory
                    if writer_created == False:
                        f.write(",".join(map(str, row.keys())) + "\n")
                        writer_created = True
                    f.write(",".join(map(str, row.values)) + "\n")
                    pbar.update(1)  # Update progress bar
                    
    except Exception as e:
        print(e)
        f_2.write(str(index))
        f_2.close()
        f.close()
        raise e
    except KeyboardInterrupt:
        f_2.write(str(index))
        f_2.close()
        f.close()







