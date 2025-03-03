# Some functions that will be used as tool, often used.

from constants import CTMParameters

def traffic_light(segment_id, time):
    pass # To Kian





# # test the function
# den = [0,0 ,0 , 0, 0]
# print(veh_to_density(den, 20))


# returns whether there is a traffic light at the end of the segment or not
def is_tl(segment_id, traffic_lights_df):
    if segment_id in traffic_lights_df.nearest_segment_id.values:
        return True
    return False

# returns status of the traffic light, green = 1, red = 0
# note for Amir: you probably need to add (time, segment id) arguments to this function
def tl_status(time, segment_id, traffic_lights_df, traffic_lights_dict_states):
    
    # Now lets find the traffic light id in that segment
    if segment_id in traffic_lights_df.nearest_segment_id.values:
        tl_id = traffic_lights_df[traffic_lights_df.nearest_segment_id == segment_id].index[0]
    # Extract the dataframe out of that dictionary that I made in the ctm_model.ipynb
    traffic_light_status_df = traffic_lights_dict_states[tl_id]
    # Finding the closest time to the given time
    closest_idx = (traffic_light_status_df['time'] - time).abs().idxmin()
    # Getting the status of the traffic light at that time
    status = traffic_light_status_df.loc[closest_idx, 'traffic_status']
    if status == "green":
        return 1
    else:
        return 0
    




    
def initialize_density(ctm_parms, segment_length, initial_density=0):
    """
    Initialize densities for each cell in the segment.

    Args:
        segment_length (float): Total length of the segment (meters).
        cell_length (float): Length of each cell (meters).
        initial_density (float): Initial density for each cell (vehicles/meter).

    Returns:
        list of float: Initial densities for each cell.
    """
    # Calculate the number of cells
    num_cells = int(segment_length / ctm_parms.cell_length)
    
    # Create a list of densities for all cells
    densities = [initial_density] * num_cells

    return densities