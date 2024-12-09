# Some functions that will be used as tool, often used.

from CTM_class import CTMParameters
cell_length = CTMParameters().cell_length
vehicle_length = CTMParameters().vehicle_length

def traffic_light(segment_id, time):
    pass # To Kian



# gets vehicle info and adds it to the cell density
def veh_to_density(densities, vehicle_position):
    """
    Adds a vehicle's density contribution to the appropriate cells.

    Args:
        densities (list of float): Current density values for each cell (vehicles/meter) in this link.
        vehicle_position (float): Position of the vehicle (meters from the start of the segment) in this link. Note this value is the middle of the vehicle
          Starts from 0 and ends by vehicle length.

    Returns:
        list of float: Updated densities with the vehicle's updated.
    """
    num_cells = len(densities)

    vehicle_position = vehicle_position - vehicle_length / 2  # get the start of the vehicle and use this as vehicle position

    # Calculate the start and end cell indices
    start_cell = int(vehicle_position // cell_length)
    end_cell = int((vehicle_position + vehicle_length) // cell_length)

    # Handle cases when the vehicle spans only one cell
    if start_cell == end_cell:
        densities[start_cell] += 1 / cell_length   ## add 1 veh/m to the cell density

    
    # vehicle is on the border of two cells:
    else:
        # Fractional overlap with the start cell
        start_fraction = ((start_cell + 1) * cell_length - vehicle_position )/ vehicle_length
        densities[start_cell] += start_fraction / cell_length

        # Fractional overlap with the end cell (if applicable)
        end_fraction = ((vehicle_position + vehicle_length) - end_cell * cell_length) / vehicle_length
        densities[end_cell] += end_fraction / cell_length

    return densities


# # test the function
# den = [0,0 ,0 , 0, 0]
# print(veh_to_density(den, 20))