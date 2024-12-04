class CTMParameters:
    def __init__(self):
        """
        Initialize the parameters for the Cell Transmission Model.

        Args:
            cell_length (float): Length of each cell (meters).
            free_flow_speed (float): Free flow speed (meters/second).
            wave_speed (float): Backward wave speed (meters/second).
            jam_density (float): Maximum density (vehicles/meter).
        """
        # Default parameter values
        self.cell_length = 20  # meters
        self.vehicle_length = 5  # meters
        self.free_flow_speed = 15  # meters/second
        self.wave_speed = 5  # w: meters/second
        self.jam_density = 0.2  # vehicles/meter
        self.segment_length = 100

    def max_flow(self):
        """
        Calculate the maximum flow in the system based on the fundamental diagram.

        Returns:
            float: Maximum flow (vehicles/second).
        """
        return min(self.free_flow_speed, self.wave_speed) * self.jam_density / 2

    # def __repr__(self):
    #     """
    #     String representation of the CTM parameters.
    #     """
    #     return (
    #         f"CTMParameters("
    #         f"cell_length={self.cell_length}, "
    #         f"free_flow_speed={self.free_flow_speed}, "
    #         f"wave_speed={self.wave_speed}, "
    #         f"jam_density={self.jam_density}, "
    #         f"max_flow={self.max_flow():.2f})"
    #     )

def initialize_density(initial_density=0):
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
    num_cells = int(CTMParameters().segment_length / CTMParameters().cell_length)
    
    # Create a list of densities for all cells
    densities = [initial_density] * num_cells

    return densities


ctm_params = CTMParameters()
## Cell transmission model: update cell density
def update_cell_status(densities):
    num_cells = len(densities)
    new_densities = densities.copy()
    for i in range(num_cells):  # iterate over all cells
        if i == 0:      # first cell
            inflow = 0
        else:           # for all other cells
            inflow = min(ctm_params.free_flow_speed * densities[i-1], ctm_params.wave_speed * (ctm_params.jam_density - densities[i]))

        if i == num_cells - 1:  # last cell
            outflow = 0
        else:
            outflow = min(ctm_params.free_flow_speed * densities[i], ctm_params.wave_speed * (ctm_params.jam_density - densities[i+1]))

        new_densities[i] = densities[i] + (inflow - outflow) / ctm_params.cell_length   # n(t+1) = n(t) + (y(i) - y(i+1))/dx

    return new_densities
