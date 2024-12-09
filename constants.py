class CTMParameters:
    def __init__(self, time_step = 1 , cell_length=6, vehicle_length=5, free_flow_speed=15, wave_speed=5, jam_density=0.2, segment_length=100):
        """
        Initialize the parameters for the Cell Transmission Model.

        Args:
            cell_length (float): Length of each cell (meters).
            free_flow_speed (float): Free flow speed (meters/second).
            wave_speed (float): Backward wave speed (meters/second).
            jam_density (float): Maximum density (vehicles/meter).
        """
        # Default parameter values
        self.time_step = time_step
        self.cell_length = cell_length
        self.vehicle_length = vehicle_length
        self.free_flow_speed = free_flow_speed
        self.wave_speed = wave_speed
        self.jam_density = jam_density
        self.segment_length = segment_length
        
        

    def max_flow(self):
        """
        Calculate the maximum flow in the system based on the fundamental diagram.

        Returns:
            float: Maximum flow (vehicles/second).
        """
        return min(self.free_flow_speed, self.wave_speed) * self.jam_density / 2
    


