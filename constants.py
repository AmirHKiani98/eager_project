class CTMParameters:
    def __init__(self , cell_length=6, vehicle_length=5, free_flow_speed=15, wave_speed=5, segment_length=40, num_lanes=3):
        """
        Initialize the parameters for the Cell Transmission Model.

        Args:
            cell_length (float): Length of each cell (meters).
            free_flow_speed (float): Free flow speed (meters/second).
            wave_speed (float): Backward wave speed (meters/second).
            jam_density (float): Maximum density (vehicles/meter).
        """
        # Default parameter values
        self.time_step = cell_length / free_flow_speed
        self.num_lanes = num_lanes
        self.cell_length = cell_length
        self.vehicle_length = vehicle_length
        self.free_flow_speed = free_flow_speed
        self.wave_speed = wave_speed
        self.jam_density = 150/1000 * cell_length * self.num_lanes
        self.jam_density_link = 130  # jam density for the link: 130 vehicles per km
        self.jam_density_FD = 150 * self.num_lanes # jam density for the road, vehciles per km
        self.segment_length = segment_length
        self.max_flow_link = 2000 / 3600 * self.num_lanes # vehicle per seconds
        
        
        self.segment_length = segment_length
        self.max_flow_link = 2000 / 3600 * self.num_lanes # vehicle per seconds
        
        

    def max_flow(self):
        """
        Calculate the maximum flow in the system based on the fundamental diagram.

        Returns:
            float: Maximum flow (vehicles/second).
        """
        return min(self.free_flow_speed, self.wave_speed) * self.jam_density * self.num_lanes / 2 
    


