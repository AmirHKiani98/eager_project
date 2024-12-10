from constants import CTMParameters
from helper import *
import math

# Notes for Maziar:
## Do not use CTMParameters().object at all! If you use it, it will create a new object every time you call it. Instead, use the instance that you make of it!
## Try avoidng using the istance of CTMParemeters as a global object. Instead, pass it as an argument to the functions that need it.
## The docstring could be formatted for clarity: 1- Mention all parameters and their expected types! 2- Include an explanation of the function's output!
## Check for invalid or empty densities.
## Use descriptive variable names for better readability.
## Avoid magic numbers like 1 for the green light status; use constants instead.

# ctm_params = CTMParameters() # in this example, we use the default parameters


## Cell transmission model: update cell density
# arguments: densities: list of float + ctm_params: CTMParameters + entry_flow: float (flow of the first cell of the segment which comes from previous segments' outflow)
# " inflow is the number of vehicles enter from the previous cell to the current cell in one step"
# " outflow is the number of vehicles exit from the current cell to the next cell in one step"
# assumptions: jam density and max flow are constant for all cells
def update_cell_status(time, segment_id, densities, ctm_params, entry_flow, traffic_lights_df, traffic_lights_dict_states):
    num_cells = len(densities)
    new_densities = densities.copy()
    dt = ctm_params.time_step
    for i in range(num_cells):  # iterate over all cells
        if i == 0:      # first cell
            inflow = entry_flow # no inflow
        else:           # for all other cells: minimum of max flow and the flow from the previous cell
            inflow = min(ctm_params.max_flow(), ctm_params.free_flow_speed * densities[i-1] * dt, ctm_params.wave_speed * (ctm_params.jam_density - densities[i]) * dt)

        if i == num_cells - 1:  # last cell
            # check if there is a traffic light at the end of the segment
            if is_tl(segment_id, traffic_lights_df):
                # check the status of the traffic light
                if tl_status(time, segment_id, traffic_lights_df, traffic_lights_dict_states) == 1: # green light
                    outflow = min(ctm_params.max_flow(), ctm_params.free_flow_speed * densities[i] * dt, math.inf)
                else:
                    outflow = 0
            else:
                outflow = min(ctm_params.max_flow(), ctm_params.free_flow_speed * densities[i] * dt, math.inf)
        else:               # for all other cells: minimum of max flow and the flow to the next cell
            outflow = min(ctm_params.max_flow(), ctm_params.free_flow_speed * densities[i] * dt, ctm_params.wave_speed * (ctm_params.jam_density - densities[i+1]) * dt)

        new_densities[i] = densities[i] + (inflow - outflow) / ctm_params.cell_length   # n(t+1) = n(t) + (y(i) - y(i+1))/dx

    return new_densities


# test the function
# Density = [0.1, 0.15, 0, 0.2, 0.1, 0.1]
# entry_flow = 0
# time = 5
# updated_density = update_cell_status(time, 1, Density, ctm_params, entry_flow)
# print(updated_density)