from constants import CTMParameters
from helper import *
import math
from point_queue_helper import cumulativecount_up, cumulativecount_down   # kian make this from your data

# Notes for Maziar:
## Do not use CTMParameters().object at all! If you use it, it will create a new object every time you call it. Instead, use the instance that you make of it!
## Try avoidng using the istance of CTMParemeters as a global object. Instead, pass it as an argument to the functions that need it.
## The docstring could be formatted for clarity: 1- Mention all parameters and their expected types! 2- Include an explanation of the function's output!
## Check for invalid or empty densities.
## Use descriptive variable names for better readability.
## Avoid magic numbers like 1 for the green light status; use constants instead.


# load params
ctm_params = CTMParameters() # for now, we use same parameters we had in CTM model

# define the function for finding cumulative counts
N_upstr = cumulativecount_up
N_downstr = cumulativecount_down

## spatial queue model: update number of vehicles in the link
# time: current simulation time
# arguments: N_upstrs: cumulative count of vehicles in upstream at a given time. I'll pass N_downstr(t) and it should tell me the cumulative count
# N_downstr: cumulative count at link downstream at time t (current time)
# entry_flow: for now constant, average number of vehicles entering to the link in vehicles per second
# assumptions: queue forms at the link downstream
def update_point_queue(time, segment_id, entry_flow, traffic_lights_df, traffic_lights_dict_states):

    #find the receiving flow, including the entry flow
    receceiving_flow = math.min( entry_flow*ctm_params.time_step, ctm_params.segment_length*ctm_params.jam_density - (N_upstr(time,segment_id) - N_downstr(time,segment_id)), ctm_params.max_flow*ctm_params.time_step)
    
    # check if there is a traffic light at the end of the segment
    if is_tl(segment_id, traffic_lights_df):
        # check the status of the traffic light
        if tl_status(time, segment_id, traffic_lights_df, traffic_lights_dict_states) == 1: # green light
            # find the link sending flow using point queue model 
            sending_flow = math.min( N_upstr(time + ctm_params.time_step - (ctm_params.segment_length/ctm_params.free_flow_speed), segment_id) - N_downstr, ctm_params.max_flow*ctm_params.time_step)      
        else:
            sending_flow = 0
    else: # no traffic light at the end of the link
        sending_flow = math.min( N_upstr(time + ctm_params.time_step - (ctm_params.segment_length/ctm_params.free_flow_speed), segment_id) - N_downstr, ctm_params.max_flow*ctm_params.time_step)

    # find the number of vehicles in the link at the next time step
    n_current = N_downstr(time) - N_upstr   # current number of vehicles
    n_updated = n_current + receceiving_flow - sending_flow 

    link_outflow = sending_flow
    link_density = n_updated / ctm_params.segment_length
    return link_density, link_outflow    # please note the order 

# # test the function
# Density = [0.1, 0.15, 0, 0.2, 0.1, 0.1]
# entry_flow = 0.1
# time = 5
# updated_density = update_point_queue(time, 1, Density, ctm_params, entry_flow)
# print(updated_density)