from constants import CTMParameters
from helper import *
import math
# from point_queue_helper import cumulativecount_up, cumulativecount_down   # kian make this from your data


# load params
ctm_params = CTMParameters() # for now, we use same parameters we had in CTM model

# # define the function for finding cumulative counts
# N_upstr = cumulativecount_up
# N_downstr = cumulativecount_down

def N_upstr(time, segment_id):
    return (48 + 48*time/60) # for now, I just put a random number, you should replace it with the actual function

def N_downstr(time, segment_id):
    return 12 


# cumulative count at upstream at time t
# def N_0(time, segment_id, entr):


# returns the cumulative count of a characteristic at the location x at time t+horizon
def LTM_count(time, segment_id, entry_flow, location, horizon):
    # function parameters:
    # entry_Flow: flow of vehicles entering the link at the current time step, unit: vehicles per km
    # location: location in the segment in m, 0 <= location <= segment_length
    # horizon: time horizon for the cumulative count in seconds, start from "time" and end by "time + L/ffs"

    target1 = N_upstr(time + horizon - location/ctm_params.free_flow_speed, segment_id) # count for the uncongested route
    
    # find the point of congestion wave intersection with the characteristic, using for N(x_c)
    congestion_intersect_x1 = location + ctm_params.wave_speed*horizon # the intersection of the congestion wave with the segment, line 1
    congestion_intersect_x2 = ctm_params.segment_length # the intersection of the congestion wave with the segment, line 2
    if 0 <= congestion_intersect_x1 <= ctm_params.segment_length:
        congestion_intersect_x = congestion_intersect_x1 
    else:
        congestion_intersect_x = congestion_intersect_x2
    
    target2 = (ctm_params.jam_density_FD/1000)* (congestion_intersect_x - location) + N_downstr((ctm_params.wave_speed*horizon + location - ctm_params.segment_length)/ctm_params.wave_speed , segment_id) # count for the congested route
    # print("uncongested : " , target1,  " congested: " , target2)
    print("output:" , min(target1, target2))

    return min(target1, target2) 



# returns density and flow at the location x at time t+horizon using LTM cumulative counts
def LTM_states(time, segment_id, entry_flow, location, horizon):
    eps = ctm_params.free_flow_speed/100  # a small number, unit in meters
    density = (LTM_count(time, segment_id, entry_flow, location, horizon) - LTM_count(time, segment_id, entry_flow, location+eps, horizon)) / eps
    flow = -(LTM_count(time, segment_id, entry_flow, location, horizon) - LTM_count(time, segment_id, entry_flow, location, horizon+eps)) / eps
    print("density: ", density, " flow: ", flow)
    return density, flow

# test the function
time = 0
horizon = 30
location = 800
LTM_states(0, 1, 1, location, horizon) # expected output: 0.15, 0.15*26.8 = 4.02



