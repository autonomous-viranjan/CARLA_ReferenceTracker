# Author: Viranjan Bhattacharyya
# Autonomous Control Systems Lab, Illinois Institute of Technology
# June 2021

import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass


import carla
import csv
import random
import time
import numpy as np
import math
from agents.navigation.controller import VehiclePIDController

actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    def draw_waypoints(waypoints, road_id=None, lane_id=None, life_time=50.0):
        """Fuction to draw waypoints on lanes"""

        for waypoint in waypoints:

            if(waypoint.road_id == road_id and waypoint.lane_id == lane_id):
                world.debug.draw_string(waypoint.transform.location, 'x', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=life_time, persistent_lines=True)

    def find_target_waypoint(road, ref_point):
        """
            This function takes a reference point and 
            returns the index of nearest point on road
            road: input waypoints in road lanes
            ref_point: reference point
            
        """
     
        dis = np.zeros(len(road)) 
        for j in range(len(road)):
            dis[j] = math.sqrt((road[j][0] - ref_point[0])**2 + (road[j][1] - ref_point[1])**2)
        
        p = np.argmin(dis)
        return p
    
    
    # Generate and draw road waypoints                                    
    waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)
    # draw_waypoints(waypoints, road_id=3, lane_id=-2, life_time=20)
    # draw_waypoints(waypoints, road_id=3, lane_id=-1, life_time=20)

    # Vehicles
    vehicle_blueprint = client.get_world().get_blueprint_library().filter('model3')[0]
    neighbor_blueprint = client.get_world().get_blueprint_library().filter('tt')[0]
    
    # Road waypoints list
    filtered_waypoints = []
    for waypoint in waypoints:
        if(waypoint.road_id == 3 and (waypoint.lane_id == -2 or waypoint.lane_id == -1)):
            filtered_waypoints.append(waypoint)

    # for i in range(len(filtered_waypoints)):
    #     print(filtered_waypoints[i].transform.location)

    # Save road lane waypoints as a text file
    road = np.zeros((len(filtered_waypoints), 2))
    for i in range(len(filtered_waypoints)):
        road[i][0] = filtered_waypoints[i].transform.location.x
        road[i][1] = filtered_waypoints[i].transform.location.y
    np.savetxt('road_waypoints.txt', road, fmt='%f', delimiter=' ', newline='\n')

    # Load Reference txt files        
    ref1 = np.loadtxt("waypoints1.txt", unpack=False)
    
    ref2 = np.loadtxt("waypoints2.txt", unpack=False)
           
    # Find points in 'filtered_waypoints' near 'ref' as Targets
    targetIdx1 = []    
    for i in range(len(ref1)):
        targetIdx1.append(find_target_waypoint(road, ref1[i]))
    
    targetIdx2 = []
    for i in range(len(ref2)):
        targetIdx2.append(find_target_waypoint(road, ref2[i]))

    target_waypoint1 = []
    for i in range(len(targetIdx1)):
        target_waypoint1.append(filtered_waypoints[targetIdx1[i]])
    
    target_waypoint2 = []
    for i in range(len(targetIdx2)):
        target_waypoint2.append(filtered_waypoints[targetIdx2[i]])

    targetNbr1 = filtered_waypoints[208]
    targetNbr2 = filtered_waypoints[209]
    
    # Spawn vehicles
    spawn_point1 = filtered_waypoints[targetIdx1[0]].transform
    spawn_point1.location.z += 2
    vehicle1 = client.get_world().spawn_actor(vehicle_blueprint, spawn_point1)

    spawn_point2 = filtered_waypoints[targetIdx2[0]].transform
    spawn_point2.location.z += 2
    vehicle2 = client.get_world().spawn_actor(vehicle_blueprint, spawn_point2)

    spawn_point3 = filtered_waypoints[36].transform
    spawn_point3.location.z += 2
    nbr1 = client.get_world().spawn_actor(neighbor_blueprint, spawn_point3)

    spawn_point4 = filtered_waypoints[39].transform
    spawn_point4.location.z += 2
    nbr2 = client.get_world().spawn_actor(neighbor_blueprint, spawn_point4)
    
    custom_controller1 = VehiclePIDController(vehicle1, args_lateral = {'K_P': 1, 'K_D': 15.0, 'K_I': 0.2}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})
    custom_controller2 = VehiclePIDController(vehicle2, args_lateral = {'K_P': 1, 'K_D': 15.0, 'K_I': 0.2}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

    custom_controller3 = VehiclePIDController(nbr1, args_lateral = {'K_P': 1, 'K_D': 15.0, 'K_I': 0.2}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})
    custom_controller4 = VehiclePIDController(nbr2, args_lateral = {'K_P': 1, 'K_D': 15.0, 'K_I': 0.2}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

    i=0
    j=0
    veh1locList = []
    veh2locList = []
    nbr1locList = []
    nbr2locList = []
    while True:    
        # Get Vehicle Locations  
        vehicle_loc1 = vehicle1.get_location()
        veh1locList.append([vehicle_loc1.x, vehicle_loc1.y])
        vehicle_loc2 = vehicle2.get_location()     
        veh2locList.append([vehicle_loc2.x, vehicle_loc2.y])

        nbr_loc1 = nbr1.get_location()     
        nbr1locList.append([nbr_loc1.x, nbr_loc1.y])
        nbr_loc2 = nbr2.get_location()     
        nbr2locList.append([nbr_loc2.x, nbr_loc2.y])

        ### Control for Vehicle 1 ###
        dist1 = math.sqrt( (target_waypoint1[i].transform.location.x - vehicle_loc1.x)**2 + (target_waypoint1[i].transform.location.y - vehicle_loc1.y)**2 )
        desiredSpeed1 = ref1[i][2]
        # print ("Distance before the loop is ", dist)
        control_signal1 = custom_controller1.run_step(desiredSpeed1, target_waypoint1[i])
        vehicle1.apply_control(control_signal1)
    
        if i==(len(ref1)-1):
            print("last waypoint reached for vehicle 1")
            break
            
        if (dist1<3.5):
            # print ("The distance is less than 3.5")  
            #Get next way point only when the distance between our vehicle and the current                                
            #waypoint is less than 3.5 meters 
            
            desiredSpeed1 = ref1[i][2]
            control_signal1 = custom_controller1.run_step(desiredSpeed1, target_waypoint1[i])
            vehicle1.apply_control(control_signal1)
            i=i+1
        
        ### Control for Vehicle 2 ###
        dist2 = math.sqrt( (target_waypoint2[j].transform.location.x - vehicle_loc2.x)**2 + (target_waypoint2[j].transform.location.y - vehicle_loc2.y)**2 )
        desiredSpeed2 = ref2[j][2]
        
        # print ("Distance before the loop is ", dist)
        control_signal2 = custom_controller2.run_step(desiredSpeed2, target_waypoint2[j])
        vehicle2.apply_control(control_signal2)
    
        if j==(len(ref2)-1):
            print("last waypoint reached for vehicle 2")
            break
    
        if (dist2<3.5):
            # print ("The distance is less than 3.5")  
            #Get next way point only when the distance between ego vehicle and the current                                
            #waypoint is less than 3.5 meters 
            
            desiredSpeed2 = ref2[j][2]
            control_signal2 = custom_controller2.run_step(desiredSpeed2, target_waypoint2[j])
            vehicle2.apply_control(control_signal2)
            j=j+1

        ### Control for Neighbor 1 ###
        desiredSpeed3 = 12
        
        # print ("Distance before the loop is ", dist)
        control_signal3 = custom_controller3.run_step(desiredSpeed3, targetNbr1)
        nbr1.apply_control(control_signal3)

        ### Control for Neighbor 2 ###
        desiredSpeed4 = 12
        
        # print ("Distance before the loop is ", dist)
        control_signal4 = custom_controller4.run_step(desiredSpeed4, targetNbr2)
        nbr2.apply_control(control_signal4)   
       
    
    veh1traj = np.array(veh1locList)
    veh2traj = np.array(veh2locList)
    np.savetxt('veh1_trajectory.txt', veh1traj, fmt='%f', delimiter=' ', newline='\n')
    np.savetxt('veh2_trajectory.txt', veh2traj, fmt='%f', delimiter=' ', newline='\n')

    actor_list.append(vehicle1)
    actor_list.append(vehicle2)
    actor_list.append(nbr1)
    actor_list.append(nbr2)
    # sleep for 10 seconds, then finish:
    time.sleep(1)

finally:
    print('Cleaning up actors...')
    for actor in actor_list:
        actor.destroy()
    print('Done, Actors cleaned-up successfully!')