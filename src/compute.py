### Sharbat 2025, based off Timothee 2024

### To compute all the necessary values to the analysis

#All the things we need to compute : velocity along the axis, speed in general, Probability of action over time, cumulative probability of action during a time window

from math import sqrt
import numpy as np
from math import asin


# action_names = ["run","cast","stop","hunch","backup","roll","small_actions","backup_sequence","bend_static"]
# nb_action = [i for i in range(len(action_names))]
# action_dict = dict(zip(action_names,nb_action))

def array_with_nan(list):
    return np.array([np.nan if x is None else x for x in list])


def continuous_track(t,_time_max=1):
    """Function that return the time windows in which larva has been tracked"""
    time_windows = list()
    mem = t[0]
    for i in range(1,len(t)):
        diff_2_frames=t[i]-t[i-1] #difference between 2 frame
        if diff_2_frames > _time_max or i == len(t)-1 : #_time_max is the maximum value that we accept to tell that the tracking is continuous
            time_windows.append((mem,t[i-1]))
            mem = t[i]
    return time_windows




def compute_v_and_axis(trx_dict):
    """Function that computes the velocity according the x and y axis for each larva

    INPUT
    --------
    trx : trx_dict["data"]

    OUTPUT
    -------
    trx_with_velocity_axis : trx with velocity for each axis for each larva in new fields -> c_axis_x / c_axis_y
    """
    trx = trx_dict["data"]
    for larva in trx:

        c_axis_x = list()
        c_axis_y = list()
        speed = list()

        time = larva["t"]
        x_center = larva["x_center"]
        y_center = larva["y_center"]
        time_windows = continuous_track(time)

        j=0
        for t_start,t_stop in time_windows:
            while time[j]<t_stop:
                dx = x_center[j+1]-x_center[j]
                dy = y_center[j+1]-y_center[j]
                dt = time[j+1]-time[j]
                c_x = dx / dt
                c_y = dy / dt
                v = sqrt(c_x**2 + c_y**2)
                c_axis_x.append(c_x)
                c_axis_y.append(c_y)
                speed.append(v)
                j+=1
            c_axis_x.append(None)
            c_axis_y.append(None)
            speed.append(None)

        larva["c_axis_x"] = c_axis_x
        larva["c_axis_y"] = c_axis_y
        larva["time_window_continuous"] = time_windows
        larva["speed"] = speed
        array_length = array_with_nan(larva["larva_length_smooth_5"])
        larva["mean_size"] = np.nanmean(array_length)

        array_axis_x = array_with_nan(c_axis_x)
        array_axis_y = array_with_nan(c_axis_y)
        array_speed = array_with_nan(speed)
        larva["c_axis_x_normalized"] = array_axis_x / larva["mean_size"]
        larva["c_axis_y_normalized"] = array_axis_y / larva["mean_size"]
        larva["speed_normalized"] = array_speed / larva["mean_size"]

    trx_dict["data"] = trx
    return trx_dict

def compute_naviational_index(trx_dict):
    """
    :param trx_dict: trx that has already been computed through the compute_v_and_axis() function -> has speed; c_axis_x
        and c_axis_y fields

    :return: trx : Same trx as the one  NI_x and NI_y which are the 2 navigational indexes


    """
    all_means = list()
    trx_data = trx_dict["data"]
    for larva in trx_data :
        larva["mean_c_x"] = np.nanmean(larva["c_axis_x_normalized"])
        larva["mean_c_y"] = np.nanmean(larva["c_axis_y_normalized"])
        larva["mean_speed"] = np.nanmean(larva["speed_normalized"])

    trx_dict["data"] = trx_data

    all_means_c_x = [larva["mean_c_x"] for larva in trx_data]
    all_means_c_y = [larva["mean_c_y"] for larva in trx_data]
    all_mean_speed = [larva["mean_speed"] for larva in trx_data]

    m_c_x = np.mean(all_means_c_x)
    m_c_y = np.mean(all_means_c_y)
    m_speed = np.mean(all_mean_speed)

    trx_dict["NI_x"] = m_c_x / m_speed
    trx_dict["NI_y"] = m_c_y / m_speed

    return trx_dict


""" Functions to add to be able to compute the probabilities of actions over time :
	- t_indexes = [i if larva["t"] > t_avant and larva["t"] <= t_apres for i in range(nb_action)] -> give us the indexes of of the values in which the larva is between 2 time step (script)
	- Keep the data frame of the time step within the data <its depends on the time step of the script dt and the time step of the video> using the t_indexes list
	- Compute de probabilites within the time steps (video) and then make a mean for all the time steps of with T_l 
	- Do this for each action 
	- Nb of active larva : to do the probabilites -> Larva that perform at least one action during / is tracked between the time steps
	
	/!\ dt should be longer than the actual time step of the videos ? because it could do some empty values in the probabilities of action in each time steps
	
	-> Cumulative probability :
		Almost the same but we do an average over all larva over time 
		
	"""


def proba_action_over_time(trx_dict,action_dict,t_start,t_stop,dt):

    trx_dict["active_larva_over_time"] = list()
    trx_dict["proba_over_time"] = list()

    mem = t_start
    time = []
    str = "Larva_proba"

    while mem <= t_stop+1 :
        #print(time)
        time.append(mem)
        mem += dt



    trx_data = trx_dict["data"]
    for t in time[1:-1]:
        # A list that will contain all the probability of actions over each time t per larva
        # Each index correspond to the action dict corresponding key
        nb_active_larva = 0

        for larva in trx_data :

            list_proba_single_larva = list()
            # We compute the indexes from larva["t"] that are between 2 time steps (time)
            ind = list()
            for i in range(len(larva["t"])):
                # print(t)
                # print(larva["t"][i] > t-dt and larva["t"][i] <= t+dt)
                # print(larva["t"][i], larva["id"])
                if larva["t"][i] > t-dt and larva["t"][i] <= t+dt:
                    ind.append(i)

            # We convert all the float values from this field to int (used to be float)
            larva["global_state_large_state"] = [int(i) for i in larva["global_state_large_state"]]

            # If there are ind (= any actions occured within the 2 time steps)
            if ind :

                nb_active_larva +=1 #The larva is active

                # We compute all the probabilites for each actions
                # (=pourcentage of occurence of each action within the 2 time steps)

                for nb_action in action_dict.values():
                   bool_action = [1 if larva["global_state_large_state"][i] == nb_action else 0 for i in ind]
                   proba_action = np.mean(bool_action)
                   print(nb_action, proba_action)
                   list_proba_single_larva.append(proba_action)

            # Add this new computed value to the larva dictionary
            larva[str] = list_proba_single_larva
            print("len(Larva[str]) = ", len(larva[str]))

        trx_dict["active_larva_over_time"].append(nb_active_larva)

        # Probabilities over all the larvas in this time step
        for i in range(len(action_dict.values())) :
            proba_all_larva = np.array([larva[str][i] if larva[str] else np.nan for larva in trx_data])
            proba_mean = np.nansum(proba_all_larva)/nb_active_larva if nb_active_larva !=0 else np.nan
            array_sem = proba_mean * np.ones(len(proba_all_larva))
            proba_sem = np.nansum(array_sem - np.square(proba_all_larva))/nb_active_larva if nb_active_larva>1 else np.nan
            proba_sem = np.sqrt(proba_sem/(nb_active_larva-1))


            if t==time[1] :
                trx_dict["proba_over_time"].append([(proba_mean,proba_sem)])
            else :
                trx_dict["proba_over_time"][i].append((proba_mean,proba_sem))

    return  trx_dict


def points_to_vec(coord):

    # We take the coord in x and in y
    x_array = coord[0]
    y_array = coord[1]

    # We compute the vectors values
    v1 = np.array([x_array[0]-x_array[1],y_array[0]-y_array[1]])
    v2 = np.array([x_array[2]-x_array[1],y_array[2]-y_array[1]])

    return(v1,v2)


def relative_angle_2_vec(v1,v2):
    """We take the relative angles between the first vector defined by [x0 - x1 , y0 - y1] and the second vector defined
    by [x2 - x1 , y2 - y1]
    The angle will be in the angle in the counter clockwise direction

    /!\ We assume that the angle between 2 segments of the spine will not be less than pi/2, if this is not the case
     this will not work

    NB : To have the absolute angle of a vector compute the angle between your vector v=[x_v,y_v] and [1,0]

    :argument : coord : a list en the coord of the 3 points in this format : [[x1,x2,x3],[y1,y2,y3]]

    :return : relative_angle : the relative angle between the 2 vectors
    """
    # We compute the norm of those vectors
    norm_v1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Cross product of the 2 vectors -> We use this rather than the dot product because it is not a commutative
    #  operation
    v1_cross_v2 = np.cross(v1,v2)

    # We can the sin of the angle between the 2 vectors from this formula:
    sin_angle = v1_cross_v2 / (norm_v1 * norm_v2)

    angle = asin(sin_angle)
    # Angle > 0 if it is in [0;pi/2] and < if in ]pi; 3pi/2[

    # We translate this angle to have
    relative_angle = np.pi - angle
    return relative_angle

def angle_from_list(x_coord,y_coord,nb_point_spine,abs = False):

    angle_list = []
    #Relative angles
    if not abs :

        for i in range(1, nb_point_spine - 1):

            ind_triplet = [i-1,i,i+1]
            triplet_x = [x_coord[ind] for ind in ind_triplet]
            triplet_y = [y_coord[ind] for ind in ind_triplet]
            coord = [triplet_x,triplet_y]
            v1,v2 = points_to_vec(coord)
            angle = relative_angle_2_vec(v1,v2)
            angle_list.append(angle)

    #Aboslute angles
    else :

        for i in range(1, nb_point_spine - 1):

            ind_triplet = [i - 1, i, i + 1]
            triplet_x = [x_coord[i-1] for ind in ind_triplet]
            triplet_y = [y_coord[ind] for ind in ind_triplet]
            coord = [triplet_x,triplet_y]
            v1,v2 = points_to_vec(coord)
            v2 = [1,0]
            angle = relative_angle_2_vec(v1,v2)
            angle_list.append(angle)

    #angle_list = [round(((angle - np.pi) / np.pi) * 180,4) for angle in angle_list]

    return angle_list


def compute_angle(trx_dict):
    """ Function that compute all the relative angles of the larva for each time step from the spine coordinates
    :argument : trx_dict : already computed trx

    :return : trx dict modified
    """

    trx_data = trx_dict["data"]
    nb_point_spine = len(trx_data[0]["x_spine"]) #Should be equal to 11

    #For each larva
    for larva in trx_data :

        angle_relative = list()
        angle_absolute = list()

        #For each time step
        for t in range(len(larva['t'])):

            # We retrieve all the coordinates for a time step
            x_spine_t = [larva["x_spine"][j][t] for j in range(nb_point_spine)]
            y_spine_t = [larva["y_spine"][j][t] for j in range(nb_point_spine)]

            #
            angle_relative_t = angle_from_list(x_spine_t,y_spine_t,nb_point_spine)
            #angle_absolute_t = angle_from_list(x_spine_t,y_spine_t,nb_point_spine,abs=True)

            angle_relative.append(angle_relative_t)
            #angle_absolute.append(angle_absolute_t)

        larva["relative_angle"] = angle_relative

        #larva["absolute_angle"] = angle_absolute

        #each time step is a list of the different angles of the spine from the head to the tail

    trx_dict["data"] = trx_data

    return(trx_dict)