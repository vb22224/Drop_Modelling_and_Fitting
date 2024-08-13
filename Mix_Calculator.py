# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:49:56 2024
Caculator to work out the realtive size of peaks in number density at a given mixing ratio
@author: vb22224
"""

import numpy as np
import Continuous_Trace_Model as ctm
import matplotlib.pyplot as plt



def plot_all_den(dp, den_list, y_label='frequency density / %',  dpi=600):

    'function to plot all the frequency densities'

    plt.figure(dpi=dpi)
    
    for den in den_list:
        plt.plot(dp, den)
        
    plt.xscale('log')
    plt.xlabel('d$_p$ / $\mu$m')
    plt.ylabel(y_label)
    
    return



def trap_int(x_array, y_array, x1, x2):
    
    """ Performs a numerical integration using the trapezium rule """
    
    
    def linear_interp(x, x1, x2, y1, y2):
        
        """ Performas a linear interpolation """

        y =  y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        
        return y
    
    total_integral = 0
    
    for i, x in enumerate(x_array):
        if x > x1 and x < x2:
            total_integral += (x - x_array[i - 1]) * (y_array[i] + y_array[i - 1]) / 2
            if x_array[i - 1] > x: # correction at beginning of trapezium
                y_x1 = linear_interp(x_array[i - 1], x1, x, y_array[i - 1], y_array[i])
                total_integral -= (x1 - x_array[i - 1]) * (y_x1 + y_array[i - 1]) / 2
    
    return total_integral



if __name__ == "__main__":
    
    # Inputs
    mix_ratio = [1, 1] # Relative amount of each mixture e.g. for a 50:50 mix: mix_amounts = [1, 1]
    all_mode_sizes = np.array([[0.9600932720257107], [0.9819259994956601]]) # list of each mixtures list of mode's relative sizes
    all_mode_means = np.array([[325.0976857912173], [600.0000147720889]]) # list of each mixtures list of mode's mean in um
    all_mode_stds = np.array([[0.09053671829707784], [0.11041574170111086]]) # list of each mixtures list of mode's standard deviation for its lognormal distribution
    
    # Preliminary calculations
    mix_ratio = np.array(mix_ratio) / np.sum(mix_ratio) # Normalising the mix ratio
    if all_mode_sizes.shape != all_mode_means.shape or all_mode_sizes.shape != all_mode_stds.shape: # Checks the array shapes match
        raise ValueError("Shapes of the mixtures sises, modes, and means do not match!")
    dp = np.logspace(np.log10(0.1), np.log10(10000), num=1000, base=10.0) # Evenly logspaced dp to veiw functions

    # Calculating the number density at each size for each mix
    vol_den_list = []
    for i, mode_sizes in enumerate(all_mode_sizes):
        vol_den = []
        total_size = ctm.check_modes(mode_sizes, all_mode_means[i], all_mode_stds[i])
        for d in dp:
            vol_den.append(ctm.calc_size_freq(d, mode_sizes, all_mode_means[i], all_mode_stds[i], total_size))
        vol_den_list.append(vol_den)
    
    # Plots and conversion
    vol_den_list = np.array(vol_den_list)
    plot_all_den(dp, vol_den_list, y_label='Volume density', dpi=200) # Plots the number densities to check they are ok
    
    vp = np.pi * np.power(np.array(dp), 3) / 6 
    num_den_list = vol_den_list / vp # Assumes all particels are spherical and is converting to volume distribution
    plot_all_den(dp, num_den_list, y_label='Number density', dpi=200)
    
    # Now integreate these curves
    num_ints, vol_ints, log_dp = [], [], np.log10(dp)
    for i, num_den in enumerate(num_den_list):
        num_ints.append(trap_int(log_dp, num_den, min(log_dp), max(log_dp)))
        vol_ints.append(trap_int(log_dp, vol_den_list[i], min(log_dp), max(log_dp)))
    
    # Normalisations
    vol_ints /= sum(vol_ints)
    num_ints /= sum(num_ints)
    
    # Print results
    print(f'Relative sizes of the modes in the volume distribution: {", ".join(map(str, vol_ints))}')
    print(f'Relative sizes of the modes in the number distribution: {", ".join(map(str, num_ints))}')
    
    
    