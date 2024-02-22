# -*- coding: utf-8 -*-
"""
Code to fit a multimodal lognormal size distribution

Created: 21/02/24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statistics as stats
from scipy.stats import norm



def mode_check(mode_sizes, mode_means, mode_stds):
    
    '''Checks that the modes of a distribution match'''
    
    if len(mode_sizes) < 1:     
        raise ValueError("No mode sizes given")

    elif len(mode_sizes) != len(mode_means) or len(mode_sizes) != len(mode_stds):   
        raise ValueError("Mode parameter lengths differ")
    
    return



def get_den(den_option, num_den, sa_den, vol_den):
    
    '''Uses the den_option to return the required density distribution and performs a normalisation'''
    
    Total = np.sum(num_den)
    
    if den_option == 'N':
        den = num_den
    elif den_option == 'SA':
        den = sa_den * Total / np.sum(sa_den)
    elif den_option == 'V':
        den = vol_den * Total / np.sum(vol_den)
    else:
        raise ValueError(f'den_option should be number (N), surface area (SA), or volume (V), rather than: {den_option}')
    
    return den



def calc_size_freq(d, mode_sizes, mode_means, mode_stds, get_modes=False):
    
    '''Function to calulate the frequency denisity for a given particle size'''
    
    frequency_density, mode_fd = 0, []
    
    for number, mode_size in enumerate(mode_sizes):
        fd = mode_size * norm.pdf(np.log10(d), np.log10(mode_means[number]), mode_stds[number])
        frequency_density += fd
        
        if get_modes:
            mode_fd.append(fd)
            
    if get_modes:
        return frequency_density, mode_fd
    
    return frequency_density



def multimodal_dist(dp, mode_sizes, mode_means, mode_stds, get_modes=False):
    
    '''Calculates a multimodal distribution'''
    
    mode_check(mode_sizes, mode_means, mode_stds)
    frequency_density, mode_fds = [] , []
    
    for n in range(len(mode_sizes)):
        mode_fds.append([])
    
    for d in dp:
        if get_modes:
            output = calc_size_freq(d, mode_sizes, mode_means, mode_stds, get_modes=True)
            frequency_density.append(output[0])
            mode_fd =output[1]
            
            for mode, fd in enumerate(mode_fd):
                mode_fds[mode].append(fd)
            
        else:
            frequency_density.append(calc_size_freq(d, mode_sizes, mode_means, mode_stds))
    
    if get_modes:
        return frequency_density, mode_fds
    
    return frequency_density



def plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, len_fit=1000, exp_data=[], dpi=600, min_dp=0.01):
    
    '''Plots a multimodal lognormal distribution and has the option to plot the experimental data'''
    
    if den_option == 'N':
        y_label = 'Number Density / %'
    elif den_option == 'SA':
        y_label = 'Surface Area Density / %'
    elif den_option == 'V':
        y_label = 'Volume Density / %'
    
    mode_check(mode_sizes, mode_means, mode_stds)
    dp_fit = np.logspace(np.log10(min(dp)), np.log10(max(dp)), len_fit, base=10)
    plt.figure(dpi=dpi)
    frequency_density, mode_fds = multimodal_dist(dp_fit, mode_sizes, mode_means, mode_stds, get_modes=True)
    
    for mode, fds in enumerate(mode_fds):
        plt.plot(dp_fit, fds, '-', label=f'Mode {mode + 1}')
        
    plt.plot(dp_fit, frequency_density, '--', color='black', label='fit')
    
    if len(exp_data) != 0:
        plt.plot(dp, exp_data, '-', color='black', label='Real Data')
    
    plt.xscale('log')
    plt.xlabel('dp / um')
    plt.ylabel(y_label)
    plt.xlim([min_dp, max(dp)])
    plt.legend(loc='upper left')
    plt.show()
    
    return



def calc_charge_res(x_list, y_list, y_fit):
    
    '''Calculates the total residual and R² value between a set of values and a fit'''
    
    TR, SSE, SST = 0, 0, 0
    y_av = stats.mean(y_list)
    
    for n, y in enumerate(y_list):
        R = np.abs(y_fit[n] - y_list[n])
        TR += R
        SSE += R ** 2
        SST += (y - y_av) ** 2
        
    R2 = 1 - SSE / SST
    
    return TR, R2
    


def objective_function(variables, *args):
    
    '''Used for the function minimisation with scipy.minimize'''
    
    dp, num_den, n, minimise_op = args
    v = variables
    mode_sizes, mode_means, mode_stds = v[:n], v[n:(n * 2)], v[(n * 2):]
    
    y_fit = multimodal_dist(dp, mode_sizes, mode_means, mode_stds, get_modes=False)
    TR, R2 = calc_charge_res(dp, num_den, y_fit)
    
    if minimise_op == "TR":
        result = TR
    elif minimise_op == "R2":
        result = 1 - R2
    else:
        raise ValueError(f"minimise_op should be either total residual (TR), or R² value (R2). Not: {minimise_op}.")
        
    print(f'Total residual: {round(TR, 3)}, R² value: {round(R2, 4)}')
    
    return result



def positivity_constraint(variables):
    return variables



if __name__ == "__main__":

    file_name = 'Volcano_Volume_Density.csv'
    df = pd.read_csv(file_name, sep=',')
    dp = df['Size'] # Diameters in um
    vol_den = df['Grimsvotn'] # Asumes the data is given ans volume distribution
    den_option = 'SA' # Number (N), surface area (SA), or volume (V)
    
    num_den = (6 * vol_den / np.pi) ** (1 / 3) # Assumes all particels are spherical  and is converting to number distribution
    sa_den = 4 * np.pi * num_den ** 2
    den = get_den(den_option, num_den, sa_den, vol_den)
    
    minimise_op = 'TR' # Minimisation option, either total residual (TR), or R² value (R2)
    mode_sizes = 0.10, 0.7, 2 # relative sizes of modes
    mode_means = 0.8, 10, 80 # means of modes / um
    mode_stds = 0.09, 0.35, 0.3 # modes standard distributions (logspace)
    
    # plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, len_fit=1000, exp_data=den, dpi=600, min_dp=0.1)
    
    initial_guess = mode_sizes + mode_means + mode_stds
    n = int(len(initial_guess) / 3)
    result = minimize(objective_function, initial_guess, args=(dp, den, n, minimise_op), constraints={'type': 'ineq', 'fun': positivity_constraint})
    v = result.x
    mode_sizes, mode_means, mode_stds = v[:n], v[n:(n * 2)], v[(n * 2):]
    plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, len_fit=1000, exp_data=den, dpi=600, min_dp=0.1)
    print(f'mode_sizes = {mode_sizes}')
    print(f'mode_means = {mode_means}')
    print(f'mode_stds = {mode_stds}')
    
    
    
    