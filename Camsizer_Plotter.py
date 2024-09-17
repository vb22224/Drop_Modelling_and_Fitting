# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:59:22 2024
Code to plot camsizer data
@author: vb22224
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Mastersizer_Plotter as sdf
from scipy.optimize import minimize



def log_trap_int(x_array, y_array, x1=0, x2=float('inf')):
    
    """ Performs a numerical integration in logspace using the trapezium rule bewteen the limits x1 and x2 """

    x_array = np.log10(x_array)  # Normalising in logspace

    def linear_interp(x, x1, x2, y1, y2):
        
        """ Performas a linear interpolation """

        y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        return y

    total_integral = 0

    for i in range(1, len(x_array)):
        x = x_array[i]
        if x > x1 and x < x2:
            total_integral += (x - x_array[i - 1]) * \
                (y_array[i] + y_array[i - 1]) / 2
            if x_array[i - 1] > x:  # correction at beginning of trapezium
                y_x1 = linear_interp(
                    x_array[i - 1], x1, x, y_array[i - 1], y_array[i])
                total_integral -= (x1 - x_array[i - 1]) * \
                    (y_x1 + y_array[i - 1]) / 2

    return total_integral



def plot_frquency_density(dp, df, normalise=False, fit_type=False, mode_guess_list=[], mean_list=[], std_list=[], minimise_op='TR', dpi=200):
    
    fig, ax = plt.subplots(dpi=dpi)
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "c", "m", "y", "darkgoldenrod"]
    return_data = None

    for c, column in enumerate(df.columns[2:]):
        
        if normalise:
            df[column] /= log_trap_int(dp, df[column])  # normalisation
        
        ax.plot(dp, df[column], '-', color=color_list[c], label=column)  # Plotting the raw data
            
        if fit_type == 'n':  # No fit so only raw data plotted
            return_data = df.columns[2:]
        elif fit_type == 'g':  # Guess fit used to find a good initial guess
            frequency_density = sdf.multimodal_dist(dp, mode_guess_list[c], mean_list[c], std_list[c], get_modes=False)
        elif fit_type == 'f':  # Fitted distributions
            initial_guess = mode_guess_list[c] + mean_guess_list[c] + std_guess_list[c]
            n = int(len(initial_guess) / 3)
            result = minimize(sdf.objective_function, initial_guess, args=(dp, df[column], n, minimise_op, False), constraints={'type': 'ineq', 'fun': sdf.positivity_constraint})
            v = result.x
            mode_sizes, mode_means, mode_stds = v[:n], v[n:(n * 2)], v[(n * 2):]
            frequency_density = sdf.multimodal_dist(dp, mode_sizes, mode_means, mode_stds, get_modes=False)
            
            print(f'{column}:')
            print(f'mode_sizes = [{", ".join(map(str, mode_sizes))}]')
            print(f'mode_means = [{", ".join(map(str, mode_means))}]')
            print(f'mode_stds = [{", ".join(map(str, mode_stds))}]\n')
        else:
            raise ValueError(f'fit_type should be none (n), guess fit (g), or fitted (f).\n But instead: fit_type = {fit_type}')

        if fit_type in {'g', 'f'}:
            ax.plot(dp, frequency_density, '--', color=color_list[c])
            return_data = np.array(frequency_density) if return_data is None else np.vstack((return_data, np.array(frequency_density)))
    
    ax.set_xscale('log')
    ax.set_xlabel('x$_{c}$ $_{min}$ / $\mu$m')
    if normalise:
        ax.set_ylabel('Normalised Volume Frequency Density')
    else:
        ax.set_ylabel('Relative Frequency Density / %')
    ax.set_xlim([min(dp), 1000])

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    ax.legend(handles, labels)
    plt.show()

    return return_data



def plot_mixes(df, mix_data, mixes, y_label='Value', number=False, dpi=200):
    
    """ Plots the data for volume distributions of mixes """
    
    color_list = ["r", "g", "b", "c", "m", "y", "darkgoldenrod"]
    plt.figure(dpi=dpi)
    len_mixes = sum([len(sublist[0]) for sublist in mixes])
    vp = np.pi * np.power(np.array(dp), 3) / 6 # volumes at given sizes
    
    if len(mix_data) != len_mixes:
        raise ValueError(f'then lengths of the mix_data and mixes do not match at: {len(mix_data)} and {len_mixes}')
    
    for i, mix in enumerate(mixes):
        total_mix = np.sum(mix_data[mix[0]] * np.array(mix[1])[:, np.newaxis], axis=0)
        if number:
            total_mix /= vp
            total_mix /= log_trap_int(dp, total_mix)
        plt.plot(dp, total_mix.T, color=color_list[i], label=mix[2])
    
    # Options for plotting the firgure
    plt.xscale('log')
    plt.xlabel('x$_{c}$ $_{min}$ / $\mu$m')
    plt.ylabel(y_label)
    plt.xlim([min(dp), 10000])
    
    plt.legend()
    plt.show()
    


if __name__ == "__main__":

    file_path = '../Camsizer Data/Volcano_Final/sizes_csv.csv'
    # file_path = '../Camsizer Data/Sand/sizes_csv.csv'
    normalise = True # Normalise the data?
    fit_type = 'f' # No fit (n), guess fit (g), or fitted (f)
    # mean_guess_list = [[100], [200], [300], [500], [600], [800]] # List of list so can be multimodal if required
    # std_guess_list = [[0.27], [0.2], [0.1], [0.1], [0.15], [0.1]]
    mode_guess_list = [[1.0], [0.05, 0.85, 0.1], [1.0], [0.25, 0.6, 0.05]]
    mean_guess_list = [[100], [30, 90, 180], [100], [50, 110, 600]] # List of list so can be multimodal if required
    std_guess_list = [[0.08], [0.15, 0.15, 0.15], [0.08], [0.1, 0.2, 0.15]]
    minimise_op = 'R2' # Minimisation option, either total residual (TR), or R² value (R2)
    dpi = 600
    
    plot_mixes_op = False # For plotting mixtures of size fractions
    mixes = [[[3], [1.0], '355-500'], [[0, 6], [0.5, 0.5], 'MixA'], [[1, 5], [0.5, 0.5], 'MixB'], [[2, 4], [0.5, 0.5], 'MixC']] # List of which fractions are mixed, their relative contributions and labels
    
    df = pd.read_csv(file_path, sep=',')
    dp = df['Bin Centre']
    
    mix_data = plot_frquency_density(dp, df, normalise, fit_type, mode_guess_list, mean_list=mean_guess_list, std_list=std_guess_list, minimise_op=minimise_op, dpi=dpi)
    
    if plot_mixes_op:
        plot_mixes(df, mix_data, mixes, y_label='Normalised Volume Frequency Density', dpi=dpi) # Plot volume density 
        plot_mixes(df, mix_data, mixes, y_label='Normalised Number Frequency Density', number=True, dpi=dpi) # Plot number density
        
    