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



def log_trap_int(x_array, y_array, x1=0, x2=float('inf')):
    
    """ Performs a numerical integration in logspace using the trapezium rule bewteen the limits x1 and x2 """

    x_array = np.log10(x_array)  # Normalising in logspace
    
    if x1 == 0:
        x1 = min(x_array)
    if x2 == float('inf'):
        x2 = max(x_array)
    
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



def get_label(den_option):
    
    '''returns the correct y_label for the correcsponding den_option'''
    
    if den_option == 'N':
        y_label = 'Number Density / %'
    elif den_option == 'SA':
        y_label = 'Surface Area Density / %'
    elif den_option == 'V':
        y_label = 'Volume Density / %'
    else:
        raise ValueError(f'den_option should be number (N), surface area (SA), or volume (V), instead: {den_option}')
    
    return y_label



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



def plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, dp_fit, exp_data=[], dpi=600, min_dp=0.01):
    
    '''Plots a multimodal lognormal distribution and has the option to plot the experimental data'''
    
    mode_check(mode_sizes, mode_means, mode_stds)
    plt.figure(dpi=dpi)
    frequency_density, mode_fds = multimodal_dist(dp_fit, mode_sizes, mode_means, mode_stds, get_modes=True)
    
    for mode, fds in enumerate(mode_fds):
        plt.plot(dp_fit, fds, '-', label=f'Mode {mode + 1}')
        
    plt.plot(dp_fit, frequency_density, '--', color='black', label='fit')
    
    if len(exp_data) != 0:
        plt.plot(dp, exp_data, '-', color='black', label='Real Data')
    
    plt.xscale('log')
    plt.xlabel('d$_p$ / $\mu$m')
    plt.ylabel(get_label(den_option))
    plt.xlim([min_dp, max(dp)])
    plt.legend(loc='upper left') # loc='upper left'
    # plt.savefig('plot.pdf')
    plt.show()
    
    return frequency_density



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
    
    dp, num_den, n, minimise_op, show_steps = args
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
    
    if show_steps:
        # print(f'mode_sizes = {", ".join(map(str, mode_sizes))}')
        # print(f'mode_means = {", ".join(map(str, mode_means))}')
        # print(f'mode_stds = {", ".join(map(str, mode_stds))}')
        print(f'Total residual: {round(TR, 3)}, R² value: {round(R2, 4)}')
    
    return result



def positivity_constraint(variables):
    return variables



def get_all_dists(dp_fit, freq_den, den_option, dpi=200):
    
    '''Gets all the relevant distributions and plots them'''
    
    # Calculating the conversion factors at the fitted particle sizes
    vp_fit = np.pi * np.power(dp_fit, 3) / 6
    sa_fit = np.pi * np.power(dp_fit, 2)
    
    if den_option == 'V':
        v = freq_den
        n = v / vp_fit
        sa = n * sa_fit 
    elif den_option == 'SA':
        sa = freq_den
        n = sa / sa_fit
        v = n * vp_fit
    elif den_option == 'N':
        n = freq_den
        sa = n * sa_fit
        v = n * vp_fit
        
    # Normalisations
    n /= log_trap_int(dp_fit, n)
    sa /= log_trap_int(dp_fit, sa)
    v /= log_trap_int(dp_fit, v)
    
    # Plotting
    plt.figure(dpi=dpi)
    plt.plot(dp_fit, v, label='Volume')
    plt.plot(dp_fit, sa, label='Surface Area')
    plt.plot(dp_fit, n, label='Number')
    plt.xscale('log')
    plt.xlabel('d$_p$ / $\mu$m')
    plt.ylabel('Normalised Frequency Density')
    plt.xlim([min(dp_fit), max(dp_fit)])
    plt.legend()
    # plt.savefig('plot.pdf')
    plt.show()
    
    return n, sa, v



if __name__ == "__main__":

    file_name = 'Volcano_Volume_Density.csv'
    # file_name = 'Mastersizer G90.csv'
    
    df = pd.read_csv(file_name, sep=',')
    # print(df.columns)
    dp = df['Size'] # Diameters in um
    vol_den = df['Grimsvotn'] # Asumes the data is given ans volume distribution
    den_option = 'V' # Distribution type to convert to before fitting: number (N), surface area (SA), or volume (V)
    minimise_op = 'R2' # Minimisation option, either total residual (TR), or R² value (R2)
    fit_option = True # If True will fit the data. False is useful for finding an initial guess
    post_conversion = True # If true will convert the fit to all distributions
    dpi = 600
    
    mode_sizes = 0.008379052, 0.635798575, 6.150637405 # relative sizes of modes
    mode_means = 0.828113968, 32.99460371, 96.6914689 # means od modes / um
    mode_stds = 0.083381234, 0.348839354, 0.226022998 # modes standard distributions (logspace)
    
    vp = np.pi * np.power(np.array(dp), 3) / 6 # volumes at given sizes
    sa = np.pi * np.power(np.array(dp), 2) # surface area at given sizes
    num_den = vol_den / vp # Assumes all particels are spherical and is converting to number distribution
    sa_den = num_den * sa
    den = get_den(den_option, num_den, sa_den, vol_den)
    den /= log_trap_int(dp, den) # Normalisation
    dp_fit = np.logspace(np.log10(min(dp)), np.log10(max(dp)), 1000, base=10)
    freq_den = plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, dp_fit, exp_data=den, dpi=dpi, min_dp=0.1)
    
    if fit_option:
        
        initial_guess = mode_sizes + mode_means + mode_stds
        n = int(len(initial_guess) / 3)
        result = minimize(objective_function, initial_guess, args=(dp, den, n, minimise_op, True), constraints={'type': 'ineq', 'fun': positivity_constraint})
        val = result.x
        mode_sizes, mode_means, mode_stds = val[:n], val[n:(n * 2)], val[(n * 2):]
        fds = plot_multimodal(dp, mode_sizes, mode_means, mode_stds, den_option, dp_fit, exp_data=den, dpi=dpi, min_dp=0.1)
        
        print(f'mode_sizes = [{", ".join(map(str, mode_sizes))}]')
        print(f'mode_means = [{", ".join(map(str, mode_means))}]')
        print(f'mode_stds = [{", ".join(map(str, mode_stds))}]')
        
    if post_conversion: # This is for the giess not the fit! Rerun wit the fit result as the guess to get it for the fit!
        n, sa, v = get_all_dists(dp_fit, freq_den, den_option, dpi=dpi)
        
        
    
    
    