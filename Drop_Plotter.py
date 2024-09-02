# -*- coding: utf-8 -*-
"""
Code for Producing a Figure from Voltage Drop Data for multiple drops
Will plot Charge or Voltage
Can plot the trace, boxplot, or both
Seperated into functions

Created: 27/09/23
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statistics as stats
import math as m
import os
import json
from scipy.signal import find_peaks, peak_prominences
import numpy as np
import re
import random

    

def get_dif(d, change_type):
    
    """ Calculates the secified difference in a charge or voltage trace """
    
    if change_type == "mm":
        dif = max(d) - min(d)
    elif change_type == "sf":
        dif = d[-1] - d[0]
    elif change_type == "mmsf":
        dif = max(d) - min(d) - abs(d[-1] - d[0])
    else:
        raise ValueError("Invalid change type,  should be start to finish (sf), min and max (mm), of mm minus sf (mmsf)")
    return dif



def decimal_place(number):
    
    """ Returns the place value of the highest digit for rounding """
    
    # Convert the number to a string
    number_str = str(number)
    
    # Find the index of the first non-zero digit
    index = next((i for i, digit in enumerate(number_str) if digit != '0' and digit != '.'), None)
    
    # If index is None, it means all digits are zeros, return 0
    if index is None:
        return 0
    
    # If the first non-zero digit is after the decimal point, return its position
    if '.' in number_str:
        return index - number_str.index('.')
    
    # If the first non-zero digit is before the decimal point, return the negative of its position
    return -index



def get_times(sample_rates, data):
    
    """ Gets the time step, array of times and total time required for the trcae data """
    
    time_step = 1 / sample_rates[0]
    times = [i * time_step for i in range(len(data[0]))]
    time = len(data[0]) * time_step
    
    return time_step, times, time



def get_axis_title(p, adjust_by_mass=False):
    
    """ Looks up the axis title from parameter """
    
    if p == "V" or p == "RV":
        axis = "Voltage / V"
    elif p == "Q":
        axis = "Charge / pC"
        
    if adjust_by_mass:
        axis = "Specific " + axis + " g$^{-1}$"
        
    return axis



def custom_sort(file_names):
    
    """ Sorts the file names such that they will be in a logical order in the plot legend """
    
    def sorting_key(name):
        
        # Split the file name into parts of numbers and non-numbers
        parts = re.split(r'(-?\d+)', name)
        
        # Convert numeric parts to integers for proper numerical sorting, remove the - before int(part) to reverse
        parts = [-int(part) if part.lstrip('-').isdigit() else part for part in parts]
        
        return parts
    
    sorted_names = sorted(file_names, key=sorting_key)
    return sorted_names



def bootstrap_statistic(data, num_samples=1000, stat_func=stats.mean):
    
    """Generate bootstrap samples and calculate the statistic for each sample."""
    
    boot_samples = []
    
    for _ in range(num_samples):
        resample = [random.choice(data) for _ in range(len(data))]
        boot_samples.append(stat_func(resample))
        
    return boot_samples



def average_data(data, times, start_index=0, end_index=None, error_lines=False, av_option='mean'):
    
    """
    Averages data from the data list of traces in specified range satrt_index to end_index
    Clips this data to the shortest trace in the range and also returns the times required to plot this average
    Will also return the error bounds if error_lines = True
    """
    
    if end_index is None:
        end_index = len(data[0])
    
    valid_data = data[start_index:end_index]
    data_lengths, averaged_data, average_minus_error, average_plus_error = [], [], [], []
    bootstrap_samples = 1000
    
    for d in valid_data:
        data_lengths.append(len([x for x in d if not m.isnan(x)]))
    
    for n in range(min(data_lengths)):
        n_data = []
        for d in valid_data:
            n_data.append(d[n])
       
        if av_option == 'mean':
            av = stats.mean(n_data)
        elif av_option == 'median':
            av = stats.median(n_data)
        elif av_option == 'bootstrap':
            boot_means = bootstrap_statistic(n_data, num_samples=bootstrap_samples, stat_func=stats.mean)
            av = stats.mean(boot_means)
        else:
            raise ValueError(f"av_option should be 'mean', 'median', or 'bootstrap', instead: {av_option}")
        averaged_data.append(av)
        
        if error_lines:
            if len(n_data) > 1:
                SEM = stats.stdev(n_data) / stats.sqrt(len(n_data))
                average_minus_error.append(av - SEM)
                average_plus_error.append(av + SEM)
            else:
                average_minus_error, average_plus_error = 'NaN', 'NaN'
        
    av_times = times[:min(data_lengths)]
        
    return averaged_data, av_times, average_minus_error, average_plus_error



def calibration_constants(step_heights, peak_heights):
    
    """ Takes array of Faraday cup steps and ring probe peaks, returning the corresponding calibration constants and associated errors """
    
    if len(step_heights) > 0 and (len(peak_heights) == 1 or all(len(peak_heights[i]) == len(step_heights) for i in range(len(peak_heights)))):
        peak_heights = np.array(peak_heights)

        for ring, peaks in enumerate(peak_heights):
            if len(peaks) == len(step_heights):
                ratios = step_heights / peaks
                ratio_mean = stats.mean(ratios)
                ratio_sem = stats.stdev(ratios) / np.sqrt(len(peaks))
                print(f"The calibration constant of ring {ring + 1} is: {round(ratio_mean, 2)} ± {round(ratio_sem, 2)}")
            else:
                print(f"Warning: The number of peaks for ring {ring + 1} does not match the number of Faraday cup steps.")
    else:
        raise ValueError(f"Mismatch in lengths: Faraday cup steps [{len(step_heights)}] and ring peaks {[len(peak_heights[i]) for i in range(len(peak_heights))]}.")
        
    return



def get_steps(d, time_step, step_background_time, step_threshold, step_width):
    
    """ Takes a Faraday cup calibration trace and finds the steps produced by each droplet """

    background_stdev = stats.stdev(d[:int(step_background_time / time_step)])
    average_level = stats.mean(d[:int(step_background_time / time_step)])
    threshold = step_threshold * background_stdev
    step_width_index = int(step_width / time_step)

    step_indices, step_times, step_middles, step_heights = [], [], [], []

    for n, d_point in enumerate(d[:len(d) - step_width_index]):  # Scan over trace
        stdev = stats.stdev(d[n:n + step_width_index])

        if stdev > threshold and all(abs(n - i) > step_width_index for i in step_indices):
            step_indices.append(n)

    for n in step_indices:  # Going through each step working out their properties
        average_after = stats.mean(d[int(n + step_width_index):int(n + step_width_index * 2)])
        step_height = average_after - average_level
        step_middle = average_level + step_height / 2
        step_time = time_step * (n + np.argmin(np.abs(np.array(d[int(n):int(n + step_width_index)]) - step_middle)))

        step_heights.append(step_height)  # Appending values
        step_times.append(step_time)
        step_middles.append(step_middle)

        average_level = average_after  # Updating new height

    step_times, step_middles, step_heights = np.array(step_times), np.array(step_middles), np.array(step_heights)

    return step_heights, step_times, step_middles
    
    
    
def get_peaks(d, peak_std_dev_threshold, include_negatives=False):
    
    """ Takes a ring probe calibration trace and finds the peaks produced by each droplet """
    
    peak_threshold = np.std(d) * peak_std_dev_threshold
    peaks, properties = find_peaks(d)
    prominences = peak_prominences(d, peaks)[0]
    peak_height_row = []
    peak_indices = []
    
    if include_negatives:
        neg_peaks, neg_properties = find_peaks(-np.array(d))
        prominences = np.concatenate([peak_prominences(d, peaks)[0], peak_prominences(-np.array(d), neg_peaks)[0]])
        peaks = np.concatenate([peaks, neg_peaks])
    
    for peak_num, p in enumerate(peaks): 
        if prominences[peak_num] > peak_threshold:
            peak_height_row.append(prominences[peak_num])
            peak_indices.append(p)
    
    return peak_height_row, peak_indices
    
    

def average_multiple(collated_data, collated_times, filtered_names, parameter_lis=["Q", "RV", "RV"]):
    
    """ Takes data from multiple runs and averages them to the required format """
        
    averaged_names, lol_data, averaged_times, averaged_data = [], [], [], [[]]
    data_sets = int(len(collated_data) / len(parameter_lis))
    
    if data_sets != len(filtered_names):
        raise ValueError(f"Error with the number of files! Data sets: {data_sets}, filtered names: {filtered_names}")
    
    for i1, p in enumerate(parameter_lis):
        for i2, n in enumerate(filtered_names):
            name = n.split('_')[0] + '_' + p + '_' + str(parameter_lis[:i1].count(p) + 1)
            index = i1 * data_sets + i2
    
            if name not in averaged_names:
                averaged_names.append(name)
                lol_data.append([collated_data[index]])
                averaged_times.append(collated_times[index])
            else:
                lol_data[averaged_names.index(name)].append(collated_data[index])
        
    for i, lol in enumerate(lol_data):
        for d in range(len(lol[0])):
            total = 0
            
            for l in lol:
                total += l[d]
                
            av = total / len(lol)
            
            if d == 0:
                averaged_data.append([])
                
            averaged_data[i].append(av)  
    
    return averaged_data[:-1], averaged_times, averaged_names
    
    

def get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=[]):
    
    lol_structure, file_names = [], []
    
    if input_type == "dir" or input_type == "cat":
        target_directory = os.path.join(os.getcwd(), path)
        
        if not os.path.exists(target_directory):
            raise ValueError(f"Directory not found: {target_directory}")
            
        all_file_names = os.listdir(target_directory)
        
    elif input_type == "lis":
        all_file_names = file_names_lis
        
    elif input_type =="lol":
        all_file_names = []
        
        for lis in file_names_lol:
            lol_structure.append(0)
            for l in lis:
                all_file_names.append(l)
                lol_structure[-1] += 1
    else:
        raise ValueError("Invalid input_type, requires directory (dir), categorised (cat), list (lis), or list of lists (lol)")
        
    all_file_names = [file for file in all_file_names if file.endswith(".txt")]
    
    for remove_file in remove_files:
        all_file_names = [file for file in all_file_names if remove_file not in file]
        
    for file in all_file_names:
        if "README" not in file: # Ignores the README file
            try:
                read = open(os.path.join(path, file), "r", encoding="utf-8").read()
                if "Keithley" in read:
                    file_names.append(file)
                    
            except FileNotFoundError:
                print(f"File not found: {file}")
                
            except UnicodeDecodeError as e:
                raise ValueError(f"Error reading {file}: {e}")
    
    if input_type == "cat":
        
        if spec_cat:
            file_names = [f for f in file_names if f.split("_")[0] in spec_cat]
        
        # file_names.sort() # Simple sort
        file_names = custom_sort(file_names) # More complex sort
        
        lol_labels, lol_structure = [], []
        
        for f in file_names:
            if f.split("_")[0] in lol_labels:
                lol_structure[-1] += 1
            else:
                lol_labels.append(f.split("_")[0])
                lol_structure.append(1)
    #print(lol_labels, lol_structure)

    return file_names, lol_structure, lol_labels



def read_and_convert_data(file_names, target_directory, plot_parameter, ignore_len_errors, base_time=1, tolerance_amp=2, tolerance_len=100,
                          trim=False, manual_trim={}, store_dict=False, read_dict=False, data_col = 0, adjust_by_mass=False, adjust_for_decay=False, masses_name='Masses.csv', time_const=310.5):
    
    data, sample_rates, temperatures, humidities, filtered_names = [], [], [], [], []
    trim_file = os.path.join(target_directory, "manual_trim_dictionary")
    
    if adjust_by_mass:
        target_mass_dir = f'{target_directory}\{masses_name}'
        if os.path.exists(target_mass_dir):
            mass_df = pd.read_csv(target_mass_dir)
        else:
            raise FileNotFoundError(f"The file {target_mass_dir} does not exist.")
        
    
    if os.path.exists(trim_file):
        if read_dict==True:
            with open(trim_file, 'r') as json_file:
                manual_trim = json.load(json_file)
        else:
            print(f"{trim_file} does not exist")        
    
    if store_dict==True:
        with open(trim_file, 'w') as json_file:
            json.dump(manual_trim, json_file)
    
    for file in file_names:
        with open(os.path.join(target_directory, file), 'r') as f:
            for line_number, line in enumerate(f):
                if "Keithley" in line:
                    detected_header = line_number
        
        dat = pd.read_csv(os.path.join(target_directory, file), header= detected_header).values.tolist()
        converted_data = []
        
        if len(dat[0]) < data_col + 1:
            print(f"No column {data_col} in file: {file}")
        else:
            filtered_names.append(file)
            
            if adjust_by_mass:
                if file.split('.')[0] in mass_df['files'].values:
                    indices = np.where(mass_df['files'].values == file.split('.')[0])[0][0]
                    mass = mass_df['masses'][indices]
                else:
                    raise FileNotFoundError(f"{target_mass_dir} does not contain {file}.")
            else:
                mass = 1
            
            metadata = open(os.path.join(target_directory, file), "r").read().split("Keithley")[0]
            sample_rate = float(metadata.split(",")[1])
            sample_rates.append(sample_rate)
            last_d, total_offset = 0, 0 # no previous datapoint or offset
            
            if "Temperature" in metadata and "Humidity" in metadata:
                temperature = float(metadata.split(",")[-2].split(" ")[-2])
                humidity = float(metadata.split(",")[-1].split(" ")[-3])
                temperatures.append(temperature)
                humidities.append(humidity)
        
            for d in dat:
                if plot_parameter == "V":
                    converted_data.append(d[data_col] * 10 / mass)
                elif plot_parameter == "RV":
                    converted_data.append(d[data_col] / mass)
                elif plot_parameter == "Q":
                    converted_data.append(d[data_col] * 10 * 130 / mass)
                else:
                    raise ValueError("Invalid plot_parameter, requires charge (Q), voltage (V), or raw volatge(RV)")
                
                if adjust_for_decay or file.split("_")[0] == "Grimsvotn": # Adjusting for the decay due to the conductivity of air
                    time = 1 / sample_rate
                    decay = last_d * (1 - np.exp(- time / time_const)) # Standard exponential decay from the last timetep
                    total_offset += decay
                    last_d = converted_data[-1] # Storing datapoint before the ajustement
                    converted_data[-1] = converted_data[-1] + total_offset # The data ajusted for the decay
                
            data.append(converted_data)
    
    if len(sample_rates) == 0:
        raise ValueError(f'No output files found in directory: {target_directory}')
        
    elif len(set(sample_rates)) != 1:
        raise ValueError(f'Sample rates do not match: {sample_rates}')
    
    if trim: # Trimming off the start of each trace
        
        for count, f in enumerate(file_names): # Doing the manual trim
            if f in manual_trim:
                trim_time = manual_trim[f]
                trim_steps = trim_time * sample_rates[count]
                data[count] = data[count][int(trim_steps):]
            
        for count, dat in enumerate(data):
            
            samples = round(base_time * sample_rates[count])
            amp_s = max(dat[:samples]) - min(dat[:samples])
            
            for c, d in enumerate(dat):
                if len(dat) > c + tolerance_len:
                    if abs(d - dat[c + tolerance_len]) > amp_s * tolerance_amp:
                        # t = c / sample_rates[count]
                        # print(f"For {file_names[count]} trace starts at time: {t} s")
                        data[count] = data[count][c:]
                        break
                else:
                    print(f"No significant trace detected in {file_names[count]} at this significance level")
                    break
    
    if ignore_len_errors != "Ignore": 
        for count_1, d_1 in enumerate(data):
             for count_2, d_2 in enumerate(data):
                 if len(d_2) != len(d_1):
                     sr = sample_rates[count_1]
                     t_1 = len(d_1) / sr
                     t_2 = len(d_2) / sr
                     if ignore_len_errors == "Error":
                         print(f'Sample durations do not match: {t_1} s and {t_2} s in {file_names[count_1]} and {file_names[count_2]}.')
                     elif ignore_len_errors == "Extend":
                         if t_2 < t_1:
                             while len(data[count_2]) < len(data[count_1]):
                                 data[count_2].append(float('nan'))
                         elif t_2 > t_1:
                             while len(data[count_2]) > len(data[count_1]):
                                 data[count_1].append(float('nan'))
                     elif ignore_len_errors == "Crop":
                         if t_2 < t_1:
                             while len(data[count_2]) < len(data[count_1]):
                                 del data[count_1][-1]
                         elif t_2 > t_1:
                             while len(data[count_2]) > len(data[count_1]):
                                 del data[count_2][-1]
                     else:
                         raise ValueError('ignore_len_error should be "Ignore", "Crop", "Extend", or "Error".')
    
    return data, sample_rates, temperatures, humidities, filtered_names



def plot_figure(data, times, time, input_type, plot_parameter, file_names, target_directory, lol_structure, lol_labels, change_type, plot_option,
                temperatures=[], humidities=[], plot_average=True, show=True, get_av_data=False, Show_T_RH=False, simultaneous=False, error_lines='N',
                get_error=False, calibration_args=[False], adjust_by_mass=False, av_option='mean', label_sep_runs=False, dpi=200):

    """ Plots data with many options, for more information see the comments by each option in __main__ """
    
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "c", "m", "y", "darkgoldenrod", "yellowgreen", "indigo", "r", "g", "b"]
    loosely_dashed = (15, 15)

    if calibration_args[0] == True:
        calibration, step_times, step_middles, peak_indecies, peak_heights, time_step = calibration_args
    
    if get_error:
        error_lines='A'

    if plot_option != "Box" and plot_option != "Trace" and plot_option != "Both":
        raise ValueError('plot_option should be "Box" for box plot, "Trace" for just the trace, or "Both" for both plots.')
    
    if show:
        fig, ax1 = plt.subplots(dpi=dpi) # will fix at regular dimensions: figsize=(6, 4)
        
        if Show_T_RH == True and len(temperatures) > 0 and len(humidities) > 0:
            av_temp, av_RH = stats.mean(temperatures), stats.mean(humidities) 
            
            if len(temperatures) > 1 and len(humidities) > 1:
                sem_temp = stats.stdev(temperatures) / np.sqrt(len(temperatures))
                sem_RH = stats.stdev(humidities) / np.sqrt(len(humidities))
                temp_round, RH_round = decimal_place(sem_temp), decimal_place(sem_RH)
                text_to_display = f"T = {round(av_temp, temp_round)} ± {round(sem_temp, temp_round)} °C\n RH = {round(av_RH, RH_round)} ± {round(sem_RH, RH_round)}%"
            else:
                text_to_display = f"T = {round(av_temp, 2)} °C\n RH = {round(av_RH, 2)}%"
                
            plt.text(0.95, 0.9, text_to_display, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        
        if plot_option == "Both":
            gs = GridSpec(1, 2, width_ratios=[3, 1])
            plt.subplot(gs[0, 0])
        
        if plot_option != "Box":
            plt.plot([0, time], [0, 0], '--', linewidth=1.0, color='black')
        
    box_data = []
    
    if input_type == "dir" or input_type == "lis":
        check_ax2 = None
        
        for d in range(len(data)):
            if plot_option != "Box" and plot_average != True and show:
                if simultaneous:
                    if d == 0:
                        check_ax1 = file_names[d].split('_')[1]
                        ax1.set_ylabel(get_axis_title(check_ax1, adjust_by_mass))
                        ax1.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d])
                    elif file_names[d].split('_')[1] == check_ax1:
                        ax1.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d])
                    elif check_ax2 == None:
                        if calibration_args[0] == True:
                            for s, step in enumerate(step_heights):
                                ax1.plot([step_times[s], step_times[s]], [step_middles[s] - step / 2, step_middles[s] + step / 2], '-', color='black')
                        ax2 = ax1.twinx()
                        check_ax2 = file_names[d].split('_')[1]
                        ax2.set_ylabel(get_axis_title(check_ax2, adjust_by_mass))
                        ax2.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d])
                    elif file_names[d].split('_')[1] == check_ax2:
                        ax2.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d])
                    else:
                        print(f"WARNING! {file_names[d]} not plotted as it doesn't match either axis of {check_ax1} or {check_ax2}")
                        
                else:
                    # plt.plot(times, data[d], '-', linewidth=1.0, label=file_names[d].split(".")[0])
                    plt.plot(times, data[d], '-', linewidth=1.0, label=file_names[d])
                
            lol_labels = []
            box_data.append(get_dif(data[d], change_type))
       
        if calibration_args[0] == True and show:
            for row_num in range(len(peak_indecies)):
                    if all(len(peak_heights[i]) == len(peak_heights[0]) for i in range(len(peak_heights))):
                        peak_indecies = np.array(peak_indecies)
                        ax2.plot(peak_indecies[row_num] * time_step, [data[row_num + 1][i] for i in peak_indecies[row_num]], 'x', color='black')
                    else:
                        raise ValueError(f"The number of peaks picked out for each ring do not match: {[len(peak_heights[i]) for i in range(len(peak_heights))]}")
                        
        if len(box_data) >= 2:
            mean = stats.mean(box_data)
            stdev = stats.stdev(box_data)
            sem = stdev / m.sqrt(len(box_data))
            
            if plot_parameter == "Q" and not simultaneous:
                print(f"Average charge drop: {round(mean, 1)} pC, with a standard error of {round(sem, 3)} pC")
        
        if plot_average:
            averaged_data, av_times, average_minus_error, average_plus_error = average_data(data, times, error_lines=error_lines, av_option=av_option)
            if show:
                plt.plot(av_times, averaged_data, '-', linewidth=1.0, label="Average Trace", color=color_list[0])
                if error_lines == 'A':
                    plt.fill_between(av_times, average_minus_error, average_plus_error, color=color_list[0], alpha=0.3)
                elif error_lines == 'L':
                    plt.plot(av_times, average_minus_error, '--', dashes=loosely_dashed, linewidth=0.5, label="Uncertainty", color=color_list[0])
                    plt.plot(av_times, average_plus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[0])
                elif error_lines != 'N':
                    raise ValueError(f'error_lines should be area (A), line (L), or none (N) but instead: {error_lines}')
                
        
    elif input_type == "lol" or input_type == "cat":
        count, counter = 0, 0
        
        for counter2, d in enumerate(data):
            dif = get_dif(d, change_type)
            sep_runs_opacity = 1 - 0.5 * ((float(file_names[counter2].split('.')[0].split('_')[-1]) - 1) / lol_structure[counter])
            
            if count == 0:
                if plot_option != "Box": 
                    if show:
                        plt.plot([0, time], [0, 0], '--', linewidth=1.0, color='black')
                        
                    if plot_average:
                        start_index = sum(lol_structure[:counter])
                        end_index = start_index + lol_structure[counter]
                        averaged_data, av_times, average_minus_error, average_plus_error = average_data(data, times, start_index, end_index, error_lines=error_lines, av_option=av_option)
                        if show:
                            plt.plot(av_times, averaged_data, '-', linewidth=1.0, label=lol_labels[counter], color=color_list[counter])
                            # print(f'{lol_labels[counter]} -> first: {averaged_data[0]}, last: {averaged_data[-1]}') # Printing the beginning and end of each averaged trace
                            if error_lines == 'A':
                                plt.fill_between(av_times, average_minus_error, average_plus_error, color=color_list[counter], alpha=0.3)
                            elif error_lines == 'L':
                                plt.plot(av_times, average_minus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[counter])
                                plt.plot(av_times, average_plus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[counter])
                            elif error_lines != 'N':
                                raise ValueError(f'error_lines should be area (A), line (L), or none (N) but instead: {error_lines}')
                    elif show:
                        if plot_average or not label_sep_runs:
                            plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter], label=lol_labels[counter])
                        else:
                            plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter], label=file_names[counter2], alpha=sep_runs_opacity)
                            
                box_data.append([dif])
                
            else:
                if plot_option != "Box" and not plot_average and show:
                    if label_sep_runs:
                        plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter], label=file_names[counter2], alpha=sep_runs_opacity)
                    else:
                        plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter])
                box_data[-1].append(dif)
            count += 1
            if count >= lol_structure[counter]:
                count = 0
                counter += 1
    
    if show:
                    
        if plot_option != "Box": 
            plt.xlabel("Time / s")  
            
            if simultaneous:
                ax1.legend(frameon=True, loc="upper right")
                ax2.legend(frameon=True, loc="lower right")
                pass
            else:
                plt.ylabel(get_axis_title(plot_parameter, adjust_by_mass))
                plt.legend(frameon=True) # Option: loc="lower left"
                
            ax = plt.gca()
            ax.set_xlim([0, 20])
            left_subplot_ylim = plt.gca().get_ylim() # Extracting the y axis limits
        
        if plot_option == "Both":
            plt.subplot(gs[0, 1])
        
        if not simultaneous:
            plt.ylabel(get_axis_title(plot_parameter, adjust_by_mass))

        if plot_option != "Trace":
            # colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D93536'] # Chosing s for median line
            box = plt.boxplot(box_data, showfliers=True)
            # for i, color in enumerate(colors): # Requeried to set the median lines to specific colours
            #     median_line = box['medians'][i]   
            #     median_line.set_color(color)
            # lol_labels = ["Gr$\mathrm{í}$msv$\mathrm{ö}$tn", "Atitl$\mathrm{á}$n", "Fuego", "St. Helen's"] # For specific labels
            plt.xticks(range(1, len(lol_labels) + 1), lol_labels, rotation=90)
            # plt.ylim(left_subplot_ylim) # Sets y axis same af other subplot
        
        plt.subplots_adjust(wspace=0.7)
        save_name = os.path.join(target_directory, target_directory.split("\\")[-1] + "_" + plot_parameter + "_" + plot_option + ".png")
        plt.savefig(fname=save_name)
        plt.show()
        
    if get_av_data:
        if plot_average:
            if get_error:
                return av_times, averaged_data, average_minus_error, average_plus_error
            return av_times, averaged_data
        else:
            print(f"No Averaged data as plot_average is: {plot_average}")
    
    if plot_average:
        time_out = av_times
    else:
        time_out = times
    if get_error:
        return data, time_out, average_minus_error, average_plus_error
    
    return data, time_out



if __name__ == "__main__":
    
    path = "..\\Drops\\Volcano_Final"
    input_type = "cat" # Directory (dir), categorised (cat), list (lis), or list of lists (lol)
    # Note: the categroised (cat) input type requires filenames formatted such that the type is follwed by underscore then index e.g. Atitlan_1.txt 
    plot_parameter = "Q" # Charge (Q), voltage (V), or raw volatge (RV)
    plot_option = "Trace" # Box plot (Box), The trace (Trace), or both (Both)
    change_type = "mm" # How the boxplot is measured: by the change from the start to finish (sf), by the diference between min and max (mm), or by mm - abs(sf) => (mmsf)
    ignore_len_errors = "Extend" # Should be "Crop" or "Extend" if you want to shorten data to shortest series or extend to the longest "Error" returns error. Note: if "Crop" selected this will affect teh data plotted in the boxplot too
    file_names_lis = ['Lab2Small_1.txt', 'Lab2Small_2.txt', 'Lab2Small_3.txt', 'Lab2Small_4.txt']
    file_names_lol = [['Calibration_1.txt']]
    remove_files = ['Airport_3.txt', '63-75_1.txt', '63-75_4.txt', '63-125_4.txt', '63-125_5.txt', '63-80_2.txt', 'Atitlan_3.txt', 'StHelens_1.txt', 'StHelens_3.txt', 'Eiya_1.txt']
    lol_labels = ["Control", "Zerostat", "Rested"]
    spec_cat = ['Atitlan', 'Grimsvotn'] # specify categories to include if input type is categorised (file names before the underscore), leave empty to include all
    # '63-125', '63-75', '75-125', '80-125', '63-80'
    
    trim = True # If True removes the begining of the trace such that all traces start at the same time (required to calulate average trace)
    plot_average = True # If True plots the average trace instead of all individual traces
    av_option = 'mean' # How the average is calculated, should be 'mean', 'median', or 'bootstrap,
    label_sep_runs = True # If input_type = "cat" and plot_average = False, this will make the runs different hues and labelled, not availible in simulatneous
    error_lines = 'A' # Shows the error when averaged, should be area (A), line (L), or none (N)
    manual_trim = {'Lab2Small_1.txt':4.1, 'Lab2Small_2.txt':3.25, 'Lab2Small_3.txt':2, 'Lab2Small_4.txt':3.7} # Dictionary of file names and the time (in seconds) that you want to maually trim from the start
    store_dict, read_dict = False, True # Options to store or read from file the manual trim dictionary
    Show_T_RH = False # Weather or not to show Temperature and humidity on the plot (if multiple traces plotted than it is an average)
    adjust_by_mass = True
    adjust_for_decay, time_const = False, 310.5 # in [s] and comes from the conductivity of air
    base_time, tolerance_amp, tolerance_len = 0.5, 1.5, 300 # Number of seconds to use as the baseline, How many times greater the trace needs to be than baseline noise to register start over how many steps
    
    data_col = 0 # Selects which column of data file to read (default is the first column = 0)
    simultaneous_plot = False # True plots both ring and cup together
    calibration = False # Picks the peaks for clibration, requires that: parameter_lis = ["Q", "RV", "RV"] and that is is of one dataset (ie input_type = "lis" and plot_average = False) 
    parameter_lis = ["Q", "RV", "RV"] # Charge (Q), voltage (V), or raw volatge (RV), require simultaneous_plot = True
    peak_std_dev_threshold = 2
    step_background_time, step_threshold, step_width = 1, 1.6, 0.2 # The time (/ s) to take a standard deviation of at the start of the trace, the the threshold number of standard deviations to count as a step, time width of a step (/ s)
    dpi = 200
    
    
    ###########################################################################
    # Code to Make Single Plot
    
    if not simultaneous_plot:
        file_names, lol_structure, lol_labels = get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
        data, sample_rates, temperatures, humidities, filtered_names = read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim, manual_trim, store_dict, read_dict, data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const)
        time_step, times, time = get_times(sample_rates, data)
        plot_figure(data, times, time, input_type, plot_parameter, filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, plot_average=plot_average, Show_T_RH=Show_T_RH, error_lines=error_lines, adjust_by_mass=adjust_by_mass, av_option=av_option, label_sep_runs=label_sep_runs, dpi=dpi)
        
    
    ###########################################################################
    # Code to plot cup and ring data together an calibrate
    
    if simultaneous_plot:
        peak_heights, peak_indecies, collated_data, collated_times = [], [], [], []
        file_names, lol_structure, lol_labels = get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
    
        for count, parameter in enumerate(parameter_lis):
            data_col = count
    
            data, sample_rates, temperatures, humidities, filtered_names = read_and_convert_data(file_names, path, parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim, manual_trim, store_dict, read_dict, data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const)
            time_step, times, time = get_times(sample_rates, data)
            data_out, times_out = plot_figure(data, times, time, input_type, parameter, filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, plot_average=plot_average, Show_T_RH=Show_T_RH, show=False, dpi=dpi)

            for index, d in enumerate(data_out):
                times_out = np.arange(0, len(d) * time_step, time_step)            
                collated_times.append(times_out)
                collated_data.append(d)
                
                if calibration:
                    if count == 0:
                        step_heights, step_times, step_middles = get_steps(d, time_step, step_background_time, step_threshold, step_width)
                    else:
                        peak_height_row, peak_index_row = get_peaks(d, peak_std_dev_threshold)
                        peak_heights.append(peak_height_row)
                        peak_indecies.append(peak_index_row)
        
        if calibration:
            calibration_args = calibration, step_times, step_middles, peak_indecies, peak_heights, time_step
        else:
            calibration_args = [False]
        
        # Possibly need to change this if you don't want to plot average but do want to plot multiple
        collated_data, collated_times, filtered_names = average_multiple(collated_data, collated_times, filtered_names, parameter_lis)
        
        
        plot_figure(collated_data, collated_times, time, "lis", "Q", filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, plot_average=False, Show_T_RH=Show_T_RH, show=True, simultaneous=True, calibration_args=calibration_args, av_option=av_option, dpi=dpi)
        
        if calibration:
            calibration_constants(step_heights, peak_heights)
            