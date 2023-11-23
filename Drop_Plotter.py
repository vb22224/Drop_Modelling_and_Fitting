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



def get_dif(d, change_type):
    
    if change_type == "mm":
        dif = max(d) - min(d)
    elif change_type == "sf":
        dif = d[-1] - d[0]
    elif change_type == "mmsf":
        dif = max(d) - min(d) - abs(d[-1] - d[0])
    else:
        raise ValueError("Invalid change type,  should be start to finish (sf), min and max (mm), of mm minus sf (mmsf)")
    return dif



def get_times(sampls_rates, data):
    
    """ Gets the time step, array of times and total time required for the trae data """
    
    time_step = 1 / sample_rates[0]
    times = [i * time_step for i in range(len(data[0]))]
    time = len(data[0]) * time_step
    
    return time_step, times, time



def average_data(data, times, start_index=0, end_index=None):
    
    """
    Averages data from the data list of traces in specified range satrt_index to end_index
    Clips this data to the shortest trace in the range and also returns the times required to
    plot this average
    """
    
    if end_index is None:
        end_index = len(data[0])
    
    valid_data = data[start_index:end_index]
    data_lengths, averaged_data = [], []
    
    for d in valid_data:
        data_lengths.append(len([x for x in d if not m.isnan(x)]))
    
    for n in range(min(data_lengths)):
        n_data = []
        for d in valid_data:
            n_data.append(d[n])
        averaged_data.append(stats.mean(n_data))
        
    av_times = times[:min(data_lengths)]
        
    return averaged_data, av_times
    
    

def get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels):
    
    lol_structure = []
    
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
    # all_file_names = ['Grims_1.txt', 'Grims_2.txt', 'Grims_3.txt', 'Grims_4.txt', 'Grims_5.txt',
    # 'Grims_6.txt', 'Grims_7.txt', 'Grims_8.txt', 'Grims_9.txt', 'Grims_10.txt',
    # 'Grims_11.txt', 'Grims_12.txt', 'Grims_13.txt', 'Grims_14.txt', 'Grims_15.txt',
    # 'Atitlan_1.txt', 'Atitlan_2.txt', 'Atitlan_3.txt', 'Atitlan_4.txt',
    # 'Atitlan_5.txt', 'Atitlan_7.txt', 'Atitlan_8.txt', 'Atitlan_9.txt',
    # 'Atitlan_10.txt', 'Atitlan_11.txt', 'Atitlan_12.txt', 'Atitlan_13.txt',
    # 'Atitlan_14.txt', 'Atitlan_15.txt', 'Fuego_1.txt', 'Fuego_2.txt', 'Fuego_3.txt', 'Fuego_4.txt', 'Fuego_5.txt','StHelens_1.txt', 'StHelens_2.txt', 'StHelens_3.txt']
    # print(all_file_names)
    
    for remove_file in remove_files:
        all_file_names = [file for file in all_file_names if remove_file not in file]
        
    for file in all_file_names:
        try:
            read = open(os.path.join(path, file), "r", encoding="utf-8").read()
            if "Keithley" in read:
                file_names.append(file)
        except UnicodeDecodeError as e:
            raise ValueError(f"Error reading {file}: {e}")
    
    if input_type == "cat":
        file_names.sort()
        
        lol_labels, lol_structure = [], []
        for f in file_names:
            if f.split("_")[0] in lol_labels:
                lol_structure[-1] += 1
            else:
                lol_labels.append(f.split("_")[0])
                lol_structure.append(1)
    #print(lol_labels, lol_structure)

    return file_names, lol_structure, lol_labels



def read_and_convert_data(file_names, target_directory, plot_parameter, ignore_len_errors, trim_start=False):
    
    data, sample_rates = [], []

    for file in file_names:
        dat = pd.read_csv(os.path.join(target_directory, file), header=2).values.tolist()
        converted_data = []
        for d in dat:
            if plot_parameter == "V":
                converted_data.append(d[0] * 10)
            elif plot_parameter == "Q":
                converted_data.append(d[0] * 10 * 130)
            else:
                raise ValueError("Invalid plot_parameter, requires charge (Q) or voltage (V)")
        
        metadata = open(os.path.join(target_directory, file), "r").read().split("Keithley")[0]
        data.append(converted_data)
        sample_rate = float(metadata.split(",")[1])
        sample_rates.append(sample_rate)
    
    if len((sample_rates)) == 0:
        raise ValueError(f'No output files found in directory: {target_directory}')
        
    elif len(set(sample_rates)) != 1:
        raise ValueError(f'Sample rates do not match: {sample_rates}')
    
    if trim_start == True: # Trimming off the start of each trace
        
        base_time = 1.0 # Number of seconds to use as the baseline
        tolerance_amp, tolerance_len = 2, 100 # How many times greater the trace needs to be than baseline noise to register start over how many steps
            
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
    
    return data, sample_rates



def plot_figure(data, times, time, input_type, plot_parameter, file_names, target_directory, lol_structure, lol_labels, change_type, plot_option, plot_average=True):
    
    if plot_option != "Box" and plot_option != "Trace" and plot_option != "Both":
        raise ValueError('plot_option should be "Box" for box plot, "Trace" for just the trace, or "Both" for both plots.')
    
    plt.figure(dpi=1200, figsize=(4, 3)) # figsize=(9, 4) optional parameter
    
    if plot_option == "Both":
        gs = GridSpec(1, 2, width_ratios=[3, 1])
        plt.subplot(gs[0, 0])
        plt.plot([0, time], [0, 0], '--', linewidth=1.0, color='black')
    
    box_data = []
    
    if input_type == "dir" or input_type == "lis":
        for d in range(len(data)):
            if plot_option != "Box" and plot_average != True:
                # plt.plot(times, data[d], '-', linewidth=1.0, label=file_names[d].split(".")[0])
                plt.plot(times, data[d], '-', linewidth=1.0, label=file_names[d])
            lol_labels = []
            box_data.append(get_dif(data[d], change_type))
        mean = stats.mean(box_data)
        stdev = stats.stdev(box_data)
        sem = stdev / m.sqrt(len(box_data))
        
        if plot_average == True:
            averaged_data, av_times = average_data(data, times)
            plt.plot(av_times, averaged_data, '-', linewidth=1.0, label="Average Trace")
        
        print(f"Average charge drop: {mean} pC, with a standard error of {sem} pC")
        
    elif input_type == "lol" or input_type == "cat":
        count, counter = 0, 0
        color_list = ["r", "g", "b", "c", "m", "y", "darkgoldenrod", "yellowgreen", "indigo"]
        for d in data:
            dif = get_dif(d, change_type)
            
            if count == 0:
                if plot_option != "Box":
                    if plot_average == True:
                        start_index = sum(lol_structure[:counter])
                        end_index = start_index + lol_structure[counter]
                        averaged_data, av_times = average_data(data, times, start_index, end_index)
                        plt.plot(av_times, averaged_data, '-', linewidth=1.0, label=lol_labels[counter], color=color_list[counter])
                    else:
                        plt.plot(times, d, '-', linewidth=1.0, label=lol_labels[counter], color=color_list[counter])
                box_data.append([dif])
            else:
                if plot_option != "Box" and plot_average != True:
                    plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter])
                box_data[-1].append(dif)
            count += 1
            if count >= lol_structure[counter]:
                count = 0
                counter += 1
                
    if plot_option != "Box": 
        plt.xlabel("Time / s")   
        if plot_parameter == "V":
            plt.ylabel("Voltage / V")
        elif plot_parameter == "Q":
            plt.ylabel("Charge / pC")
        ax = plt.gca()
        ax.set_xlim([0, time])
        left_subplot_ylim = plt.gca().get_ylim() # Extracting the y axis limits
        plt.legend(frameon=True) # Option: loc="lower left"
    
    if plot_option == "Both":
        plt.subplot(gs[0, 1])
    
    if plot_parameter == "V":
        plt.ylabel("Voltage / V")
    elif plot_parameter == "Q":
        plt.ylabel("Charge / pC")
        
    if plot_option != "Trace":
        # colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D93536'] # Chosing colours for median line
        box = plt.boxplot(box_data, showfliers=True)
        # for i, color in enumerate(colors): # Requeried to set the median lines to specific colours
        #     median_line = box['medians'][i]
        #     median_line.set_color(color)
        # lol_labels = ["Gr$\mathrm{í}$msv$\mathrm{ö}$tn", "Atitl$\mathrm{á}$n", "Fuego", "St. Helen's"] # For specific labels
        plt.xticks(range(1, len(lol_labels) + 1), lol_labels)
        # plt.ylim(left_subplot_ylim) # Sets y axis same af other subplot
    
    plt.subplots_adjust(wspace=0.7)
    save_name = os.path.join(target_directory, target_directory.split("\\")[-1] + "_" + plot_parameter + "_Box_Plot.png")
    plt.savefig(fname=save_name)

    plt.show()



if __name__ == "__main__":
    
    path = "..\\Drops\\Grims_Fit"
    input_type = "cat" # Directory (dir), categorised (cat), list (lis), or list of lists (lol)
    # Note: the categroised (cat) input type requires filenames formatted such that the type is follwed by underscore then index e.g. Atitlan_1.txt
    plot_parameter = "Q" # Charge (Q) or voltage (V)
    change_type = "mm" # How the boxplot is measured: by the change from the start to finish (sf), by the diference between min and max (mm), or by mm - abs(sf) (mmsf)
    plot_option = "Trace" # Box plot (Box), The trace (Trace), or both (Both)
    ignore_len_errors = "Extend" # Should be "Crop" or "Extend" if you want to shorten data to shortest series or extend to the longest "Error" returns error. Note: if "Crop" selected this will affect teh data plotted in the boxplot too
    trim_start = True # Boolean input, if True removes the begining of the trace such that all traces start at the same time (required to calulate average trace)
    plot_average = True # Boolean input, if True plots the average trace instead of all individual traces
    file_names_lis = ["StHelens_1.txt", "StHelens_2.txt", "StHelens_3.txt"]
    file_names_lol = [["Atitlan_1.txt", "Atitlan_2.txt", "Atitlan_3.txt", "Atitlan_4.txt", "Atitlan_5.txt"],
                      ["Atitlan_6.txt", "Atitlan_7.txt", "Atitlan_8.txt", "Atitlan_9.txt", "Atitlan_10.txt"]]
    remove_files, file_names = ["Atitlan_2.txt", "Atitlan_4.txt", "Atitlan_6.txt", "Atitlan_11.txt", "Atitlan_15.txt", "StHelens_1.txt", "Fuego_3.txt"], []
    lol_labels = ["Min", "Max"]
    
    file_names, lol_structure, lol_labels = get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels)
    data, sample_rates = read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, trim_start)
    time_step, times, time = get_times(sample_rates, data)
    plot_figure(data, times, time, input_type, plot_parameter, file_names, path, lol_structure, lol_labels, change_type, plot_option, plot_average)
    
    
    
    