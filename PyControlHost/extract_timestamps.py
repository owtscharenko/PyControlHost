import tables as tb
import numpy as np
from os import walk
import os
# from pybar.scans.analyze_ext_trigger_SHiP import analyze_ext_trigger_ship
from optparse import OptionParser
from inspect import getmembers, isclass, getargspec


def get_files(dir, run_number, output_folder):
    ''' scans given directory for .h5 files with given run_number in title and feeds this to "extract_timestamp" '''
    raw_data_file_list = []
        
    for dirpath,_,filenames in os.walk(dir):
        for f in filenames:
            if '.h5' in f and str(run_number) in f:
                raw_data_file_list.append(os.path.abspath(os.path.join(dirpath, f)))
    for scan_file in raw_data_file_list:
#         print scan_file
        if str(run_number) in scan_file:
            extract_timestamp(data_file= scan_file, output_folder = output_folder)


def extract_timestamp(data_file = None, output_folder = None):
    
    ''' takes ship format .h5 file and dumps spill number + trigger timestamps to .txt file. The run_number is in the file name.
    -------------------
    input: data_file : input ship format .h5 file
    output_folder: directory where to write .txt file
    
'''
    with tb.open_file(data_file, 'r') as in_file:
        for table in in_file.walk_nodes('/', classname= 'Table'):
            if table.name == 'Headers':
                header = table[:]
                
                print header.dtype
                header_out = np.zeros(shape = header.shape[0], dtype = [("cycleID",np.uint64),
                                                                      ("frameTime", np.uint64)])
    
                header_out['cycleID'] = header['cycleID']
                header_out['frameTime'] = header['frameTime']
                print header_out.dtype
    #             np.savetxt(data_file[:-3] + 'hits_eventnr.txt', hits, fmt="%i    %i    %i")
                if output_folder :
                    np.savetxt(output_folder + os.path.split(data_file)[1][:-3] + 'headers.txt', header_out, fmt="%i    %i", header= "cycleID\tframeTime")
                else:    
                    np.savetxt(data_file[:-3] + 'headers.txt', header_out, fmt="%i    %i", header= "cycleID\tframeTime")
            
        
if __name__ == "__main__":
    
    scifi_runs = [2781,2782,2783,2784,2785,2786,2788,2789,2790,2787,2793,
              2794,2795,2796, 2797,2798,2799,2800,2801,2805,2806,2807,
              2810,2811,2812,2814]
    dir = '/media/data/SHiP/charm_exp_2018/data/run_data'
    output_folder = '/media/data/SHiP/charm_exp_2018/data/extract_ts/'
    
    for run in scifi_runs:
        get_files(dir = dir, run_number = run, output_folder = output_folder)
            
    
    
    
