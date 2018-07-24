import tables as tb
import numpy as np
from os import walk
import os
from pybar.scans.analyze_ext_trigger_SHiP import analyze_ext_trigger_ship
from optparse import OptionParser
from inspect import getmembers, isclass, getargspec


def get_files(dir, analyze_raw_file = True):
    raw_data_file_list = []
        
    for dirpath,_,filenames in os.walk(subpartition_dir):
        for f in filenames:
            raw_data_file_list.append(os.path.abspath(os.path.join(dirpath, f)))
#     print raw_data_file_list
    for scan_file in raw_data_file_list:
#         print scan_file
        if scan_file[-9:] == 's_hi_p.h5':
            h5_to_tsv(raw_data_file= scan_file, format='pybar', raw_file= analyze_raw_file)

    print "------------converted all files to tab separated value .txt------------"

def h5_to_tsv(raw_data_file = None,format=None ,raw_file=True):
    
    ''' takes raw data input file, interprets it and additionally dumps col, row, module to tab separated value file.'''
    if format=='pybar':
#         try:
        if raw_file==True:
            hit_file = analyze_ext_trigger_ship(raw_data_file, return_file = True )[:-4] + '_interpreted.h5'
#         except IncompleteInputError:
#             pass
    #     moduleID = hit_file.split('module')[1].split('/')[0][1]
    #     print moduleID
        else: 
            hit_file = raw_data_file
            
        with tb.open_file(hit_file, 'r') as in_file:
            hits = in_file.root.Hits[:]
            hits_out = np.zeros(shape = hits.shape[0], dtype = [('column', '<u1'),('row', '<u2')]) #,('moduleID', '<u1')])
            hits_out['row'] = hits['row']
            hits_out['column'] = hits['column']
    #         hits_out['moduleID'] = moduleID
            print hit_file[:-3] + '.txt'
            np.savetxt(hit_file[:-3] + '.txt', hits_out, fmt="%i    %i")
    elif format=='ship':
        with tb.open_file(raw_data_file, 'r') as in_file:
            hits = in_file.root.Spill_2016_11_08_13_44_02.Hits[:] #TODO: find way to walk arbitrary groups
            hits_out = np.zeros(shape = hits.shape[0], dtype = [('channelID', '<u2'),('hit_data', '<u2')])
            hits_out['channelID'] = hits['channelID']
            hits_out['hit_data'] = hits['hit_data']
            
            header = in_file.root.Spill_2016_11_08_13_44_02.Headers[:] # TODO: find way to walk arbitrary groups
            print header.dtype
            header_out = np.zeros(shape = header.shape[0], dtype = [("size",   np.uint16),
                                                                  ("partID", np.uint16),
                                                                  ("cycleID",np.uint64),
                                                                  ("frameTime", np.uint64),
                                                                  ("timeExtent", np.uint16),
                                                                  ("flags",  np.uint16)])
            header_out['size'] = header['size']
            header_out['partID'] = header['partID']
            header_out['cycleID'] = header['cycleID']
            header_out['frameTime'] = header['frameTime']
            header_out['timeExtent'] = header['timeExtent']
            header_out['flags'] = header['flags']
            
            np.savetxt(raw_data_file[:-3] + 'hits_eventnr.txt', hits, fmt="%i    %i    %i")
            np.savetxt(raw_data_file[:-3] + 'headers_eventnr.txt', header, fmt="%i    %i    %i    %i    %i    %i    %i")
            # write binary files
            hits_out.tofile(raw_data_file[:-3] + 'hits.txt')
            header_out.tofile(raw_data_file[:-3] + 'headers.txt')
            
            
    else:
        print "please specify format: pybar, or ship?"
        
if __name__ == "__main__":
    
    usage = "Usage: %prog directory analyze_raw_file (Boolean)"
    description = "dir = subpartiotion directory with module directories containing raw_data files analyze_raw_file= whether or not raw data files should be analyzed, if not, the directory has to contain analyzed files"
    parser = OptionParser(usage, description=description)
    options, args = parser.parse_args()
    if len(args) == 2 :
        subpartition_dir = args[0]
        analyze_raw_file = bool(args[1])
    else:
        parser.error("incorrect number of arguments")
    
    get_files(subpartition_dir,analyze_raw_file)
            
    
    
    
