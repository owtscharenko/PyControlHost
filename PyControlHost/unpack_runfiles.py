from __future__ import division

import os
import numpy as np
import tables as tb
import datetime
from numba import njit, jit
from ship_data_converter import FrHeader, Hit
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
logger = logging.getLogger('Converter')


def get_file_date(cycleID):
    ''' convert spillID / cycleID to human readable date'''
    
    start_date = datetime.datetime(2015, 04, 8, 13, 00) # careful: python datetime is 1 hour behind c time.h module. TODO: check why
    return (start_date + datetime.timedelta(seconds = cycleID / 5.)).strftime("%Y_%m_%d_%H_%M_%S")


def unpack_run_file(input_file):
    partID = input_file.split('_')[-3]
    run_number = input_file.split('_')[-1].split('.')[0]
    directory = os.path.dirname(input_file)
    spillpath =  "%s/part_%s/RUN_%s/" % (directory, partID, run_number)
    logger.info('Unpacking partition %s run %s' % (partID, run_number))
    if not os.path.exists(spillpath):
                os.makedirs(spillpath)

    with tb.open_file(input_file, mode='r') as in_file:
        for group in in_file.walk_groups():
            spill_number = group._v_name

            if spill_number == '/':
                continue
            
            else:
                spill_number = spill_number.split('_')[-1]
                file_name =  get_file_date(np.uint32(spill_number))
#                 logger.info('spill %s corresponds to date %s' %(spill_number,file_name))
                headers_in = group.Headers[:]
                hits_in = group.Hits[:]
                logger.info('found %s hits in %s frames for spill %s / %s' %(hits_in.shape[0], headers_in.shape[0], spill_number, file_name))
                i = 0
                with open(spillpath + file_name + '.txt', 'w') as spillfile:
                    for row in range(0,headers_in.shape[0]):
                        try:
                            nhits = int((headers_in[row]['size']-16)/4)
#                             print nhits
                            if nhits <= 0 :
                                logger.warning('header %s has size %s'% (row,nhits))
                                if row > 0:
                                    ''' also breaks if empty header at the end of file. this empty header usually is followed by nhits empty headers.
                                        in fact no known occurence of an empty header in between normal frames was found but only empty headers at the end.
                                    '''
                                    logger.error('empty header in between frames') 
                                    raise RuntimeWarning 
                                continue
                            else:
                                header_out = np.zeros(shape=(1,), dtype = FrHeader)
                                hits_out = np.zeros(shape = nhits, dtype = Hit)
                                
                                header_out['size'] = headers_in[row]['size']
                                header_out['partID'] = headers_in[row]['partID']
                                header_out['cycleID'] = headers_in[row]['cycleID']
                                header_out['frameTime'] = headers_in[row]['frameTime']
                                header_out['timeExtent'] = headers_in[row]['timeExtent']
                                header_out['flags'] = headers_in[row]['flags']
                                
                                hits_out['channelID'] = hits_in[i:(i+nhits)]['channelID']
                                hits_out['hit_data'] = hits_in[i:(i+nhits)]['hit_data']
    #                             np.savetxt(spillfile, headers[row].reshape((1,)))
    #                             np.savetxt(spillfile, hits[i:(i+nhits)])
#                                 headers[row].tofile(spillfile)
#                                 hits[i:(i+nhits)].tofile(spillfile)
                                header_out.tofile(spillfile)
                                hits_out.tofile(spillfile)
                                i += nhits
                        except RuntimeWarning:
                            logger.info('ended loop')
                            break
                        
                
def check_bytefiles(input_bytefile):
    with open(input_bytefile, 'r') as in_file:
        data  = in_file.read().encode('hex')
        print data
        
if __name__ == '__main__':
    
    raw_data_file_list = []
    
    for data_file in os.listdir('/media/data/ship_charm_xsec_Jul18/run_data/'):
        if data_file[-3:] == '.h5':
            raw_data_file_list.append(os.path.abspath(os.path.join('/media/data/ship_charm_xsec_Jul18/run_data/', data_file)))
    
    for scan_file in tqdm(raw_data_file_list):
        unpack_run_file(scan_file)
#         check_bytefiles('/media/data/ship_charm_xsec_Jul18/run_data/part_0x800/RUN_2467/2018_07_26_16_44_10.txt')
        
        
        