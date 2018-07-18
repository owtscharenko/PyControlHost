''' This is a producer faking data coming from bdaq53

    Real data is send in chunks with correct timing.
    This producer is needed for debugging and testing.
'''

import time
from subprocess import Popen

import numpy as np
import tables as tb
import logging

import zmq
import progressbar

from pybar.daq.fei4_raw_data import send_data


def transfer_file(file_name, socket_addr):  # Function to open the raw data file and sending the readouts periodically
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(socket_addr) # change to socket.bind in case ob PUB /SUB
#     recv_socket = context.socket(zmq.PULL)
#     recv_socket.connect('tcp://127.0.0.1:5011')
    logging.info("data sent to %s" % socket_addr)
    with tb.open_file(file_name, mode="r") as in_file_h5:
        start = time.time()
        meta_data = in_file_h5.root.meta_data[:]
        raw_data = in_file_h5.root.raw_data[:]
        n_readouts = meta_data.shape[0]

        last_readout_time = time.time()
        try:
            scan_parameter_names = in_file_h5.root.configuration.conf[:]['name'] #in_file_h5.root.scan_parameters.dtype.names
        except tb.NoSuchNodeError:
            scan_parameter_names = None
        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=meta_data.shape[0], term_width=80)
        progress_bar.start()
        
        for i in range(n_readouts):

            # Raw data indeces of readout
            i_start = meta_data['index_start'][i]
            i_stop = meta_data['index_stop'][i]

            # Time stamps of readout
            t_stop = meta_data[i]['timestamp_stop']
            t_start = meta_data[i]['timestamp_start']

            # Create data of readout (raw data + meta data)
            data = []
            data.append(raw_data[i_start:i_stop])
            data.extend((float(t_start),
                         float(t_stop),
                         int(meta_data[i]['error'])))
#             scan_par_id = int(meta_data[i]['scan_param_id'])

            send_data(socket, data)
                
            if i == 0:  # Initialize on first readout
                last_timestamp_start = t_start
            now = time.time()
#             if now - start > 180:
#                 break
            delay = now - last_readout_time
            additional_delay = t_start - last_timestamp_start - delay
            if additional_delay > 0:
                # Wait if send too fast, especially needed when readout was
                # stopped during data taking (e.g. for mask shifting)
                time.sleep(additional_delay)
            last_readout_time = time.time()
            last_timestamp_start = t_start
            time.sleep(meta_data[i]['timestamp_stop'] - meta_data[i]['timestamp_start'])
            progress_bar.update(i)
        progress_bar.finish()
    socket.close()
    context.term()


if __name__ == '__main__':
    # Open th online monitor
    socket_addr = "tcp://127.0.0.1:5001"
#     Popen(["python", "../../pybar/online_monitor.py", socket_addr])  # if this call fails, comment it out and start the script manually
    # Prepare socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(socket_addr)
    time.sleep(3)
    # Transfer file to socket
    transfer_file("/media/data/SHiP/charm_exp_2018/test_data_converter/elsa_testbeam_data/take_data/module_7/117_module_7_ext_trigger_scan_s_hi_p.h5", socket=socket)
    # Clean up
    
    socket.close()
    context.term()