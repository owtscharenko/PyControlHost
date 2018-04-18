import sys
import time
from threading import Event, Lock
from optparse import OptionParser

import zmq
import numpy as np

from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from pybar_fei4_interpreter.data_histograming import PyDataHistograming

# from ch_transmission import control_host_coms as ch
# from ch_transmission import ship_data_format


class DataConverter():

    def __init__(self):
        self.integrate_readouts = 1
        self.n_readout = 0
        self._stop_readout = Event()
        self.setup_raw_data_analysis()
        self.reset_lock = Lock()

    def setup_raw_data_analysis(self):
        self.interpreter = PyDataInterpreter()
        self.histogram = PyDataHistograming()
        self.interpreter.set_warning_output(False)
        self.histogram.set_no_scan_parameter()
        self.histogram.create_occupancy_hist(True)
        self.histogram.create_tot_hist(True)
        self.interpreter.set_FEI4B(True)
        try:
            self.histogram.create_tdc_distance_hist(True)
            self.interpreter.use_tdc_trigger_time_stamp(True)
        except AttributeError:
            self.has_tdc_distance = False
        else:
            self.has_tdc_distance = True

    def connect(self, socket_addr):
        self.socket_addr = socket_addr
        self.context = zmq.Context()
        self.socket_pull = self.context.socket(zmq.SUB)  # subscriber
        self.socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data
        self.socket_pull.connect(self.socket_addr)

    def on_set_integrate_readouts(self, value):
        self.integrate_readouts = value

    def reset(self):
        with self.reset_lock:
            self.histogram.reset()
            self.interpreter.reset()
            self.n_readout = 0

    def analyze_raw_data(self, raw_data):
        self.interpreter.interpret_raw_data(raw_data)
        hits_array = self.interpreter.get_hits()

    def process_data(self, hits_array):  # process each hit to one string, representing bits
        ch_hit_data = []
        for hit in hits_array:
            row = hits_array['row'][hit]
            column = hits_array['column'][hit]
            channelID = "{0:b}".format(row * column)
            
            hitTime = "{0:b}".format(hits_array['relative_BCID'][hit])
            tot = "{0:b}".format(hits_array['tot'][hit])
            feID = "{0:b}".format(7)
            
            ch_additional_dataword = hitTime + feID + tot
            ch_hit_data.append(channelID, ch_additional_dataword)
        print ch_hit_data
        
        self.finished.emit()

    def send_data(self,address, data):
        ch.init_disp(address)
        ch.send_fulldata()

    def stop(self):
        self._stop_readout.set()

if __name__ == '__main__':
    usage = "Usage: %prog ADDRESS"
    description = "ADDRESS: Remote address of the sender (default: tcp://127.0.0.1:5678)."
    parser = OptionParser(usage, description=description)
    options, args = parser.parse_args()
    if len(args) == 0:
        socket_addr = 'tcp://127.0.0.1:5678'
    elif len(args) == 1:
        socket_addr = args[0]
    else:
        parser.error("incorrect number of arguments")

