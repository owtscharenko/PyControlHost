import sys
import time
from threading import Event, Lock
from optparse import OptionParser

import zmq
import numpy as np
from numba import njit

from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from pybar_fei4_interpreter.data_histograming import PyDataHistograming

# from ch_transmission import control_host_coms as ch


class DataConverter():

    def __init__(self, socket_addr):
        self.integrate_readouts = 1
        self.n_readout = 0
        self._stop_readout = Event()
        self.setup_raw_data_analysis()
        self.reset_lock = Lock()
        self.connect(socket_addr)
        self.run()

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
        self.create_empty_event_hits = True
        self.trigger_data_format = 1
        self.max_trigger_number = 2 ** 31 - 1  # 31 bit?
        self.interpreter.interpret_raw_data(raw_data)
        self.interpreter.align_at_trigger(True)
        self.interpreter.get_hits()
#         self.process_data(self.interpreter.get_hits(),moduleID)


    def process_data(self, data_array, moduleID):
        '''
        each hit is converted to two 16bit datawords, 1st word is pixel of
        FE, second word is relBCID + number of FE + tot
        
        order of data_array:
                [('event_number', '<i8'),
                ('trigger_number', '<u4'),
                ('trigger_time_stamp', '<u4'), 
                ('relative_BCID', 'u1'),
                ('LVL1ID', '<u2'),
                ('column', 'u1'),
                ('row', '<u2'), 
                ('tot', 'u1'),
                ('BCID', '<u2'),
                ('TDC', '<u2'), 
                ('TDC_time_stamp', 'u1'), 
                ('trigger_status', 'u1'),
                ('service_record', '<u4'),
                ('event_status', '<u2')]
        '''

        ch_hit_data = []
#         events = build_events_from_raw_data(data_array)
        for i in range(len(data_array)):

            hit_row = data_array['row'][i]
            column = data_array['column'][i]
            channelID = "{:016b}".format(hit_row * column)
            hitTime = "{:04b}".format(data_array['relative_BCID'][i])
            tot = "{:04b}".format(data_array['tot'][i])
            feID = "{:02b}".format(moduleID)

            ch_additional_dataword = hitTime + feID + tot
            ch_hit_data.extend((channelID, ch_additional_dataword))
        hit_data = np.ascontiguousarray(ch_hit_data, dtype=str)
        print ch_hit_data[:2]


    def build_header(self, n_hits, partitionID, cycleID, trigger_timestamp, bcids=16, flag=0):
        """
        builds data frame header from input information,
        python variables have to be converted to bitstrings.
        8 bit = 1 byte, 2 hex digits = 1 byte

        input variables:
        n_hits: int , used to calculate length of data frame, 1 hit = 4 byte
        partitionID: int, fixed for each MMC3 board
        cycleID: int, see below, provided by run control
        trigger_timestamp: int, used as frameTime (see below)

        format:
            size (2 byte) = length of data frame in bytes including header
            partitionID (2 byte) = 0000 to 0002
            cycleIdentifier (4 byte) = time of spill in SHiP time format: 0.2 seconds steps since the 8 April 2015
            frameTime (4 byte) = start of trigger window relative to SoC in 25ns steps
            timeExtent (2 byte) = length of the trigger window in 25ns steps
            flags (2 byte) = empty for now

        """
        data_header = "{:16b}".format(n_hits * 4 + 16) + "{:16b}".format(partitionID) + "{:32b}".format(
            cycleID) + "{:32b}".format(trigger_timestamp) + "{:04b}".format(bcids - 1) + "{:04b}".format(flag)

        return data_header

    def build_event_frame(self, data_header, hit_data):
        """
        sends data to controlhost dispatcher
        input:
            address: IP-address of dispatcher
            data_header: bitstring with frame_header info
            hit_data: list of bitstrings, 2 strings (2 byte each) for each hit in event_frame
        """
        event_frame = []
        event_frame.extend((data_header, hit_data))
        event_frame = np.ascontiguousarray(event_frame, str)

        return event_frame

    def run(self):
        while(not self._stop_readout.wait(0.01)):  # use wait(), do not block here
            with self.reset_lock:
                try:
                    meta_data = self.socket_pull.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                else:
                    name = meta_data.pop('name')
                    if name =='Filename':
                        print meta_data.pop('conf')
                         
                    elif name == 'ReadoutData':
                        data = self.socket_pull.recv()
                        # reconstruct numpy array
                        buf = buffer(data)
                        dtype = meta_data.pop('dtype')
                        shape = meta_data.pop('shape')
                        data_array = np.frombuffer(
                            buf, dtype=dtype).reshape(shape)
                        self.analyze_raw_data(data_array)
                        self.process_data(self.interpreter.get_hits(), moduleID=1)

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

    conv = DataConverter(socket_addr=socket_addr)
