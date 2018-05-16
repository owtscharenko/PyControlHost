import sys
import time
from threading import Event, Lock
from optparse import OptionParser
import tables as tb
import progressbar
from bitarray import bitarray
import array
import zmq
import numpy as np
import datetime
from numba import njit,jit
import timeit
import cProfile, pstats
import pprint


from build_binrep import hit_to_binary
from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from pybar_fei4_interpreter.data_histograming import PyDataHistograming
from pybar.daq.readout_utils import build_events_from_raw_data, is_trigger_word, get_col_row_tot_array_from_data_record_array
from PyControlHost.run_control import run_control


from subprocess import Popen
from pybar.daq.fei4_raw_data import send_data


def transfer_file(file_name, socket):  # Function to open the raw data file and sending the readouts periodically
    with tb.open_file(file_name, mode="r") as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        raw_data = in_file_h5.root.raw_data[:]
        try:
            scan_parameter_names = in_file_h5.root.scan_parameters.dtype.names
        except tb.NoSuchNodeError:
            scan_parameter_names = None
        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=meta_data.shape[0], term_width=80)
        progress_bar.start()
        for index, (index_start, index_stop) in enumerate(np.column_stack((meta_data['index_start'], meta_data['index_stop']))):
            data = []
            data.append(raw_data[index_start:index_stop])
            data.extend((float(meta_data[index]['timestamp_start']), float(meta_data[index]['timestamp_stop']), int(meta_data[index]['error'])))
            if scan_parameter_names is not None:
                scan_parameter_value = [int(value) for value in in_file_h5.root.scan_parameters[index]]
                send_data(socket, data, scan_parameters=dict(zip(scan_parameter_names, scan_parameter_value)))
            else:
                send_data(socket, data)
            time.sleep(meta_data[index]['timestamp_stop'] - meta_data[index]['timestamp_start'])
            progress_bar.update(index)
        progress_bar.finish()
        
@njit             
def process_data_numba(data_array, moduleID, flag=0):
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

    for i in range(data_array.shape[0]):
        row = data_array['row'][i]
        column = data_array['column'][i]
          
        '''
        channelID = np.uint16(80*column + row) counts from left to right and from bottom to top
        eg: channelID =  1 : row = 0, column =1,
            channelID =  2 : row = 0, column = 2
            channelID = 81 : row = 1, column = 1
            channelID = 162: row = 2, column = 2
        use function decode_channelID to decode
        '''

        channelID = np.uint16(column<<9 ^ row)
        '''
        channelID: 16 bit word
                    first 7 bit: column
                    following 9 bit: row
        ch_2nd_dataword: 16 bit word. From MSB(left) to LSB(right)
                    highest 4 bit: BCID
                    second 4 bit: 0000 (in-BCID time res. can not be provided)
                    third 4 bit: moduleID (0-7 => 0000 to 0111)
                    lowest 4 bit: ToT
        '''

        ch_2nd_dataword = np.uint16(data_array['tot'][i]<<12 ^ moduleID<<8 ^ flag<<4 ^ np.uint8(data_array['relative_BCID'][i]))  
        ch_hit_data.extend((channelID, ch_2nd_dataword))
    return ch_hit_data

@njit
def decode_channelID(channelID):
    '''
#     converts channelID (0 to 26879, each value corresponds to one pixel of one FE) to column, row
    converts channelID to column and pixel. 16 bit uint: highest 7bit = column, lowest 9 bit = row
    input:
        uint16
    returns tuple column(np.uint8), row(np.uint16)
    '''

#     row = np.uint16(0)
#     column = np.uint16(0)
#     if channelID == 0:
#         row = 0
#         column = 0
#         return column, row
#     column = channelID % 80
    column = np.uint8(channelID>>9)
    row=np.uint16(channelID ^ (column<<9))
#     row = (channelID - 79) / 80
    return column, row

@njit
def decode_second_dataword(dataword):
    '''
    converts second dataword (16bit: 4 bit tot, 4bit moduleID, 4bit flags=0000, 4bit BCID) to single values
    input:
        int16
    returns:
        tuple (tot(np.uint8), moduleID(np.uint8), flags(np.uint8), rel_BCID(np.uint8)) each value is only 4bit but 8bit is smallest dtype
    '''

    tot = np.uint8(dataword>>12)
    moduleID = np.uint8(dataword ^ (tot<<12)>>8)
    flag = np.uint8(dataword ^ (tot<<12 ^ moduleID<<8)>>4)
    rel_BCID = np.uint8(dataword ^ (tot<<12 ^ moduleID<<8 ^ flag<<4))
    
    return tot, moduleID, flag, rel_BCID

            
class DataConverter(object):

    def __init__(self, socket_addr):
        self.connect(socket_addr)
#         self.integrate_readouts = 1
        self.n_readout = 0
        self._stop_readout = Event()
        self.setup_raw_data_analysis()
        self.reset_lock = Lock()
        self.kill_received = False
        self.run()
        

    def cycle_ID(self):
        # counts in 0.2s steps from 08. April 2015
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return np.uint32((datetime.datetime.now() - start_date).total_seconds() * 5)


    def setup_raw_data_analysis(self):
        self.interpreter = PyDataInterpreter()
        self.interpreter.create_empty_event_hits(True)
        self.interpreter.set_trigger_data_format(1)
#         self.interpreter.set_max_trigger_number(2 ** 31 - 1)  # 31 bit?
        self.interpreter.align_at_trigger(True)
        self.interpreter.set_warning_output(False)
        self.interpreter.set_FEI4B(True)


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
        


#     def process_data(self,data_array, moduleID):
#         '''
#         each hit is converted to two 16bit datawords, 1st word is pixel of
#         FE, second word is relBCID + number of FE + tot
#         order of data_array:
#                 [('event_number', '<i8'),
#                 ('trigger_number', '<u4'),
#                 ('trigger_time_stamp', '<u4'), 
#                 ('relative_BCID', 'u1'),
#                 ('LVL1ID', '<u2'),
#                 ('column', 'u1'),
#                 ('row', '<u2'), 
#                 ('tot', 'u1'),
#                 ('BCID', '<u2'),
#                 ('TDC', '<u2'), 
#                 ('TDC_time_stamp', 'u1'), 
#                 ('trigger_status', 'u1'),
#                 ('service_record', '<u4'),
#                 ('event_status', '<u2')]
#         '''
#         ch_hit_data = []
#   
# #         if data_array.shape[0] != 0:
# #             bitwords = data_array[['row','column']].copy()
# #             print bitwords.dtype
# # #             bitwords = np.array(map(bin,bitwords.flatten())).reshape(bitwords.shape)
# #              
# #             bitwords['row'] = (bitwords['row']).tobytes()
# #             print bitwords['row']
# #             bitwords['column'] = np.binary_repr(bitwords['column'][:],widht=7)
# #             channelID.extend([bitwords['row']<<7 ^ bitwords['column']])
#  
#         for i in range(data_array.shape[0]):
#             row = data_array['row'][i]
#             column = data_array['column'][i]
#               
#             '''
#             channelID = np.uint16(80*column + row) counts from left to right and from bottom to top
#             eg: channelID =  1 : row = 0, column =1,
#                 channelID =  2 : row = 0, column = 2
#                 channelID = 81 : row = 1, column = 1
#                 channelID = 162: row = 2, column = 2
#             use function decode_channelID to decode
#             '''
#     #             channelID = struct.pack('H', np.uint16(80*column + row)) # unique ID for each pixel on FE.
#             channelID = np.uint16(80*column + row)
#             '''
#             channelID: 16 bit word
#                         first 7 bit: column
#                         following 9 bit: row
#             ch_2nd_dataword: 16 bit word. From MSB(left) to LSB(right)
#                         highest 4 bit: BCID
#                         second 4 bit: 0000 (in-BCID time res. can not be provided)
#                         third 4 bit: moduleID (0-7 => 0000 to 0111)
#                         lowest 4 bit: ToT
#             '''
# #             channelID = bitarray()
# #             channelID.extend(np.binary_repr(row<<7 ^ column, width=16))
# #             ch_2nd_dataword = bitarray()
# #             ch_2nd_dataword.extend(np.binary_repr(data_array['tot'][i]<<12 ^ moduleID<<8 ^ 0<<4 ^ data_array['relative_BCID'][i],width = 16))
#             ch_2nd_dataword = np.uint16(np.uint8(7*moduleID + data_array['tot'][i])<<8 ^ np.uint8(data_array['relative_BCID'][i]))
#           
#             ch_hit_data.extend((channelID, ch_2nd_dataword))

#     @profile
#     def process_data(self, data_array, moduleID):
#         '''
#         each hit is converted to two 16bit datawords, 1st word is pixel of
#         FE, second word is relBCID + number of FE + tot
#         --------------
#         Input:
#             moduleID: int, number of module in readout thread, to be determined from meta data
#             data_array: numpy array with interpreted hit data, dtypes as following
#                 [('event_number', '<i8'),
#                 ('trigger_number', '<u4'),
#                 ('trigger_time_stamp', '<u4'), 
#                 ('relative_BCID', 'u1'),
#                 ('LVL1ID', '<u2'),
#                 ('column', 'u1'),
#                 ('row', '<u2'), 
#                 ('tot', 'u1'),
#                 ('BCID', '<u2'),
#                 ('TDC', '<u2'), 
#                 ('TDC_time_stamp', 'u1'), 
#                 ('trigger_status', 'u1'),
#                 ('service_record', '<u4'),
#                 ('event_status', '<u2')]
#         ------------------
#         output:
#             self.ch_hit_data: list of bitarrays with converted hit datawords
#         '''
#         
#         self.ch_hit_data = hit_to_binary(data_array[['row','column','tot','relative_BCID']].copy(), moduleID)


#     @profile
    def build_header(self, n_hits, partitionID, cycleID, trigger_timestamp, bcids=15, flag=0):
        """
        builds data frame header from input information,
        python variables have to be converted to bitstrings.

        input variables:
        -----------------
            n_hits: int , used to calculate length of data frame, 1 hit = 4 byte
            partitionID: int, fixed for each MMC3 board
            cycleID: int, see below, provided by run control
            trigger_timestamp: int, used as frameTime (see below)

        format:
            size (2 byte): length of data frame in bytes including header
            partitionID (2 byte) = 0000 to 0002
            cycleIdentifier (4 byte): time of spill in SHiP time format: 0.2 seconds steps since the 8 April 2015
            frameTime (4 byte) = trigger_timestamp:  start of trigger window relative to SoC in 25ns steps
            timeExtent (2 byte) = relBCID : length of the trigger window in 25ns steps
            flags (2 byte) = empty for now

        """
#         data_header = "{:16b}".format(n_hits * 4 + 16) + "{:16b}".format(partitionID) + "{:32b}".format(
#         cycleID) + "{:32b}".format(trigger_timestamp) + "{:04b}".format(bcids) +
#             "{:04b}".format(flag)
#         data_header = struct.unpack(struct.pack('HHIIB', n_hits * 4 + 16, partitionID, cycleID, trigger_timestamp, "{:4b}".format(bcids) << 4))
        self.data_header = bitarray()
        self.data_header.extend(np.binary_repr(n_hits*4*16<<112 ^ partitionID<<96 ^ cycleID<<64 ^ trigger_timestamp<<32 ^ bcids<<16 ^ flag, width=128))


        print "data header:" , self.data_header


    def build_event_frame(self, data_header, hit_data):
        """
        builds frame to be sent to dispatcher
        input:
            data_header: bitarray with frame_header info
            hit_data: list of bitarrays, 2 strings (2 byte each) for each hit in event_frame
        """

        event_frame = []
        event_frame.extend((data_header,hit_data))
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
                        data_array = np.frombuffer(buf, dtype=dtype).reshape(shape)
                        event_array = build_events_from_raw_data(data_array)
                        self.analyze_raw_data(data_array)
                        self.ch_hit_data = process_data_numba(self.interpreter.get_hits(), moduleID=2)
                        self.build_header(
                            n_hits=len(self.ch_hit_data)/2, partitionID=3, cycleID=self.cycle_ID(), trigger_timestamp=54447,) # self.ch_hit_data
        
             
    def stop(self):
        self._stop_readout.set()


if __name__ == '__main__':

#     # Open th online monitor
#     socket_addr = "tcp://127.0.0.1:5678"
# #     Popen(["python", "../../pybar/online_monitor.py", socket_addr])  # if this call fails, comment it out and start the script manually
#     # Prepare socket
#     context = zmq.Context()
#     socket = context.socket(zmq.PUB)
#     socket.bind(socket_addr)
#     time.sleep(1)
#     # Transfer file to socket
#     while transfer_file("/media/data/SHiP/charm_exp_2018/test_data_converter/19_module_0_ext_trigger_scan.h5", socket=socket):
#         DataConverter(socket_addr=socket_addr)
#     # Clean up
#     socket.close()
#     context.term()
    
   
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

    DataConverter(socket_addr=socket_addr)

