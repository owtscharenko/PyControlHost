from optparse import OptionParser
import zmq
import numpy as np
import datetime
from numba import njit
import cProfile
import multiprocessing
import logging

from build_binrep import hit_to_binary
from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from pybar_fei4_interpreter.data_histograming import PyDataHistograming
from run_control import run_control, ch_communicator

        
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
            

            channelID: 16 bit word From MSB(left) to LSB(right)
                        highest 7 bit: column
                        following 9 bit: row
            ch_2nd_dataword: 16 bit word. From MSB(left) to LSB(right)
                        highest 4 bit: ToT
                        second 4 bit: moduleID (0-7 => 0000 to 0111)
                        third 4 bit:  0000 (in-BCID time res. can not be provided) used for flgas
                        lowest 4 bit: BCID

    '''
    ch_hit_data = []
    
    for i in range(data_array.shape[0]):
        row = data_array['row'][i]
        column = data_array['column'][i]
            
        channelID = np.uint16(column<<9 ^ row)

        ch_2nd_dataword = np.uint16(data_array['tot'][i]<<12 ^ moduleID<<8 ^ flag<<4 ^ np.uint8(data_array['relative_BCID'][i]))  
        ch_hit_data.extend((channelID, ch_2nd_dataword))
    return ch_hit_data


def build_header(n_hits, partitionID, cycleID, trigger_timestamp, bcids=15, flag=0):
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
    data_header = (n_hits*4+16)<<112 ^ partitionID<<96 ^ cycleID<<64 ^ trigger_timestamp<<32 ^ bcids<<16 ^ flag # may not work as expected: getsizeof(2**128) = 44 byte
#     data_header = [(n_hits*4+16)<<48 ^ partitionID<<32 ^ cycleID , trigger_timestamp<<32 ^ bcids<<16 ^ flag] # two 64bit int may be better than one 128 bit int
#         self.data_header = bitarray()
#         self.data_header.extend(np.binary_repr(n_hits*4*16<<112 ^ partitionID<<96 ^ cycleID<<64 ^ trigger_timestamp<<32 ^ bcids<<16 ^ flag, width=128))

    return data_header


@njit
def decode_channelID(channelID):

#     converts channelID (0 to 26879, each value corresponds to one pixel of one FE) to column, row
    '''
    converts channelID to column and pixel. 16 bit uint: highest 7bit = column, lowest 9 bit = row
    input:
        uint16
    returns: 
        tuple column(np.uint8), row(np.uint16)
    '''

#     row = np.uint16(0)
#     column = np.uint16(0)
#     if channelID == 0:
#         row = 0
#         column = 0
#         return column, row
#     column = channelID % 80
#     row = (channelID - 79) / 80
    column = np.uint8(channelID>>9)
    row=np.uint16(channelID ^ (column<<9))
    
    return column, row


@njit
def decode_second_dataword(dataword):
    '''
    converts second dataword (16bit: highest 4 bit(16-13) tot, bit 12-9 moduleID, bit 8-5 flags=0000, lowest 4bit (4-1) BCID) to single values
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

            
class DataConverter(multiprocessing.Process):

    def __init__(self, socket_addr, partitionID):
        multiprocessing.Process.__init__(self)
#         self.connect(socket_addr)
        self.n_readout = 0
        self.n_modules = 8
        self.socket_addr = socket_addr
        self._stop_readout = multiprocessing.Event()  # exit signal
        self.setup_raw_data_analysis()
        self.reset_lock = multiprocessing.Lock()  # exit signal
        self.kill_received = False
        self.ch = ch_communicator()
        self.total_events = 0
        self.cycle_ID = 0
        self.partitionID = partitionID
        

    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return np.uint32((datetime.datetime.now() - start_date).total_seconds() * 5)


    def setup_raw_data_analysis(self):
        self.interpreters = []
        for _ in range(self.n_modules):
            interpreter = PyDataInterpreter()
            interpreter.create_empty_event_hits(True)
            interpreter.set_trigger_data_format(1)
            interpreter.align_at_trigger(True)
            interpreter.set_warning_output(False)
            interpreter.set_FEI4B(True)
            self.interpreters.append(interpreter)


    def connect(self, socket_addr):
        self.socket_addr = socket_addr
        self.context = zmq.Context()
        self.socket_pull = self.context.socket(zmq.SUB)  # subscriber
        self.socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data
        self.socket_pull.connect(self.socket_addr)
        logging.info('DataConverter connected to %s' % self.socket_addr)


    def reset(self,cycleID=0, msg=None):
        if msg:
            logging.info(msg)
        with self.reset_lock:
            for interpreter in self.interpreters:
                interpreter.reset()
#             self.interpreter.reset()
            self.n_readout = 0
            self.total_events = 0
            logging.info('last cycleID=%s'% self.cycle_ID)
            self.cycle_ID = cycleID
            self._stop_readout.clear()


    def analyze_raw_data(self, raw_data, module):
        return self.interpreters[module].interpret_raw_data(raw_data)


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

#     @profile
    def run(self):
        ''' necessary to instantiate zmq.Context() in run method. Otherwise run has no acces to it.'''
        self.context = zmq.Context()
        self.socket_pull = self.context.socket(zmq.SUB)  # subscriber
        self.socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data
        self.socket_pull.connect(self.socket_addr)
        logging.info('DataConverter connected to %s' % self.socket_addr)
        logging.info('cycleID = %s' % self.cycle_ID)
        
        while not self._stop_readout.wait(0.01):  # use wait(), do not block here
            logging.info('DataConverter running and accepting RAWDATA')s
            with self.reset_lock:
                try:
                    meta_data = self.socket_pull.recv_json(flags=zmq.NOBLOCK) # 
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
#                         pr.enable()
                        #sort hits by frontend
                        for module in range(8):
                            selection_frontend = np.bitwise_and(data_array, 0x0F000000) == np.left_shift(module + 1, 24)
                            selection_trigger = np.bitwise_and(data_array, 0x80000000) == np.left_shift(1, 31)
                            selection = np.logical_or(selection_frontend, selection_trigger)

                            self.analyze_raw_data(raw_data=np.ascontiguousarray(data_array[selection]), module=module)
                            hits = self.interpreters[module].get_hits()
                            #TODO: build one event from all modules
                            _, event_indices = np.unique(hits['event_number'], return_index = True) # count number of events in array
                        
                        
                        for event_table in np.array_split(hits, event_indices)[1:]: # split hit array by events. 1st list entry is empty
                            self.ch_hit_data = process_data_numba(event_table, moduleID=2) # TODO: get moduleID from datastream
                            self.data_header = [build_header( # each event needs a frame header
                                n_hits=len(self.ch_hit_data)/2, partitionID=self.partitionID, cycleID=self.cycleID, trigger_timestamp=54447)] #TODO: get trigger_timestamp.
                            self.ch.send_data(np.ascontiguousarray(self.data_header + self.ch_hit_data)) 
                        self.total_events += event_indices.shape[0]
#                         logging.info('total events %s' % self.total_events)
                        
    def stop(self):
        self._stop_readout.set()
        logging.info('Stopping converter')


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
     
    CH = ch_communicator()
    pr = cProfile.Profile()
     
    try:
        DataConverter(socket_addr=socket_addr)
    except (KeyboardInterrupt, SystemExit):
        print "keyboard interrupt"
        pr.disable()
        pr.print_stats(sort='ncalls')

