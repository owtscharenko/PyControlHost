from optparse import OptionParser
import os
import zmq
import numpy as np
import datetime
from numba import njit
import cProfile
import cython
import multiprocessing
import threading, Queue
import time
import logging
from ctypes import c_ushort, c_int, c_char_p
import tables as tb

from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from  PyControlHost.ControlHost import CHostInterface, FrHeader, Hit



@njit             
def process_data_numba(data_array, flag=0):
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
        moduleID = data_array['moduleID'][i]
        channelID = np.uint16(column<<9 ^ row)

        ch_2nd_dataword = np.uint16(data_array['tot'][i]<<12 ^ moduleID<<8 ^ flag<<4 ^ np.uint8(data_array['relative_BCID'][i]))  
        ch_hit_data.extend((channelID, ch_2nd_dataword))
    return ch_hit_data


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


@cython.boundscheck(False)
@cython.wraparound(False)
def build_data(cycleID, partitionID, event_numbers, multimodule_hits, hit_data_dtype):
    headers, data = [],[]
    for event_number in event_numbers:
        event = np.array(multimodule_hits[np.where(multimodule_hits['event_number'] == event_number)],dtype = multimodule_hits.dtype)
        channelID = np.bitwise_or(event['row']<<7,event['column'],order='C',dtype=np.uint16)
        hit_data = np.bitwise_or(np.bitwise_or(event['tot']<<4,event['moduleID'])<<8,
                                 np.bitwise_or(0<<4,event['relative_BCID']),order='C',dtype=np.uint16)
        ch_hit_data = np.empty(shape = channelID.shape[0], dtype = hit_data_dtype) # TODO: can not create array in njit
        ch_hit_data["channelID"] = channelID
        ch_hit_data["hit_data"] = hit_data 
        # each event needs a frame header
        data_header = np.empty(shape=(1,), dtype= [("size", np.uint16),
                                                  ("partID", np.uint16),
                                                  ("cycleID",np.int32),
                                                  ("frameTime", np.uint32),
                                                  ("timeExtent", np.uint16),
                                                  ("flags",  np.uint16)])
        data_header['size'] = channelID.nbytes + 16
        data_header['partID'] = partitionID
        data_header['cycleID'] = cycleID
        data_header['frameTime'] = event['trigger_time_stamp'][0]
        data_header['timeExtent'] = 15
        data_header['flags'] = 0
        headers.append(data_header)
        data.append(ch_hit_data)
    return headers, data


@njit
def _build_hit_data(event):
    first_word = np.bitwise_or(event['tot']<<4,event['moduleID'])
    second_word = np.bitwise_or(0<<4,event['relative_BCID'])
    return np.bitwise_or(first_word<<8,second_word)

@njit
def _new_event(event_number_1, event_number_2):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number_1 != event_number_2
   
        
@njit
def merge_hits_tables(h1, h2, h3, h4, h5, h6, h7, h8, result):
    min_event_1 = h1["event_number"].min()
    min_event_2 = h2["event_number"].min()
    min_event_3 = h3["event_number"].min()
    min_event_4 = h4["event_number"].min()
    min_event_5 = h5["event_number"].min()
    min_event_6 = h6["event_number"].min()
    min_event_7 = h7["event_number"].min()
    min_event_8 = h8["event_number"].min()

    max_event_1 = h1["event_number"].max()
    max_event_2 = h2["event_number"].max()
    max_event_3 = h3["event_number"].max()
    max_event_4 = h4["event_number"].max()
    max_event_5 = h5["event_number"].max()
    max_event_6 = h6["event_number"].max()
    max_event_7 = h7["event_number"].max()
    max_event_8 = h8["event_number"].max()
    
    min_event = min(min_event_1, min_event_2, min_event_3, min_event_4, min_event_5, min_event_6, min_event_7, min_event_8)
    max_event = max(max_event_1, max_event_2, max_event_3, max_event_4, max_event_5, max_event_6, max_event_7, max_event_8)
    
    i, i1, i2, i3, i4, i5, i6, i7, i8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    for event in range(min_event, max_event + 1):
        while i1 < h1.shape[0] and h1[i1]["event_number"] == event:
            result[i] = h1[i1]
            i += 1
            i1 += 1
        while i2 < h1.shape[0] and h2[i2]["event_number"] == event:
            result[i] = h2[i2]
            i += 1
            i2 += 1
        while i3 < h3.shape[0] and h3[i3]["event_number"] == event:
            result[i] = h3[i3]
            i += 1
            i3 += 1
        while i4 < h4.shape[0] and h4[i4]["event_number"] == event:
            result[i] = h4[i4]
            i += 1
            i4 += 1
        while i5 < h5.shape[0] and h5[i5]["event_number"] == event:
            result[i] = h5[i5]
            i += 1
            i5 += 1
        while i6 < h6.shape[0] and h6[i6]["event_number"] == event:
            result[i] = h6[i6]
            i += 1
            i6 += 1
        while i7 < h7.shape[0] and h7[i7]["event_number"] == event:
            result[i] = h7[i7]
            i += 1
            i7 += 1
        while i8 < h8.shape[0] and h8[i8]["event_number"] == event:
            result[i] = h8[i8]
            i += 1
            i8 += 1
            


class Header_table(tb.IsDescription):
    event_number = tb.Int64Col(pos=0)    
    size = tb.UInt16Col(pos=1)
    partID = tb.UInt16Col(pos=2)
    cycleID = tb.UInt32Col(pos=3)
    frameTime = tb.UInt32Col(pos=4)
    timeExtent = tb.UInt16Col(pos=5)
    flags = tb.UInt16Col(pos=6)

    
class SHiP_data_table(tb.IsDescription):
    event_number = tb.Int64Col(pos=0)
    channelID = tb.UInt16Col(pos=1)
    hit_data = tb.UInt16Col(pos=2)



class DataConverter(multiprocessing.Process):

    def __init__(self, pybar_addr, ports, partitionID, disp_addr=None):
        
        multiprocessing.Process.__init__(self)
        self.logger = logging.getLogger('DataConverter')
#         self.connect(socket_addr)
        self.n_readout = 0
        self.n_modules = 8
        self.socket_addr = pybar_addr
        self.address = pybar_addr#[:-4]
        self.ports = ports
        
        self.multi_chip_event_dtype =[('event_number', '<i8'),
                                        ('trigger_number', '<u4'),
                                        ('trigger_time_stamp', '<u4'), 
                                        ('relative_BCID', 'u1'),
#                                         ('LVL1ID', '<u2'),
                                        ('moduleID','u1'),
                                        ('column', 'u1'),
                                        ('row', '<u2'), 
                                        ('tot', 'u1'),
#                                         ('BCID', '<u2'),
#                                         ('TDC', '<u2'), 
#                                         ('TDC_time_stamp', 'u1'), 
#                                         ('trigger_status', 'u1'),
#                                         ('service_record', '<u4'),
#                                         ('event_status', '<u2')
                                    ]
        
        self._stop_readout = multiprocessing.Event()  # exit signal
        self.SoR_flag = multiprocessing.Event()
        self.EoR_flag = multiprocessing.Event()
        self.SoS_flag = multiprocessing.Event()
        self.EoS_flag = multiprocessing.Event()
        self.EoS_data_flag = multiprocessing.Event()
        self.fifo_read = multiprocessing.Event()
        self.worker_reset_flag = multiprocessing.Event()
        self.worker_finished_flags= [multiprocessing.Event() for _ in range(self.n_modules)]
        self.worker_reset_finished = [multiprocessing.Event() for _ in range(self.n_modules)]
        self.send_data_flag = [multiprocessing.Event() for _ in range(self.n_modules)]
        self.arrays_read_flag = multiprocessing.Event()
        self.dummy_flag = multiprocessing.Event()
        self.all_workers_finished = multiprocessing.Event()
        self.reset_lock = multiprocessing.Lock()
         
#         self.raw_data_queue = multiprocessing.Queue()
        self.kill_received = False
        self.ch = CHostInterface()
        self.total_events = 0
        self.start_date = datetime.datetime(2015, 04, 8, 12, 00)
        self.cycle_ID = multiprocessing.Value('i',0)
        self.n_spills = multiprocessing.Value('i',0)
        self.n_events = multiprocessing.Value('i',0)
        self.file_date = (self.start_date + datetime.timedelta(seconds = self.cycle_ID.value /5.)).strftime("%Y_%m_%d_%H_%M_%S")
        self.run_number = multiprocessing.Value('i',0)
        self.spill_file_name = './default.h5'
        self.partitionID = partitionID # '0X0802' from 0800 to 0802 
        
        # in case of direct call of DataConverter, the partitionID is handed over as hex TODO: fix this behavior
        if disp_addr: 
            self.DetName = 'Pixels' + partitionID[4:] + '_LocDaq_0' + partitionID[2:]
            self.RAW_data_tag = 'RAW_0' + partitionID[2:]
        else: 
            self.DetName = 'Pixels' + hex(partitionID)[4:] + '_LocDaq_0' + hex(partitionID)[2:]
            self.RAW_data_tag = 'RAW_0' + hex(partitionID)[2:]
        if disp_addr:
            self.ch.connect_CH(disp_addr,self.DetName)
        

    def get_cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        return np.uint32((datetime.datetime.now() - self.start_date).total_seconds() * 5)
    
    
    def get_file_date(self,cycleID):
        '''DO NOT USE IN TESTBEAM!!!!
        function for arbitrary cycleIDs 
        '''
        return (self.start_date + datetime.timedelta(seconds = cycleID /5.)).strftime("%Y_%m_%d_%H_%M_%S")


    def setup_raw_data_analysis(self):
        self.interpreters = []
        self.hits = []
        for _ in range(self.n_modules):
            interpreter = PyDataInterpreter()
            interpreter.create_empty_event_hits(True)
            interpreter.set_trigger_data_format(1)
            interpreter.align_at_trigger(True)
            interpreter.set_warning_output(False)
            interpreter.set_FEI4B(True)
            self.interpreters.append(interpreter)
            self.hits.append(np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype,order='C')))

    def connect(self, socket_addr):
        self.socket_addr = socket_addr
        self.context = zmq.Context()
        self.socket_pull = self.context.socket(zmq.SUB)  # subscriber
        self.socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data
        self.socket_pull.connect(self.socket_addr)
        self.logger.info('DataConverter connected to %s' % self.socket_addr)


    def reset(self,cycleID=0, msg=None):
        if msg:
            self.logger.info(msg)
        self.worker_reset_flag.set()
        with self.reset_lock:
            self.n_readout = 0
            self.total_events = 0
            self.logger.info('last cycleID=%s'% self.cycle_ID.value)
#             self.cycle_ID.value = cycleID # TODO: careful with multiprocessing values!
            self.run_number.value = 0
            self.spill_file_name = './default.h5'
            self._stop_readout.clear()
            
            for worker_flag in self.worker_finished_flags:
                worker_flag.clear()
            for flag in self.send_data_flag:
                flag.clear()
            self.all_workers_finished.clear()
            self.SoR_flag.clear()
            self.EoR_flag.clear()
            self.SoS_flag.clear()
            self.dummy_flag.clear()
            self.n_spills.value = 0
            self.arrays_read_flag.clear()
            self.EoS_data_flag.clear()
            
#             for flag in self.worker_reset_finished:
#                 if flag.wait(0.1):
#                     self.worker_reset_flag.clear()
#                 else: self.worker_reset_flag.set()
            self.EoS_flag.clear()
        self.logger.info('DataConverter has been reset')

    
    def SoS_reset(self): 
        ''' 
        for each spill a file with the SHiP cycleID will be created,
        the cycleID is therefore converted to human readable string.
        format: year, month, day, hour, minute, second
        '''
        with self.reset_lock:
            self.EoS_data_flag.clear()
            self.all_workers_finished.clear()
            self.worker_reset_flag.clear()
            self.dummy_flag.clear()
            self.n_spills.value += 1
#             self.file_date = (self.start_date + datetime.timedelta(seconds = self.cycle_ID.value/5.)).strftime("%Y_%m_%d_%H_%M_%S")
#             print "spill time:", datetime.timedelta(seconds = self.cycle_ID.value/5.).strftime("%Y_%m_%d_%H_%M_%S")
#             self.spill_file_name = "./RUN_%03d/%s.txt" % (self.run_number.value, self.file_date) 
        self.logger.info('SoS reset finished')

    
    def EoS_reset(self):
        '''after all events have been sent, the event arrays are emtpied, and interpreters are reset'''
        with self.reset_lock:
            for flag in self.worker_finished_flags:
                flag.clear()
            for flag in self.worker_reset_finished:
                flag.clear()
            for flag in self.send_data_flag:
                flag.clear()
            self.arrays_read_flag.clear()
            self.fifo_read.clear()
            self.EoS_flag.clear()
        self.logger.info('EoS reset finished')
    
    
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
    
    
    def _module_worker(self,socket_addr, moduleID, send_end):
        '''one worker for each FE chip, since RAW data comes from FIFO separated by moduleID.
           It is necessary to instantiate zmq.Context() in calling method. Otherwise it has no acces when called as multiprocessing.process.
        '''
        context = zmq.Context()
        socket_pull = context.socket(zmq.PULL)  # subscriber
#         socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data needed for PUB/SUB but not for PUSH/PULL
        socket_pull.bind(socket_addr)
        self.logger.info("Worker %s started, socket %s" % (moduleID, socket_addr))
        hit_array = np.zeros(shape=(0,),dtype = self.multi_chip_event_dtype,order='C')
        counter = 0
        interpreter = PyDataInterpreter()
        interpreter.create_empty_event_hits(True)
        interpreter.set_trigger_data_format(1)
        interpreter.align_at_trigger(True)
        interpreter.set_warning_output(False)
        interpreter.set_FEI4B(True)
        
        while not self._stop_readout.wait(0.01) :  # use wait(), do not block here
            if self.worker_reset_flag.is_set() and not self.worker_reset_finished[moduleID].is_set():
                with self.reset_lock:
                    interpreter.reset()
                    self.logger.info("Resetting worker %s" % (moduleID))
                    self.worker_reset_finished[moduleID].set()
            if self.EoS_flag.is_set() and not self.worker_finished_flags[moduleID].is_set(): # EoS_flag is set in run_control after reception of EoS command 
                while counter < 3:
                    self.dummy_flag.wait(0.03)
                    try:
                        meta_data = socket_pull.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                    else:
                        name = meta_data.pop('name')
                        if name == 'ReadoutData':
                            data = socket_pull.recv()
                            # reconstruct numpy array
                            buf = buffer(data)
                            dtype = meta_data.pop('dtype')
                            shape = meta_data.pop('shape')
                            data_array = np.frombuffer(buf, dtype=dtype).reshape(shape)
                            
                            interpreter.interpret_raw_data(data_array)
        #                     self.analyze_raw_data(raw_data=np.ascontiguousarray(data_array), module=moduleID)
                            # build new array with moduleID, take only necessary data
                            hits = interpreter.get_hits()
        #                     hits = self.interpreters[moduleID].get_hits()
                            module_hits = np.zeros(shape=(hits.shape[0],),dtype = self.multi_chip_event_dtype, order = 'C')
                            module_hits['event_number'] = hits['event_number']
                            module_hits['trigger_number'] = hits['trigger_number']
                            module_hits['trigger_time_stamp'] = hits['trigger_time_stamp']
                            module_hits['relative_BCID'] = hits['relative_BCID']
                            module_hits['column'] = hits['column']
                            module_hits['row'] = hits['row']
                            module_hits['tot'] = hits['tot']
                            module_hits['moduleID'] = moduleID
                            
                            hit_array = np.concatenate((hit_array,module_hits))
                    
                    counter +=1

#                 if hit_array.shape[0] > 0:
#                     self.logger.info("hit array shape worker %s before store event" % moduleID, hit_array.shape)
#                     self.logger.info("hit array worker %s last entry before store event" % moduleID, hit_array[-1])
                
#                 interpreter.store_event()
#                   
#                 hits = interpreter.get_hits()
#  
#                 module_hits = np.zeros(shape=(hits.shape[0],),dtype = self.multi_chip_event_dtype, order = 'C')
#                 module_hits['event_number'] = hits['event_number']
#                 module_hits['trigger_number'] = hits['trigger_number']
#                 module_hits['trigger_time_stamp'] = hits['trigger_time_stamp']
#                 module_hits['relative_BCID'] = hits['relative_BCID']
#                 module_hits['column'] = hits['column']
#                 module_hits['row'] = hits['row']
#                 module_hits['tot'] = hits['tot']
#                 module_hits['moduleID'] = moduleID
#                 # append chunk to hit array
#                 hit_array = np.concatenate((hit_array,module_hits))
                
#                 if hit_array.shape[0] > 0:
#                     self.logger.info("hit array shape worker %s " % moduleID, hit_array.shape)
#                     self.logger.info("hit array worker %s last entry" % moduleID, hit_array[-1])
                self.send_data_flag[moduleID].set()
                
                if hit_array.shape[0] > 0 and self.n_spills.value > 1:
                    hit_array2 = hit_array[np.where(hit_array['event_number'] != hit_array[0]['event_number'])]
                    send_array = hit_array2.copy()
                else:
                    send_array = hit_array.copy()
                send_end.send(send_array) 
                
#                 send_array = hit_array.copy()
#                 send_end.send(hit_array) 
                self.worker_finished_flags[moduleID].set()
                self.logger.info("Worker %s finished, received %s hits" % (moduleID , hit_array.shape))
                hit_array = np.zeros(shape=(0,),dtype = self.multi_chip_event_dtype,order='C')
                counter = 0
            try:
                meta_data = socket_pull.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            else:
                name = meta_data.pop('name')
                if name == 'ReadoutData':
                    data = socket_pull.recv()
                    # reconstruct numpy array
                    buf = buffer(data)
                    dtype = meta_data.pop('dtype')
                    shape = meta_data.pop('shape')
                    data_array = np.frombuffer(buf, dtype=dtype).reshape(shape)
                    
                    interpreter.interpret_raw_data(data_array)
#                     self.analyze_raw_data(raw_data=np.ascontiguousarray(data_array), module=moduleID)
                    # build new array with moduleID, take only necessary data
                    hits = interpreter.get_hits()
#                     hits = self.interpreters[moduleID].get_hits()
                    module_hits = np.zeros(shape=(hits.shape[0],),dtype = self.multi_chip_event_dtype, order = 'C')
                    module_hits['event_number'] = hits['event_number']
                    module_hits['trigger_number'] = hits['trigger_number']
                    module_hits['trigger_time_stamp'] = hits['trigger_time_stamp']
                    module_hits['relative_BCID'] = hits['relative_BCID']
                    module_hits['column'] = hits['column']
                    module_hits['row'] = hits['row']
                    module_hits['tot'] = hits['tot']
                    module_hits['moduleID'] = moduleID
                    # append chunk to hit array
#                     hit_array = np.r_[hit_array,module_hits]
                    hit_array = np.concatenate((hit_array,module_hits))
#                     if hit_array.shape[0] > 0:
#                         self.logger.info("loop hit array worker %s last entry " %moduleID, hit_array[-1])
#                         self.logger.info("loop hit array worker %s shape" %moduleID, hit_array.shape)

    
    def run(self):
        ''' create workers upon start and collect data after EoS'''
        
        self.jobs = []
        self.pipes = []
        hit_arrays = []
        
        for _ in range(self.n_modules):
            hit_arrays.append(0)
        
        for module in range(self.n_modules):
            recv_end, send_end = multiprocessing.Pipe(False)
            worker = multiprocessing.Process(target = self._module_worker, args =(self.address + self.ports[module], module, send_end))
            worker.name = 'RecieverModule_%s' % module
            worker.daemon = True
            self.jobs.append(worker)
            self.pipes.append(recv_end)
            worker.start()
#             print "PID of worker %s:"% module, worker.pid
        
        while not self._stop_readout.wait(0.01):

            if self.EoS_flag.is_set(): # EoS_flag is set in run_control after reception of EoS command 
                #TODO: check building of common event from all modules
                with self.reset_lock:
                    start = datetime.datetime.now()
                    if not self.EoS_data_flag.is_set(): # EoS_data_flag is set after all events are sent to dispatcher
#                         if all(flag.is_set() for flag in self.worker_finished_flags):
#                             self.all_workers_finished.set()
#                         while not all(flag.is_set() for flag in self.worker_finished_flags):
#                             continue
#                         if self.all_workers_finished.is_set():                        

                        if all(pipe.poll(0.1) for pipe in self.pipes):
                            self.all_workers_finished.set() # triggers DAQDONE message for SoS
#                         if all(flag.is_set() for flag in self.send_data_flag):
                            for i, pipe in enumerate(self.pipes):
                                hit_arrays[i] = pipe.recv().copy()
                                
                            self.logger.info('All workers finished, starting conversion')
                            nhits = 0
                            for hit_array in hit_arrays:
                                nhits += hit_array.shape[0]
                                
                            start = datetime.datetime.now()
                            # merge hit data of 8 FE to one array, sorted by event_number
                            merge_table = np.zeros(shape=(nhits,),dtype = self.multi_chip_event_dtype)
                            merge_hits_tables(hit_arrays[0],hit_arrays[1],hit_arrays[2],hit_arrays[3],hit_arrays[4],
                                              hit_arrays[5],hit_arrays[6],hit_arrays[7],merge_table)
                            multimodule_hits = merge_table.copy()
                            self.arrays_read_flag.set()
                            self.SoS_flag.clear()  
                            
                            event_numbers , indices = np.unique(multimodule_hits['event_number'],return_index = True)
                            print 'time for sorting:', datetime.datetime.now() - start
                            self.n_events.value = indices.shape[0]
                            print 'nevents = %s , nhits = %s, ' %(self.n_events.value, nhits) #  last event number = %s" %(n_events, nhits, event_numbers[-1])
    
                            headers = np.zeros(shape = (self.n_events.value,), dtype = [('event_number', np.int64), ('size',np.uint16), ('partID', np.uint16), ('cycleID', np.uint32), ('frameTime', np.int32), ('timeExtent', np.uint16), ('flags', np.uint16)])
                            hits = np.zeros(shape = (nhits,), dtype = [('event_number', np.int64),('channelID', np.uint16),('hit_data', np.uint16)])
                            ch_event_numbers = np.arange(0,self.n_events.value +1)
                            with tb.open_file('./run_data/partition_%s_run_%03d.h5' % (hex(self.partitionID), self.run_number.value), mode='a', title="SHiP_raw_data") as run_file:
                                
                                self.logger.info('opening run file %s' % run_file.filename)
#                                 self.file_date = (self.start_date + datetime.timedelta(seconds = self.cycle_ID.value/5.)).strftime("%Y_%m_%d_%H_%M_%S")
                                spill_group = run_file.create_group(where = "/", name = 'Spill_%s' % self.cycle_ID.value, title = 'Spill_%s' % self.cycle_ID.value, filters = tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                                header_table = run_file.create_table(where = spill_group, name = 'Headers', description = Header_table, title = 'Headers to event data')
                                hit_table = run_file.create_table(where = spill_group, name = 'Hits', description = SHiP_data_table, title = 'Hits')
                                for i, index in enumerate(indices):
#                                     print "getting event %s" %i
                                    if i == self.n_events.value-1:
                                        event = multimodule_hits[index:]
                                    else:
                                        event = multimodule_hits[index:indices[i+1]]
                                    # build SHiP data format
                                    if event.shape[0] == 0:
                                        print '      =======no hits in event %s=======' %i
                                        print '      =======no hits in event %s=======' %i
                                        print '      =======no hits in event %s=======' %i
                                        continue
                                    channelID = np.bitwise_or(event['row']<<7,event['column'],order='C',dtype=np.uint16)
                                    first_word = np.bitwise_or(event['tot']<<4,event['moduleID'],dtype=np.uint16)
                                    second_word = np.bitwise_or(0<<4,event['relative_BCID'],dtype=np.uint16)
                                    hit_data = np.bitwise_or(first_word<<8,second_word, order = 'C', dtype= np.uint16)
                                    
                                    #fill numpy array with SHiP data
                                    self.ch_hit_data = np.zeros(channelID.shape[0], dtype = Hit)
                                    self.ch_hit_data["channelID"] = channelID
                                    self.ch_hit_data["hit_data"] = hit_data
                                    
                                    # each event needs a frame header, extra array for header
                                    self.data_header = np.zeros(shape=(1,), dtype= FrHeader)
                                    self.data_header['size'] = channelID.nbytes + 16
                                    self.data_header['partID'] = self.partitionID
                                    self.data_header['cycleID'] = self.cycle_ID.value
                                    self.data_header['frameTime'] = event[0]['trigger_time_stamp']
                                    self.data_header['timeExtent'] = ch_event_numbers[i]
                                    self.data_header['flags'] = 0
                                    
                                    # send header and hit_data to dispatcher
                                    self.ch.send_data(self.RAW_data_tag, self.data_header, self.ch_hit_data)
                                    
                                    # write hits to numpy array for local storage
                                    if i == self.n_events.value-1:
                                        hits[index:]['event_number'] = event['event_number']
                                        hits[index:]['channelID'] = self.ch_hit_data['channelID']
                                        hits[index:]['hit_data'] = self.ch_hit_data['hit_data']
                                    else:
                                        hits[index:indices[i+1]]['event_number'] = event['event_number']
                                        hits[index:indices[i+1]]['channelID'] = self.ch_hit_data['channelID']
                                        hits[index:indices[i+1]]['hit_data'] = self.ch_hit_data['hit_data']
                                    
                                    # write header to numpy array for local storage
                                    headers[i]['event_number'] = event[0]["event_number"]
                                    headers[i]['size'] = self.data_header['size']
                                    headers[i]['partID'] = self.data_header['partID']
                                    headers[i]['cycleID'] = self.data_header['cycleID']
                                    headers[i]['frameTime'] = self.data_header['frameTime']
                                    headers[i]['timeExtent'] = self.data_header['timeExtent']
                                    headers[i]['flags'] = self.data_header['flags']

                                # save numpy arrays in .h5 file
                                hit_table.append(hits)
                                header_table.append(headers)
                                header_table.flush()
                                hit_table.flush()
                            self.EoS_data_flag.set()
                            # clear hit arrays for reception of worker data
                            for i in range(self.n_modules):
                                hit_arrays[i] = 0
                            self.logger.info('Sent %s events with %s hits' %(self.n_events.value, nhits))
                            print "time needed for merging, sending and saving %s events : %s" %(self.n_events.value,(datetime.datetime.now()-start))
                
                        
    def stop(self):
        self.logger.info('Stopping converter')
        self._stop_readout.set()
#         for job in self.jobs:
#             job.stop()
#         self.logger.info('All Workers Joined')



if __name__ == '__main__':
   
    usage = "Usage: %prog ADDRESS"
    description = "ADDRESS: Remote address of the sender (default: tcp://127.0.0.1:5678).\n disp_addr: address of dispatcher to send data to (default 127.0.0.1) \n partitionID : identifier for ControlHost (default= 0x0802)"
    parser = OptionParser(usage, description=description)
    options, args = parser.parse_args()
    if len(args) == 0:
        socket_addr = 'tcp://127.0.0.1:5678'
        disp_addr = '127.0.0.1'
        partitionID = "0X0802"
    elif len(args) == 3:
        socket_addr = args[0]
        disp_addr = args[1]
        partitionID = args[2]
    else:
        parser.error("incorrect number of arguments")
     
    pr = cProfile.Profile()
    ports = ['5001','5002','5003','5004','5005','5006','5007','5008',]
    try:
        converter = DataConverter(pybar_addr=socket_addr,disp_addr = disp_addr, ports = ports, partitionID = partitionID)
        converter.start()
    except (KeyboardInterrupt, SystemExit):
        print "keyboard interrupt"
        pr.disable()
        pr.print_stats(sort='ncalls')

