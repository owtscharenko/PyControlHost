from optparse import OptionParser
import os
import zmq
import numpy as np
import datetime
from numba import njit, jit
import numba
import cProfile
import cython
import multiprocessing, threading
import time
import logging
from ctypes import c_ushort, c_int, c_char_p

from pybar_fei4_interpreter.data_interpreter import PyDataInterpreter
from pybar_fei4_interpreter.data_histograming import PyDataHistograming
from pybar_fei4_interpreter.data_struct import HitInfoTable
# import run_control.RunControl as RunControl
from  ControlHost import CHostInterface, FrHeader, Hit
from control_host_coms import build_and_send_data




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
                                                  ("frameTime", np.int32),
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
def _new_event(event_number_1, event_number_2):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number_1 != event_number_2

@njit
def sort_mutlimodule_hits(multimodule_hits):
    'sort array'
    return multimodule_hits.sort()

@njit
def build_from_sorted(multimodule_hits):
    'build events from sorted array'
    total_hits = multimodule_hits.shape[0]
    event_number = multimodule_hits[0]['event_number']
    start_event_hit_index = 0
    
    for i in range(total_hits):
        if _new_event(multimodule_hits[i]['event_number'], event_number):
            pass
            

@njit
def build_data_while(cycleID, partitionID,hit_arrays, event_numbers):
    a,b,c,d,e,f,g,h = 0
    hit_list = []
    flag = np.zeros((8,1))
    it = np.nditer(event_numbers,flags=['external_loop'])
    for _ in hit_arrays:
        module_hits = []
        hit_list.append(module_hits)
    while not it.finished:
        if hit_arrays[0]['event_number'][a] == event_number:
            hit_list[0].append(hit_arrays[0][a])
            a +=1
        else:
            flag[0] = True
        if hit_arrays[1]['event_number'][b] == event_number:
            hit_list[1].append(hit_arrays[1][b])
            b +=1
        else:
            flag[1] = True
        if hit_arrays[2]['event_number'][c] == event_number:
            hit_list[2].append(hit_arrays[2][c])
            c +=1
        else:
            flag[2] = True
        if hit_arrays[3]['event_number'][d] == event_number:
            hit_list[3].append(hit_arrays[3][d])
            d +=1
        else:
            flag[3] = True
        if hit_arrays[4]['event_number'][e] == event_number:
            hit_list[4].append(hit_arrays[4][e])
            e +=1
        else:
            flag[4] = True
        if hit_arrays[5]['event_number'][f] == event_number:
            hit_list[5].append(hit_arrays[5][f])
            f +=1
        else:
            flag[5] = True
        if hit_arrays[6]['event_number'][g] == event_number:
            hit_list[6].append(hit_arrays[6][g])
            g +=1
        else:
            flag[6] = True
        if hit_arrays[7]['event_number'][h] == event_number:
            hit_list[7].append(hit_arrays[7][h])
            h +=1
        else:
            flag[7] = True
        if flag.all:
            it.iternext()
        
        
class DataConverter(multiprocessing.Process):

    def __init__(self, pybar_addr, ports, partitionID, disp_addr=None):
        
        multiprocessing.Process.__init__(self)
        self.logger = logging.getLogger('DataConverter')
#         self.connect(socket_addr)
        self.n_readout = 0
        self.n_modules = 8
        self.socket_addr = pybar_addr
        self.address = pybar_addr[:-4]
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
        self.worker_reset_flag = multiprocessing.Event()
        self.reset_multimodule_hits = multiprocessing.Event()
        self.worker_finished_flags= [multiprocessing.Event() for _ in range(self.n_modules)]
        self.reset_lock = multiprocessing.Lock() 
        
        self.raw_data_queue = multiprocessing.Queue()
        
        self.setup_raw_data_analysis()
        
        self.kill_received = False
        self.ch = CHostInterface()
        self.total_events = 0
        self.start_date = datetime.datetime(2015, 04, 8, 00, 00)
        self.cycle_ID = multiprocessing.Value('i',0)
        self.file_date = multiprocessing.Value(c_char_p,(self.start_date + datetime.timedelta(seconds = self.cycle_ID.value /5.)).strftime("%Y_%m_%d_%H_%M_%S"))
        self.run_number = multiprocessing.Value('i',0)
        self.partitionID = partitionID # '0X0802' from 0800 to 0802 how to get this from scan instance?
        if disp_addr: # in case of direct call of DataConverter, the partitionID is handed over as hex TODO: fix this behavior
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


    def setup_raw_data_analysis(self):
        self.interpreters = []
        self.hits = []
        self.multimodule_hits = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype))
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
        with self.reset_lock:
            self.n_readout = 0
            self.total_events = 0
            self.logger.info('last cycleID=%s'% self.cycle_ID.value)
            self.cycle_ID.value = cycleID # TODO: careful with multiprocessing values!
            self._stop_readout.clear()
            for interpreter in self.interpreters:
                interpreter.reset()
            for hit in self.hits:
                hit = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype,order='C'))
            self.worker_reset_flag.set()
            self.reset_multimodule_hits.set()
            for worker_flag in self.worker_finished_flags:
                worker_flag.clear()
            self.SoR_flag.clear()
            self.EoR_flag.clear()
            self.SoS_flag.clear()
            self.EoS_flag.clear()
            self.EoS_data_flag.clear()
#             self.worker_reset_flag.set()
            self.logger.info('DataConverter has been reset')

    
    def SoS_reset(self): # TODO: implement SoS and EoS reset.
        ''' for each spill a file with the SHiP cycleID will be created,
        the cycleID is therefore converted to human readable string.
        format: year, month, day, hour, minute, second
        '''
        self.EoS_data_flag.clear()
        self.file_date.value = (self.start_date + datetime.timedelta(seconds = self.cycle_ID.value /5.)).strftime("%Y_%m_%d_%H_%M_%S")
        self.logger('SoS reset finished')

    
    def EoS_reset(self):
        '''after all events have been sent, the event arrays are emtpied, and interpreters are reset'''
        with self.reset_lock:
            for interpreter in self.interpreters:
                interpreter.reset()
            for hit in self.hits:
                hit = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype,order='C'))
            self.worker_reset_flag.set()
            self.reset_multimodule_hits.set()
            for worker_flag in self.worker_finished_flags:
                worker_flag.clear()
    
    
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
        '''one worker for each FE chip, since RAW data comes from FIFO separated by moduleID
           It is necessary to instantiate zmq.Context() in run method. Otherwise run has no acces when called as multiprocessing.process.
        '''
        context = zmq.Context()
        socket_pull = context.socket(zmq.SUB)  # subscriber
        socket_pull.setsockopt(zmq.SUBSCRIBE, '')  # do not filter any data
        socket_pull.connect(socket_addr)
        self.logger.info("Worker started, socket %s" % (socket_addr))
        while not self._stop_readout.wait(0.001) :  # use wait(), do not block here
#             with self.reset_lock:
            if self.EoS_flag.is_set(): # EoS_flag is set in run_control after reception of EoS command 
                send_end.send(self.hits[moduleID]) # TODO: make sure all evts. are read out befor sending
                self.worker_finished_flags[moduleID].set()
                self.logger.info("Worker finished, received %s hits" % (self.hits[moduleID].shape)) # TODO: logger behaviour and content of hits after EoS reset is weird
                
#             if self.worker_reset_flag.is_set():
#                 self.logger.info("started resetting worker")
#                 self.hits[moduleID] = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype,order='C'))
#                 self.worker_finished_flags[moduleID].clear() 
#                 self.worker_reset_flag.clear()
#                 self.logger.info('Worker has been reset')
            if self.worker_finished_flags[moduleID].is_set() :
                continue
                
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
                    
                    self.analyze_raw_data(raw_data=np.ascontiguousarray(data_array), module=moduleID)
                    # build new array with moduleID, take only important data
                    hits = self.interpreters[moduleID].get_hits()
                    module_hits = np.empty(shape=(hits.shape[0],),dtype = self.multi_chip_event_dtype)
                    module_hits['event_number'] = hits['event_number']
                    module_hits['trigger_number'] = hits['trigger_number']
                    module_hits['trigger_time_stamp'] = hits['trigger_time_stamp']
                    module_hits['relative_BCID'] = hits['relative_BCID']
                    module_hits['column'] = hits['column']
                    module_hits['row'] = hits['row']
                    module_hits['tot'] = hits['tot']
                    module_hits['moduleID'] = moduleID
                    
                    self.hits[moduleID] = np.r_[self.hits[moduleID],module_hits]
#                     self.logger.info("module_%s recieved %s hits" %(moduleID,self.hits[moduleID].shape))



    def run(self):
        ''' create workers upon start and collect data after EoS'''
        
#         self.multimodule_hits = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype))
        self.jobs = []
        self.pipes = []
        
#         for module in range(self.n_modules): # TODO: fast enough? only possible to check with 8 FEs
#             recv_end, send_end = multiprocessing.Pipe(False)
#             worker = multiprocessing.Process(target = self.module_worker, args =(self.address + self.ports[module], module, send_end))
#             worker.name = 'RecieverModule_%s' % module
#             self.jobs.append(worker)
#             self.pipes.append(recv_end)
#             worker.start()
#             print "PID of worker %s:"% module, worker.pid
            
        for module in range(self.n_modules): # TODO: fast enough? only possible to check with 8 FEs
            recv_end, send_end = multiprocessing.Pipe(False)
            worker = threading.Thread(target = self._module_worker, args =(self.address + self.ports[module], module, send_end))
            worker.name = 'RecieverModule_%s' % module
            self.jobs.append(worker)
            self.pipes.append(recv_end)
            worker.start()
        
        while not self._stop_readout.wait(0.1):
            if self.reset_multimodule_hits.is_set(): # set by SoS_reset upon reception of SoS signal
                self.multimodule_hits = np.ascontiguousarray(np.empty(shape=(0,),dtype = self.multi_chip_event_dtype))
                self.reset_multimodule_hits.clear()
#                 self.logger.info('Main array has been reset')
            if self.EoS_flag.is_set(): # EoS_flag is set in run_control after reception of EoS command 
                #TODO: check building of common event from all modules
                with self.reset_lock:
                    start = datetime.datetime.now()
                    if not self.EoS_data_flag.is_set(): # SoS_data_flag is set after all events are sent to dispatcher
                        for pipe in self.pipes:
                            hit_array = pipe.recv()
                            self.multimodule_hits = np.hstack((self.multimodule_hits, hit_array))
                        
                        self.SoS_flag.clear()    
                        self.multimodule_hits.sort(order='event_number')
                        event_numbers , indices = np.unique(self.multimodule_hits['event_number'],return_index = True)
                        print "time for sorting:", datetime.datetime.now() - start
                        n_events = indices.shape[0]
                        print "nevents multiarray" , n_events , "shape multiarray" , self.multimodule_hits.shape
                        with open("./RUN_%s/%s.txt" % (self.run_number.value, self.file_date.value),'a+') as spill_file:
                            for i, index in enumerate(indices):
                                if i == n_events-1:
                                    event = self.multimodule_hits[index:]
                                else:
                                    event = self.multimodule_hits[index:indices[i+1]]
    #                             print index, indices[i+1]
    #                             print event 
                                channelID = np.bitwise_or(event['row']<<7,event['column'],order='C',dtype='uint16')
                                hit_data = np.bitwise_or(np.bitwise_or(event['tot']<<4,event['moduleID'])<<8,
                                                         np.bitwise_or(0<<4,event['relative_BCID']),order='C',dtype='uint16')
                                self.ch_hit_data = np.empty(channelID.shape[0], dtype = Hit)
                                self.ch_hit_data["channelID"] = channelID
                                self.ch_hit_data["hit_data"] = hit_data
                                # each event needs a frame header
                                self.data_header = np.empty(shape=(1,), dtype= FrHeader)
                                self.data_header['size'] = channelID.nbytes + 16
                                self.data_header['partID'] = self.partitionID
                                self.data_header['cycleID'] = self.cycle_ID.value
                                self.data_header['frameTime'] = event[0]['trigger_time_stamp']
                                self.data_header['timeExtent'] = 15
                                self.data_header['flags'] = 0
                                self.ch.send_data(self.RAW_data_tag, self.data_header, self.ch_hit_data)
#                                 np.savetxt(spill_file, self.data_header) #self.data_header)
#                                 np.savetxt(spill_file, self.ch_hit_data)# self.ch_hit_data)
                        self.EoS_data_flag.set()
                        print "time needed for %s events : %s without saving" %(self.multimodule_hits['event_number'][-1],(datetime.datetime.now()-start))
#                     self.total_events += event_indices.shape[0]
                
                        
    def stop(self):
        self.logger.info('Stopping converter')
        self._stop_readout.set()
        for job in self.jobs:
            job.join()
        self.logger.info('All Workers Joined')


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

