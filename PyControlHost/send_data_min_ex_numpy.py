import sys, os
import time
import signal
import logging
import datetime
import threading
import multiprocessing
from multiprocessing import Event
from optparse import OptionParser
from inspect import getmembers, isclass, getargspec
from timeit import Timer as T
import cython
import random

import zmq
import numpy as np


from ctypes import *
import ctypes
# import control_host_coms as ch
from pybar import *
from numba import njit, jit
import control_host_coms as ch


class ch_communicator():
    
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.status = 0
#         self.connect(socket_addr,subscriber = 'Pixels') # tags identify the type of information which is transmitted: DAQCMD, DAQACK DAQDONE
        self.cmd = np.char.asarray(['0']*127, order={'C'}) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = np.ascontiguousarray(np.int8(self.cmd.nbytes)) # maximum size of numpy array in bytes. Necessary for cmd reception, ControlHost needs to know the max. cmd size.
        self.ch = CDLL('/home/niko/git/ControlHost/shared/libconthost_shared.so')
        
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)    
        
    def init_link(self, socket_addr, subscriber):
        if socket_addr == None:
            socket_addr = '127.0.0.1'
        self.status = ch.init_disp_2way(socket_addr,'w dummy','a DAQCMD')
#         self.status = self.ch.init_2disp_link(socket_addr,'w dummy','a DAQCMD')
        if self.status < 0:
            logging.error('Connection to %s failed\n status = %s' % (socket_addr, self.status))
        elif self.status >= 0:
            logging.info('Connected to %s' % socket_addr)
    
    
    def subscribe(self,DetName):
        self.status = ch.subscribe(DetName)
        if self.status >= 0: 
            logging.info('Subscribed to Host with name=%s'%DetName)
        elif self.status < 0:
            logging.error('Could not subscribe to host')
        
    def send_me_always(self):
        self.status = ch.accept_at_all_times()
        if self.status >= 0:
            logging.info('Send me always activated')
        if self.status < 0:
            logging.error('Send me always subscription was declined')
    
    def get_cmd(self):
        self.status , cmd = ch.rec_cmd()
        if self.status < 0:
            logging.warning('Command could not be recieved')
        elif self.status >= 0 :
            logging.info('Recieved command:%s' % cmd)
        return cmd.split(' ')
    
    
    def get_data(self,tag):
        self.status = self.ch.check_head(tag,self.cmdsize)
        if self.status < 0:
            logging.warning('Message head is broken')
        elif self.status >= 0:
            self.status, data = self.ch.rec_data(self.cmd,self.cmdsize)
            logging.info('Recieved command: %s' % data)
            if self.status < 0:
                logging.warning('Command is broken')
        return data
    
    
    def send_data(self, data):
#         if  type(data) is np.ndarray:
#             length = data.shape[0]*2
#             data_out = data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
#             print data_out
#         else:
        out_data = c_int* data
        length = ctypes.sizeof(out_data)
#         logging.info('sending data package with %s byte' % length)
        self.status = self.ch.put_fulldata('RAW_0802', out_data, c_int(length))
        if self.status < 0:
            logging.error('Sending package failed')
        
    def send_data_numpy(self,header,hits):
        if  isinstance(hits,np.ndarray):
            ch.send_fulldata_numpy('RAW_0802', header, hits)
        else:
            ch.send_header_numpy('RAW_0802', header)
        
        
    def send_ack(self,tag,msg):
        self.status = ch.send_fullstring(tag, msg)
        if self.status < 0:
            logging.error('Error during acknowledge')
        elif self.status >=0:
            logging.info('Acknowledged command=%s with tag=%s' % (msg,tag))
            
            
    def send_done(self,cmd, partitionID, status):
        self.status= ch.send_fullstring('DAQDONE', '%s %04X %s' %(cmd, partitionID, status))
        if self.status < 0:
            logging.error('Could not send DAQDONE')
        elif self.status >= 0:
            logging.info('DAQDONE msg sent')


class FrHeader(ctypes.Structure):
    _fields_=[("size",   c_ushort),
              ("partID", c_ushort),
              ("cycleID",c_int),
              ("frameTime", c_int),
              ("timeExtent", c_ushort),
              ("flags",  c_ushort)]
   
class Hit(ctypes.Structure):
    _fields_=[("channelID", c_ushort),
              ("hit_data", c_ushort)]
    


if __name__ == '__main__':
    
    ch_com = ch_communicator()
    ch_com.init_link('127.0.0.1', subscriber=None)
    ch_com.subscribe('Pixels2_LocDaq_0802')
    ch_com.send_me_always()
    cycle = ch_com.cycle_ID()

    avg_nhits = 0
    nevents = 50000
#     channelIDs = np.array([41296,41295,41294,0xffff],dtype=[("channelID", c_ushort)]) #80<<9 ^ 336 = 0xa150
#     add_datawords = np.array([49672,49571,49670,0xfffe],dtype=[("hit_data", c_ushort)]) # 12<<12 ^ 2<<8 ^ 0<<4 ^ 8 = 0xc208
#     channelIDs = [41296,41295,41294,0xffff] #80<<9 ^ 336 = 0xa150
#     add_datawords = [49672,49571,49670,0xfffe] # 12<<12 ^ 2<<8 ^ 0<<4 ^ 8 = 0xc208
    start = time.time()
   
    for evt in range(nevents):
        head = np.empty(shape=(1,),dtype = FrHeader)
        nhits = random.randint(100,300)
        head['size']= nhits*4+16
        head['partID'] = 0x0802
        head['cycleID'] = cycle
        head['frameTime'] = 0xFF005C01
        head['timeExtent'] = 15
        head['flags'] = 0
        
        channelIDs = np.random.randint(low=0,high=2**16,size = nhits, dtype = np.uint16)
        add_datawords = np.random.randint(low=0,high=2**16,size = nhits, dtype = np.uint16)
        hits = np.empty(nhits,dtype= Hit,order = 'c')
        hits["channelID"] = channelIDs
        hits["hit_data"] = add_datawords
        
        avg_nhits = avg_nhits + nhits
        
#     frame = [len(channelIDs)*4+16,0x0802,cycle,0xFF005C01,15,0]
#     frame.extend(hits.flatten().tolist())
    
        ch_com.send_data_conc(head,hits)
    logging.info("total time: %s" % (time.time() - start))
    logging.info("average number of hits : %s" % (avg_nhits/nevents))
#     print "frame size: %s Byte" % head['size']

    