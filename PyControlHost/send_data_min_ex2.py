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

import zmq
import numpy as np

from ctypes import *
import ctypes
# import control_host_coms as ch
from pybar import *
from numba import njit, jit


class ch_communicator():
    
    

    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.status = 0
#         self.connect(socket_addr,subscriber = 'Pixels') # tags identify the type of information which is transmitted: DAQCMD, DAQACK DAQDONE
        self.cmd = np.char.asarray(['0']*127, order={'C'}) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = np.ascontiguousarray(np.int8(self.cmd.nbytes)) # maximum size of numpy array in bytes. Necessary for cmd reception, ControlHost needs to know the max. cmd size.
        self.ch = CDLL('/home/niko/git/ControlHost1-1/shared/libconthost_shared.so')
        
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)    
        
    def init_link(self, socket_addr, subscriber):
        if socket_addr == None:
            socket_addr = '127.0.0.1'
#         self.status = ch.init_disp(socket_addr, 'a ' + subscriber)
        self.status = self.ch.init_2disp_link(socket_addr,'w dummy','a DAQCMD')
        if self.status < 0:
            logging.error('Connection to %s failed\n status = %s' % (socket_addr, self.status))
        elif self.status >= 0:
            logging.info('Connected to %s' % socket_addr)
    
    
    def subscribe(self,DetName):
        self.status = self.ch.my_id(DetName)
        if self.status >= 0: 
            logging.info('Subscribed to Host with name=%s'%DetName)
        elif self.status < 0:
            logging.error('Could not subscribe to host')
        
    def send_me_always(self):
        self.status = self.ch.send_me_always()
        if self.status >= 0:
            logging.info('Send me always activated')
        if self.status < 0:
            logging.error('Send me always subscription was declined')
    
    def get_cmd(self):
        self.status , cmd = self.ch.get_string()
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
        length = ctypes.sizeof(data)
#         logging.info('sending data package with %s byte' % length)
        self.status = self.ch.put_fulldata('RAW_0802', pointer(data), c_int(length))
        if self.status < 0:
            logging.error('Sending package failed')
        
        
    def send_ack(self,tag,msg):
        self.status = self.ch.put_fullstring(tag, msg)
        if self.status < 0:
            logging.error('Error during acknowledge')
        elif self.status >=0:
            logging.info('Acknowledged command=%s with tag=%s' % (msg,tag))
            
            
    def send_done(self,cmd, partitionID, status):
        self.status= self.ch.put_fullstring('DAQDONE', '%s %04X %s' %(cmd, partitionID, status))
        if self.status < 0:
            logging.error('Could not send DAQDONE')
        elif self.status >= 0:
            logging.info('DAQDONE msg sent')


class FrHeader(ctypes.Structure):
    _fields_=[("size",   c_ushort),
              ("partId", c_ushort),
              ("cycleId",c_int),
              ("frTime", c_int),
              ("timeEx", c_ushort),
              ("flags",  c_ushort)]
   
class Hit(ctypes.Structure):
    _fields_=[("channelID", c_ushort),
              ("hit_data", c_ushort)]
    
class EvtFrame(ctypes.Structure):
    _fields_ = [("size", c_ushort),
            ("partId", c_ushort),
            ("cycleId", c_int),
            ("frTime", c_int),
            ("timeEx", c_ushort),
            ("flags",  c_ushort),
            ("hits",   Hit * 500)]


def make_frame(nhits):
    class Frame(ctypes.Structure):
        _fields_ = [("size", c_ushort),
                ("partId", c_ushort),
                ("cycleId", c_int),
                ("frTime", c_int),
                ("timeEx", c_ushort),
                ("flags",  c_ushort),
                ("hits",   Hit * nhits)]
        #[("head", FrHeader), ("hits", Hit * nhits)] 
    return Frame()


def fill_frame(evt, channelID, add_dataword):
    evt = evt.hits
    for hit in range(len(channelID)):
        evt[hit].channelID = channelID[hit]
        evt[hit].hit_data = add_dataword[hit]

if __name__ == '__main__':
    
    ch_com = ch_communicator()
    ch_com.init_link('127.0.0.1', subscriber=None)
    ch_com.subscribe('Pixels2_LocDaq_0802')
    ch_com.send_me_always()
    cycle = ch_com.cycle_ID()
    print "cycleID:" , cycle, hex(cycle)
    
    channelID = [41296,41295,41294,0xffff] #80<<9 ^ 336 = 0xa150
    add_dataword = [49672,49571,49670,0xfffe] # 12<<12 ^ 2<<8 ^ 0<<4 ^ 8 = 0xc208
#     evt = make_frame(len(channelID))
#     print T(lambda:make_frame(len(channelID))).timeit(50000)
    evt = EvtFrame()
    fill_frame(evt,channelID,add_dataword)
    evt.head.size= len(channelID)*4+16
    evt.head.partId = 0x0802
    evt.head.cycleId = cycle
    evt.head.frTime = 0xFF005C01
    evt.head.timeEx = 15
    evt.head.flags = 0
    fill_frame(evt,channelID,add_dataword)
    
    ch_com.send_data(evt)
    print "frame size: %s byte" % evt.head.size

    