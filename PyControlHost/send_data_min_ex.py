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

import zmq
import numpy as np

from ctypes import *
import control_host_coms as ch
from pybar import *


class ch_communicator():
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.status = 0
#         self.connect(socket_addr,subscriber = 'Pixels') # tags identify the type of information which is transmitted: DAQCMD, DAQACK DAQDONE
        self.cmd = np.char.asarray(['0']*127, order={'C'}) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = np.ascontiguousarray(np.int8(self.cmd.nbytes)) # maximum size of numpy array in bytes. Necessary for cmd reception, ControlHost needs to know the max. cmd size.

        
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)    
        
    def init_link(self, socket_addr, subscriber):
        if socket_addr == None:
            socket_addr = '127.0.0.1'
#         self.status = ch.init_disp(socket_addr, 'a ' + subscriber)
        self.status = ch.init_disp_2way(socket_addr,'w dummy','a DAQCMD')
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
        self.status = ch.check_head(tag,self.cmdsize)
        if self.status < 0:
            logging.warning('Message head is broken')
        elif self.status >= 0:
            self.status, data = ch.rec_data(self.cmd,self.cmdsize)
            logging.info('Recieved command: %s' % data)
            if self.status < 0:
                logging.warning('Command is broken')
        return data
    
    
    def send_data(self, data):
        length = 16
#         logging.info('sending data package with %s byte' % length)
        self.status = ch.send_fulldata('RAW_0800', data, length)
        if self.status < 0:
            logging.error('Sending package failed')
        
        
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



    
if __name__ == '__main__':
    
    ch_com = ch_communicator()
    ch_com.init_link('127.0.0.1', subscriber=None)
    ch_com.subscribe('Pixels2_LocDaq_0802')
    ch_com.send_me_always()
    cycle = ch_com.cycle_ID()
    print "cycleID:" , cycle, hex(cycle)
    data_header = np.ndarray(shape=(1,),order='c')
    data_header[0] = (112*4+16)<<112 ^ 2050<<96 ^ cycle<<64 ^ 0xFF005C01<<32 ^ 15<<16 ^ 0
    print "data_header:" , hex(data_header[0])
    ch_com.send_data(data_header)
     
    