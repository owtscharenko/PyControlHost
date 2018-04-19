import __future__
import sys
import time
import logging
import datetime
from threading import Event, Lock
from optparse import OptionParser

import zmq
import numpy as np
import ctypes
from ch_transmission import control_host_coms as ch


class run_control():
    
    def __init__(self):
        self.status = 0
        pass
        
    def cycle_ID(self):
        # counts in 0.2s steps from 08. April 2015
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return np.uint32((datetime.datetime.now() - start_date).total_seconds()*5) 
    
    
    

class ch_communicator():
    
    def __init__(self,socket_addr):
        self.connect(socket_addr,tag = "CMD") # tags identify the type of information which is transmitted: data, cmd, etc. ALLOWED TAGS TO BE PROVIDED
        self.cmd = np.ascontiguousarray(np.char.asarray(shape=200)) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = sys.getsizeof(self.cmd) # maximum size of command in bytes ???
        
        
    def connect(self, socket_addr, tag):
        self.status = ch.init_disp(socket_addr, "a " + tag)
        return self.status
    
    
    def get_command(self,tag):
        self.status = ch.check_head(tag,self.cmdsize)
        if self.status < 0:
            raise RuntimeError('Message head is broken, abort')
            break
        elif self.status >=0:
            self.status, command = ch.rec_data(self.cmd,self.cmdsize)
            logging.info('recieved command: %s' %command)
            if self.status < 0:
                raise RuntimeError('Command %s is broken, abort' % command)
                break
        return self.status, command
    
    
    def send_data(self, data):
        length = sys.getsizeof(data)
        self.status = ch.send_fulldata(tag = 'DATA', data, length)
        if self.status <0:
            raise RuntimeError('sending package failed')
        return self.status
        
    
    