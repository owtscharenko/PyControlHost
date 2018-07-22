import logging
from ctypes import Structure, c_ushort, c_int, c_uint
import numpy as np
from PyControlHost import control_host_coms as ch
import multiprocessing
import signal


class FrHeader(Structure):
    _fields_=[("size",   c_ushort),
              ("partID", c_ushort),
              ("cycleID",c_uint),
              ("frameTime", c_uint),
              ("timeExtent", c_ushort),
              ("flags",  c_ushort)]
   
   
class Hit(Structure):
    _fields_=[("channelID", c_ushort),
              ("hit_data", c_ushort)]


class CHostReceiveHeader(multiprocessing.Process):

    def __init__(self,send_end):
        multiprocessing.Process.__init__(self)
        self.logger = logging.getLogger('CHostReceiveHeader')
        self.cmd = np.char.asarray(['0']*127, order={'C'})
        self.cmdsize = np.ascontiguousarray(np.int8(self.cmd.nbytes))
        self._stop_readout = multiprocessing.Event()
        self.status = multiprocessing.Value('i',0)
        self.head_received = multiprocessing.Event()
        self.send_end = send_end

    
    def run(self): # TODO: is not killed by ctrl+c in main loop
        
        while not self._stop_readout.wait(0.01):
            self.status.value = ch.get_head_wait('DAQCMD', self.cmdsize)
            if self.status.value >=0:
                self.head_received.set()
                self.status.value , cmd = ch.rec_cmd()
                if self.status < 0:
                    self.logger.warning('Command could not be recieved')
                elif self.status.value >= 0 :
                    self.logger.info('Recieved command: %s' % cmd)
                self.send_end.send(cmd.split(' '))


    def stop(self):
        self._stop_readout.set()


class CHostInterface():
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.status = 0
#         self.connect(socket_addr,subscriber = 'Pixels') # tags identify the type of information which is transmitted: DAQCMD, DAQACK DAQDONE
        self.cmd = np.char.asarray(['0']*127, order={'C'}) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = np.ascontiguousarray(np.int8(self.cmd.nbytes)) # maximum size of numpy array in bytes. Necessary for cmd reception, ControlHost needs to know the max. cmd size.

        
    def init_link(self, socket_addr, subscriber):
        if socket_addr == None:
            socket_addr = '127.0.0.1'
#         self.status = ch.init_disp(socket_addr, 'a ' + subscriber)
        self.status = ch.init_disp_2way(socket_addr,'w dummy','a DAQCMD')
        if self.status < 0:
            self.logger.error('Connection to %s failed\n status = %s' % (socket_addr, self.status))
        elif self.status >= 0:
            self.logger.info('Connected to %s' % socket_addr)
    
    
    def subscribe(self,DetName):
        self.status = ch.subscribe(DetName)
        if self.status >= 0: 
            self.logger.info('Subscribed to Host with name = %s'%DetName)
        elif self.status < 0:
            self.logger.error('Could not subscribe to host')
        
    def send_me_always(self):
        self.status = ch.accept_at_all_times()
        if self.status >= 0:
            self.logger.info('Send me always activated')
        if self.status < 0:
            self.logger.error('Send me always subscription was declined')
    
    def connect_CH(self,socket_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        self.init_link(socket_addr, subscriber=None)
        if self.status < 0:
            self.logger.error('Could not connect to host %s' % socket_addr)
        elif self.status >= 0 :
            self.subscribe(DetName)
        if self.status < 0:
            self.logger.error('Could not subscribe with name=%s to host %s' % (DetName, socket_addr))
        elif self.status >= 0 :
            self.send_me_always()
    
    
    def get_cmd(self):
        self.status , cmd = ch.rec_cmd()
        if self.status < 0:
            self.logger.warning('Command could not be recieved')
        elif self.status >= 0 :
            self.logger.info('Recieved command:%s' % cmd)
        return cmd.split(' ')
    
    
    def get_data(self,tag):
        self.status = ch.get_head(tag,self.cmdsize)
        if self.status < 0:
            self.logger.warning('Message head is broken')
        elif self.status >= 0:
            self.status, data = ch.rec_data(self.cmd,self.cmdsize)
            self.logger.info('Recieved command: %s' % data)
            if self.status < 0:
                self.logger.warning('Command is broken')
        return data
    
    def get_head(self):
        self.status = ch.get_head_wait('DAQCMD', self.cmdsize)
        return self.status
    
#     def send_data(self, tag, header, hits):
# #         self.logger.info('sending data package with %s byte' % length)
#         self.status = ch.send_fulldata_numpy(tag, header, hits)
#         if self.status < 0:
#             self.logger.error('Sending package failed')
            
    def send_data(self, tag, header, hits):
        if isinstance(hits,np.ndarray):
            self.status = ch.send_fulldata_numpy(tag, header, hits)
        else:
            self.status = ch.send_header_numpy(tag, header)
        if self.status < 0:
            self.logger.error('Sending package failed')
        
    def send_ack(self,tag,msg):
        self.status = ch.send_fullstring(tag, msg)
        if self.status < 0:
            self.logger.error('Error during acknowledge')
        elif self.status >=0:
            self.logger.info('Acknowledged command = %s with tag = %s' % (msg,tag))
            
            
    def send_done(self,cmd, partitionID, status):
        self.status= ch.send_fullstring('DAQDONE', '%s %04X %s' %(cmd, partitionID, status))
        if self.status < 0:
            self.logger.error('Could not send DAQDONE')
        elif self.status >= 0:
            self.logger.info('DAQDONE sent, msg = %s'% cmd )

    
if __name__ == '__main__':
    pass
