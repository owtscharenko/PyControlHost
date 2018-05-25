import __future__
import sys
import time
import logging
import datetime
import threading
from threading import Event, Lock
from optparse import OptionParser

import zmq
import numpy as np

import control_host_coms as ch
from pybar import *
from PyControlHost import ship_data_converter



class run_control():
    
    def __init__(self,socket_addr,configuration):
        self.status = 0
        self.enabled = True
        self.socket_addr = socket_addr
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.commands = {'SoR','EoR','SoS','EoS','Enable','Disable','Stop'}
        self.partitionID = '0802' # from 0800 to 0802 how to get this from scan instance?
        self.DetName = 'Pixels' + self.partitionID[-1] + '_LocDaq_' + self.partitionID
        self.connect_CH(self.socket_addr,self.DetName)
        self.mngr = RunManager(configuration)
        
        
    def cycle_ID(self):
        # counts in 0.2s steps from 08. April 2015
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return np.uint32((datetime.datetime.now() - start_date).total_seconds()*5) 
    
    
    def connect_CH(self,socket_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        ch_communicator.init_link(socket_addr, subscriber=None)
        if ch_communicator.status >= 0:
            ch_communicator.subscribe(DetName)
        else:
            logging.error('could not subscribe with name=%s , no link to host %s' % (DetName, socket_addr))
        # TODO: implement ControlHost's send_me_always. Allows dispatcher to send commands at all times.
    
    
    def run(self):
        ''' 
        main loop for reception and execution of commands.
        calls a thread corresponding to the command recieved.
        '''
        while True:
            self.status = ch.wait_head('DAQCMD', self.cmdsize)
            if self.status >=0 and ch_communicator.status >=0 :
                logging.info('recieved header')
                cmd = ch_communicator.get_cmd(self, tag='DAQCMD') # recieved command contains command word [0] and additional info [1]... different for each case
                if len(cmd) > 1:
                    command = cmd[0]
                elif len(cmd)==0 or len(cmd) ==1 :
                    command = cmd
                if command in self.commands:
                    ch_communicator.send_ack(self, tag='DAQACK',msg = ' %s %04X %s' %(cmd, self.partitionID, self.socket_addr)) # acknowledge command
#                     ch.send_fullstring('DAQACK',' %s %s %s' %(cmd, self.partitionID, self.socket_addr))
                    if command == 'Enable': # enable detector partition
                        self.enable = True
                    elif command == 'Disable': #disable detector partition
                        self.enabled = False
                    elif command == 'SoR': # Start Run. Trigger New pybar ExtTriggerScanShiP.
                        if len(cmd) > 1:
                            run_number = cmd[1]
                        else:
                            run_number = 0
                        converter = threading.Thread(name = 'ship data converter', 
                                                     target = ship_data_converter.DataConverter, args=(self.socket_addr))
                        converter.start()
                        scan = threading.Thread(name = 'ext. trigger scan', target = self.mngr.run_run(), 
                                                args = [ExtTriggerScanShiP(run_number=run_number), None, {'scan_timeout': 86400}]) # TODO: how to set pybar run number from here ?

                        scan.start()
                        ch_communicator.send_done('SoR',self.partitionID, self.status) # TODO: send nevents instead of self.status
                    elif command == 'EoR': # stop existing pybar ExtTriggerScanShiP
#                         self.mngr.run.stop(msg='DAQ command: EoR')
                        logging.info('recieved EoR, stopping scan')
                        scan.stop()
                        ch_communicator.send_done('EoR',self.partitionID, self.status)
                    elif command == 'SoS': # new spill. trigger counter will be reset by hardware signal. The software command triggers an empty header
                        if len(cmd) > 1:
                            cycleID = cmd[1]
                        else:
                            cycleID = self.cycle_ID()
                        logging.info('recieved SoS header, cycleID = %s' % cycleID)
#                         if central_cycleID != self.cycle_ID():
                        header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=cycleID, trigger_timestamp=0, bcids=0, flag=0)
                        ch_communicator.send_data(self, header)
                        ch_communicator.send_done('SoS',self.partitionID, self.status)
                    elif command == 'EoS': # trigger EoS header, sent after last event
                        logging.info('recieved EoS, local cycleID:%s' % self.cycle_ID())
                        header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=self.cycleID, trigger_timestamp=0, bcids=0, flag=0) # TODO: send EoS header after last event from spill
                        ch_communicator.send_data(self, header)
                        ch_communicator.send_done('EoS', self.partitionID, self.status)
                else:
                    logging.error('command=%s could not be identified' % cmd)
            elif self.status < 0 or ch_communicator.status <0 :
                logging.error('header could not be recieved')
            else:
                self.status = -1
                raise RuntimeError('undefined state')
        



class ch_communicator():
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.status = 0
#         self.connect(socket_addr,subscriber = 'Pixels') # tags identify the type of information which is transmitted: DAQCMD, DAQACK DAQDONE
        self.cmd = np.ascontiguousarray(np.char.asarray(['0']*200, order={'C'})) # controlHost commands are written to this variable, size given in c 'chars'
        self.cmdsize = self.cmd.nbytes # maximum size of numpy array in bytes. Necessary for cmd reception, ControlHost needs to know the max. cmd size.

        
    def init_link(self, socket_addr, subscriber):
        if socket_addr == None:
            socket_addr = '127.0.0.1'
#         self.status = ch.init_disp(socket_addr, 'a ' + subscriber)
        self.status = ch.init_disp_2way(socket_addr,'w dummy','a DAQCMD')
        if self.status < 0:
            logging.error('connection to %s failed\n status = %s' % (socket_addr, self.status))
        elif self.status >= 0:
            logging.info('connected to %s' % socket_addr)
    
    
    def subscribe(self,DetName):
        self.status = ch.subscribe(DetName)
        if self.status >= 0: 
            logging.info('subscribed to Host with name=%s'%DetName)
        elif self.status < 0:
            logging.error('could not subscribe to host')
        
    
    def get_cmd(self,tag):
        self.status , cmd = ch.rec_cmd(tag, self.cmdsize)
        if self.status < 0:
            logging.warning('Command could not be recieved')
        elif self.status >= 0 :
            logging.info('recieved command: %s' % cmd)
        return cmd.split(' ')
    
    
    def get_data(self,tag):
        self.status = ch.check_head(tag,self.cmdsize)
        if self.status < 0:
            logging.warning('Message head is broken')
        elif self.status >= 0:
            self.status, data = ch.rec_data(self.cmd,self.cmdsize)
            logging.info('recieved command: %s' % data)
            if self.status < 0:
                logging.warning('Command is broken')
        return data
    
    
    def send_data(self, data):
        length = sys.getsizeof(data)
#         logging.info('sending data package with %s byte' % length)
        self.status = ch.send_fulldata('DATA', data, length)
        if self.status < 0:
            logging.error('sending package failed')
        
        
    def send_ack(self,tag,msg):
        self.status = ch.send_fullstring(tag, msg)
        if self.status < 0:
            logging.error('error during acknowledge')
            
            
    def send_done(self,cmd, partitionID, status):
        self.status= ch.send_fullstring('DAQDONE', '%s %04X %s' %(cmd, partitionID, status))
        if self.status < 0:
            logging.error('could not send DAQDONE')
    
if __name__ == '__main__':
#     rc = run_control(socket_addr='127.0.0.1',configuration='none')
    rec = threading.Thread(name = 'reciever', target = run_control.run(), args=('127.0.0.1', 'none'))
    rec.start()
    
    
        
    
    