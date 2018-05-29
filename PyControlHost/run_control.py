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
import ship_data_converter



class run_control():
    
    def __init__(self,dispatcher_addr,converter_addr, configuration):
        self.status = 0
        self.enabled = True
        self.socket_addr = dispatcher_addr
        self.converter_socket_addr = converter_addr
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.commands = {'SoR','EoR','SoS','EoS','Enable','Disable','Stop'}
        self.partitionID = '0X0802' # from 0800 to 0802 how to get this from scan instance?
        self.DetName = 'Pixels' + self.partitionID[2:-1] + '_LocDaq_' + self.partitionID[2:]
        self.ch_com = ch_communicator()
        self.connect_CH(self.socket_addr,self.DetName)
        self.mngr = RunManager(configuration)
        self.run()
        
        
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return np.uint32((datetime.datetime.now() - start_date).total_seconds()*5) 
    
    
    def connect_CH(self,socket_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        self.ch_com.init_link(socket_addr, subscriber=None)
        if self.ch_com.status >= 0:
            self.ch_com.subscribe(DetName)
        else:
            logging.error('could not subscribe with name=%s , no link to host %s' % (DetName, socket_addr))
        # TODO: implement ControlHost's send_me_always. Allows dispatcher to send commands at all times.
    
    
    def run(self):
        ''' 
        main loop for reception and execution of commands.
        starts threads corresponding to the command recieved.
        '''
        while True:
            if self.status >=0 and self.ch_com.status >=0 :
                self.status = ch.get_head_wait('DAQCMD', self.ch_com.cmdsize)
                if self.status >=0 and self.ch_com.status >=0 :
                    logging.info('recieved header')
                    cmd = self.ch_com.get_cmd() # recieved command contains command word [0] and additional info [1]... different for each case
                    if len(cmd) > 1:
                        command = cmd[0]
                    elif len(cmd)==0 or len(cmd) ==1 :
                        command = cmd
                    if command in self.commands and self.ch_com.status >=0:
                        self.ch_com.send_ack(tag='DAQACK',msg = '%s %04X %s' %(command, int(self.partitionID,16), self.socket_addr)) # acknowledge command
                        if command == 'Enable': # enable detector partition
                            self.enable = True
                        elif command == 'Disable': #disable detector partition
                            self.enabled = False
                        elif command == 'SoR': # Start new pyBAR ExtTriggerScanShiP.
                            if len(cmd) > 1:
                                run_number = cmd[1]
                            else:
                                run_number = None
                            converter = ship_data_converter.DataConverter(self.converter_socket_addr)
                            converter.setName('DataConverter')
#                             converter = threading.Thread(name = 'ship data converter', 
#                                                          target = ship_data_converter.DataConverter,kwargs = {"socket_addr": self.converter_socket_addr})
                            converter.start()
                            scan = threading.Thread(name = 'ExtTriggerScanShiP', target = self.mngr.run_run,
                                                    kwargs = {"run": ExtTriggerScanShiP, "run_conf" : {'scan_timeout': 86400}}) # TODO: how to set pyBAR run number from here ?
                            scan.start()
                            self.ch_com.send_done('SoR',int(self.partitionID,16), self.status ) 
                            
                        elif command == 'EoR': # stop existing pyBAR ExtTriggerScanShiP
                            logging.info('recieved EoR, stopping scan')
                            scan.join() # TODO: check how to properly stop pyBAR RunManager
                            self.ch_com.send_done('EoR',int(self.partitionID,16), self.status)
                        elif command == 'SoS': # new spill. trigger counter will be reset by hardware signal. The software command triggers an empty header
                            if len(cmd) > 1:
                                cycleID = cmd[1]
                            else:
                                cycleID = self.cycle_ID()
                            logging.info('recieved SoS header, cycleID = %s' % cycleID)
    #                         if central_cycleID != self.cycle_ID():
                            header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=cycleID, trigger_timestamp=0, bcids=0, flag=0)
                            self.ch_com.send_data(self, header)
                            self.ch_com.send_done('SoS',self.partitionID, converter.total_events) # TODO: make sure send done is called after last event is converted
                        elif command == 'EoS': # trigger EoS header, sent after last event
                            logging.info('recieved EoS, local cycleID:%s' % self.cycle_ID())
                            header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=self.cycleID, trigger_timestamp=0, bcids=0, flag=0) # TODO: send EoS header after last event from spill
                            self.ch_com.send_data(self, header)
                            self.ch_com.send_done('EoS', self.partitionID, self.status)
                        elif command == 'Stop':
                            break
                    else:
                        logging.error('command=%s could not be identified' % cmd)
                elif self.status < 0 :
                    logging.error('header could not be recieved')
            else:
                self.status = -1
                raise RuntimeError('undefined state')
            converter.join()
        



class ch_communicator():
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
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
            logging.error('connection to %s failed\n status = %s' % (socket_addr, self.status))
        elif self.status >= 0:
            logging.info('connected to %s' % socket_addr)
    
    
    def subscribe(self,DetName):
        self.status = ch.subscribe(DetName)
        if self.status >= 0: 
            logging.info('subscribed to Host with name=%s'%DetName)
        elif self.status < 0:
            logging.error('could not subscribe to host')
        
    
    def get_cmd(self):
        self.status , cmd = ch.rec_cmd()
        if self.status < 0:
            logging.warning('Command could not be recieved')
        elif self.status >= 0 :
            logging.info('recieved command:%s' % cmd)
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
        elif self.status >=0:
            logging.info('acknowledged command=%s with tag=%s' % (msg,tag))
            
            
    def send_done(self,cmd, partitionID, status):
        self.status= ch.send_fullstring('DAQDONE', '%s %04X %s' %(cmd, partitionID, status))
        if self.status < 0:
            logging.error('could not send DAQDONE')
        elif self.status >= 0:
            logging.info('DAQDONE msg sent')
    
if __name__ == '__main__':
    rec = run_control(dispatcher_addr='127.0.0.1',converter_addr = 'tcp://127.0.0.1:5678', configuration='/home/niko/git/pyBAR/pybar/configuration.yaml')
#     rec = threading.Thread(name = 'reciever', target = run_control('127.0.0.1', '/home/niko/git/pyBAR/pybar/configuration.yaml'))
#     rec.start()
    
    
        
    
    