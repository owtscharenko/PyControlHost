import __future__
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

import control_host_coms as ch
from pybar import *
import ship_data_converter

punctuation = '!,.:;?'


class RunAborted(Exception):
    pass


class RunStopped(Exception):
    pass

class run_control():
    
    def __init__(self,dispatcher_addr,converter_addr, configuration, partitionID):       
        self._cancel_functions = None
        self.connect_cancel(["abort"])
        self.status = 0
        self.enabled = True
        self.abort_run = Event()
        self.stop_run = Event()
        self.socket_addr = dispatcher_addr
        self.converter_socket_addr = converter_addr
        self.pybar_conf = configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
        self.commands = {'SoR','EoR','SoS','EoS','Enable','Disable','Stop'}
        self.partitionID = int(partitionID,16) # '0X0802' from 0800 to 0802 how to get this from scan instance?
        self.DetName = 'Pixels' + partitionID[5:] + '_LocDaq_' + partitionID[2:]
        self.ch_com = ch_communicator()
        self.connect_CH(self.socket_addr,self.DetName)
        self.cmd = []
        self.command = 'none'

        
    def _signal_handler(self, signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # setting default handler... pressing Ctrl-C a second time will kill application
        self.handle_cancel(msg='Pressed Ctrl-C')    
    
    
    def connect_cancel(self, functions):
        '''Run given functions when a run is cancelled.'''
        self._cancel_functions = []
        for func in functions:
            if isinstance(func, basestring) and hasattr(self, func) and callable(getattr(self, func)):
                self._cancel_functions.append(getattr(self, func))
            elif callable(func):
                self._cancel_functions.append(func)
            else:
                raise ValueError("Unknown function %s" % str(func))
    
    
    def abort(self, msg=None):
        '''Aborting a run. Control for loops. Immediate stop/abort.
        The implementation should stop a run ASAP when this event is set. The run is considered incomplete.
        '''
        if not self.abort_run.is_set():
            if msg:
                logging.error('%s%s Aborting run...', msg, ('' if msg[-1] in punctuation else '.'))
            else:
                logging.error('Aborting run...')
        self.abort_run.set()
        self.stop_run.set()  # set stop_run in case abort_run event is not used    
        
        
    def handle_cancel(self, **kwargs):
        '''Cancelling a run.
        '''
        for func in self._cancel_functions:
            f_args = getargspec(func)[0]
            f_kwargs = {key: kwargs[key] for key in f_args if key in kwargs}
            func(**f_kwargs)
    
    
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)
    
    
    def connect_CH(self,socket_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        self.ch_com.init_link(socket_addr, subscriber=None)
        if self.ch_com.status < 0:
            logging.error('Could not connect to host %s' % socket_addr)
        elif self.ch_com.status >= 0 :
            self.ch_com.subscribe(DetName)
        if self.ch_com.status < 0:
            logging.error('Could not subscribe with name=%s to host %s' % (DetName, socket_addr))
        elif self.ch_com.status >= 0 :
            self.ch_com.send_me_always()
        
    
    def run(self):
        ''' 
        main loop for reception and execution of commands.
        starts threads corresponding to the command recieved.
        '''
        
#         if current_thread().name == 'MainThread':
#             logging.info('Press Ctrl-C to stop run')
#             signal_handler = self._signal_handler
#             signal.signal(signal.SIGINT, signal_handler)
#             signal.signal(signal.SIGTERM, signal_handler)
        try:
            converter = ship_data_converter.DataConverter(self.converter_socket_addr, self.partitionID)
            converter.name = 'DataConverter'
            converter.daemon = True
            runmngr = RunManager(self.pybar_conf)
            runmngr.daemon = True
            joinmngr = None
            while True:
                if self.status >=0 and self.ch_com.status >=0 :
                    self.status = ch.get_head_wait('DAQCMD', self.ch_com.cmdsize)
                    if self.status >=0 and self.ch_com.status >=0 :
                        logging.info('Recieved header')
                        self.cmd = self.ch_com.get_cmd() # recieved command contains command word [0] and additional info [1]... different for each case
                        if len(self.cmd) > 1:
                            self.command = self.cmd[0]
                        elif len(self.cmd)==0 or len(self.cmd) ==1 :
                            self.command = self.cmd
                        if self.command in self.commands and self.ch_com.status >=0:
                            self.ch_com.send_ack(tag='DAQACK',msg = '%s %04X %s' %(self.command, self.partitionID, self.socket_addr)) # acknowledge command
                            if self.command == 'Enable': # enable detector partition
                                self.enabled = True
                            elif self.command == 'Disable': #disable detector partition
                                self.enabled = False
                            elif self.command == 'SoR': # Start new pyBAR ExtTriggerScanShiP.
                                if len(self.cmd) > 1:
                                    run_number = self.cmd[1]
                                else:
                                    run_number = None
                                converter.reset(cycleID=self.cycle_ID(), msg = 'SoR command, resetting DataConverter')
                                if not converter.is_alive():
                                    converter.start()
                                #send special SoR header
                                header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=self.cycle_ID(), trigger_timestamp=0xFF005C01, bcids=0, flag=0)
                                self.ch_com.send_data_numpy(tag = 'RAW_0802', header, hits=0)
                                #start pybar trigger scan
                                joinmngr = runmngr.run_run(ExtTriggerScanSHiP, run_conf={'scan_timeout': 86400, 'max_triggers':0, 'ship_run_number': run_number}, use_thread=True) # TODO: how to get run number to pyBAR ?
                                
                                self.ch_com.send_done('SoR',self.partitionID, self.status ) 
                            elif self.command == 'EoR': # stop existing pyBAR ExtTriggerScanShiP
                                logging.info('Recieved EoR command')
                                if runmngr.current_run.__class__.__name__ == 'ExtTriggerScanSHiP' and joinmngr != None:
                                    runmngr.current_run.stop(msg='ExtTriggerScanSHiP') # TODO: check how to properly stop pyBAR RunManager
                                    joinmngr(timeout = 0.01)
                                else:
                                    logging.error('Recieved EoR command, but no ExtTriggerScanSHiP running')
                                if converter.is_alive():
                                    converter.reset(cycleID = self.cycle_ID(), msg='EoR command, resetting DataConverter') # reset interpreter and event counter
                                    logging.info('DataConverter has been reset')
                                else:
                                    logging.error('Recieved EoR command to reset converter, but no converter running')
                                # send special EoR header
                                header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=self.cycle_ID(), trigger_timestamp=0xFF005C02, bcids=0, flag=0)
                                self.ch_com.send_data_numpy(tag = 'RAW_0802', header, hits=0)
                                self.ch_com.send_done('EoR',self.partitionID, self.status)
                            elif self.command == 'SoS': # new spill. trigger counter will be reset by hardware signal. The software command triggers an empty header
                                if len(self.cmd) > 1:
                                    cycleID = int(self.cmd[1])
                                else:
                                    cycleID = 0 #self.cycle_ID()
                                logging.info('Recieved SoS header, cycleID = %s' % cycleID)
        #                         if central_cycleID != self.cycle_ID():
                                header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=cycleID, trigger_timestamp=0xFF005C03, bcids=0, flag=0)
                                self.ch_com.send_data_numpy(tag = 'RAW_0802', header, hits=0)
                                self.ch_com.send_done('SoS',self.partitionID, converter.total_events) # TODO: make sure send done is called after last event is converted
                            elif self.command == 'EoS': # trigger EoS header, sent after last event
                                logging.info('recieved EoS, local cycleID:%s' % self.cycle_ID())
                                header = ship_data_converter.build_header(n_hits=0, partitionID=self.partitionID, cycleID=self.cycleID(), trigger_timestamp=0xFF005C04, bcids=0, flag=0) # TODO: send EoS header after last event from spill
                                self.ch_com.send_data_numpy(tag = 'RAW_0802', header, hits=0)
                                self.ch_com.send_done('EoS', self.partitionID, self.status)
                            elif self.command == 'Stop':
                                logging.info('Recieved Stop! Leaving loop, aborting all functions')
                                break
                        else:
                            logging.error('Command=%s could not be identified' % self.cmd)
                    elif self.status < 0 :
                        logging.error('Header could not be recieved')
                else:
                    self.status = -1
                    raise RuntimeError('Undefined state')
            converter.stop()
            runmngr.abort()
            logging.error('Loop exited')
        except Exception as e:
            logging.error('Exception, terminating')
            print e.__class__.__name__ + ": " + str(e)
            



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
    
    
    def send_data(self, tag, header, hits):
#         logging.info('sending data package with %s byte' % length)
        self.status = ch.send_fulldata_numpy(tag, header, hits) # TODO: case only header, no hits. 
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
    
    usage = "Usage: %prog dispatcher_addr converter_addr configuration.yaml partitionID"
    description = "dispatcher_addr: Remote address of the sender (default: 127.0.0.1). converter_addr: local address of the SHiP data converter (default: tcp://127.0.0.1:5678). configuration: absolute path of pyBAR configuration.yaml. partitionID : ID of detector for dispatcher (default: 0X0802)"
    parser = OptionParser(usage, description=description)
    options, args = parser.parse_args()
    if len(args) == 1 and not args[0].find('configuration')==-1 :
        dispatcher_addr = '127.0.0.1'
        converter_addr = 'tcp://127.0.0.1:5678'
        configuration = args[0]
        partitionID = '0X0802'
    elif len(args) == 4:
        dispatcher_addr = args[0]
        converter_addr = args[1]
        configuration = args[2]
        partitionID = args[3]
        
    else:
        parser.error("incorrect number of arguments")
    
    rec = run_control(dispatcher_addr,
                      converter_addr,
                      configuration,
                      partitionID)
    
    rec.run()

    
#     rec = threading.Thread(name = 'reciever', target = run_control('127.0.0.1', '/home/niko/git/pyBAR/pybar/configuration.yaml'))
#     rec.start()
    
    
        
    
    