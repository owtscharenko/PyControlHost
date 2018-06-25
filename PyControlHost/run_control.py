import __future__
import sys, os
import time
import signal
import logging
import datetime
import multiprocessing
from multiprocessing import Event, Process
from optparse import OptionParser
from inspect import getmembers, isclass, getargspec

import zmq
import numpy as np

import control_host_coms as ch
from pybar import *
import ship_data_converter
from  ControlHost import ch_communicator, FrHeader
from SHiP_RunManager import SHiP_RunManager

# punctuation = '!,.:;?'


# class RunAborted(Exception):
#     pass
# 
# 
# class RunStopped(Exception):
#     pass

class run_control(object):
    
    def __init__(self,dispatcher_addr,converter_addr, configuration, partitionID):    
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')   
#         self._cancel_functions = None
#         self.connect_cancel(["abort"])
        self.status = 0
        self.enabled = True
        self.abort_run = Event()
        self.stop_run = Event()
        self.disp_addr = dispatcher_addr
        self.converter_socket_addr = converter_addr
        self.pybar_conf = configuration
        
        self.commands = {'SoR','EoR','SoS','EoS','Enable','Disable','Stop'}
        self.partitionID = int(partitionID,16) # '0X0802' from 0800 to 0802 how to get this from scan instance?
        self.DetName = 'Pixels' + partitionID[5:] + '_LocDaq_' + partitionID[2:]
        self.ch_com = ch_communicator()
        self.connect_CH(self.disp_addr,self.DetName)
        self.cmd = []
        self.command = 'none'
        self.special_header = np.empty(shape=(1,), dtype= FrHeader)
        self.special_header['size'] = 16
        self.special_header['partID'] = self.partitionID
        self.special_header['timeExtent'] = 0
        self.special_header['flags'] = 0
        
#     def _signal_handler(self, signum, frame):
#         signal.signal(signal.SIGINT, signal.SIG_DFL)  # setting default handler... pressing Ctrl-C a second time will kill application
#         self.handle_cancel(msg='Pressed Ctrl-C')    
    
    
#     def connect_cancel(self, functions):
#         '''Run given functions when a run is cancelled.'''
#         self._cancel_functions = []
#         for func in functions:
#             if isinstance(func, basestring) and hasattr(self, func) and callable(getattr(self, func)):
#                 self._cancel_functions.append(getattr(self, func))
#             elif callable(func):
#                 self._cancel_functions.append(func)
#             else:
#                 raise ValueError("Unknown function %s" % str(func))
    
    
#     def abort(self, msg=None):
#         '''Aborting a run. Control for loops. Immediate stop/abort.
#         The implementation should stop a run ASAP when this event is set. The run is considered incomplete.
#         '''
#         if not self.abort_run.is_set():
#             if msg:
#                 logging.error('%s%s Aborting run...', msg, ('' if msg[-1] in punctuation else '.'))
#             else:
#                 logging.error('Aborting run...')
#         self.abort_run.set()
#         self.stop_run.set()  # set stop_run in case abort_run event is not used    
        
        
#     def handle_cancel(self, **kwargs):
#         '''Cancelling a run.
#         '''
#         for func in self._cancel_functions:
#             f_args = getargspec(func)[0]
#             f_kwargs = {key: kwargs[key] for key in f_args if key in kwargs}
#             func(**f_kwargs)
    
    
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)
    
    
    def connect_CH(self,disp_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        self.ch_com.init_link(disp_addr, subscriber=None)
        if self.ch_com.status < 0:
            logging.error('Could not connect to host %s' % disp_addr)
        elif self.ch_com.status >= 0 :
            self.ch_com.subscribe(DetName)
        if self.ch_com.status < 0:
            logging.error('Could not subscribe with name=%s to host %s' % (DetName, disp_addr))
        elif self.ch_com.status >= 0 :
            self.ch_com.send_me_always()
        
    
    def receive(self):
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
            converter = ship_data_converter.DataConverter(pybar_addr = self.converter_socket_addr, partitionID = self.partitionID)
            converter.name = 'DataConverter'
            converter.daemon = True
            RunManager(self.pybar_conf)
#             runmngr.daemon = True
            while True:
                if self.status >=0 and self.ch_com.status >=0 :
                    self.status = ch.get_head_wait('DAQCMD', self.ch_com.cmdsize)
                    if self.status >=0 and self.ch_com.status >=0 :
                        self.cmd = self.ch_com.get_cmd() # recieved command contains command word [0] and additional info [1]... different for each case
                        if len(self.cmd) > 1:
                            self.command = self.cmd[0]
                        elif len(self.cmd)==0 or len(self.cmd) ==1 :
                            self.command = self.cmd
                        if self.command in self.commands and self.ch_com.status >=0:
                            self.ch_com.send_ack(tag='DAQACK',msg = '%s %04X %s' %(self.command, self.partitionID, self.disp_addr)) # acknowledge command
                            if self.command == 'Enable': # enable detector partition
                                self.enabled = True
                            elif self.command == 'Disable': #disable detector partition
                                self.enabled = False
                            elif self.command == 'SoR': # Start new pyBAR ExtTriggerScanShiP.
                                if len(self.cmd) > 1:
                                    run_number = self.cmd[1]
                                else:
                                    run_number = None
                                if not os.path.exists("./RUN_%s/" % run_number):
                                    os.makedirs("./RUN_%s/" % run_number)
                                if not converter.is_alive():
                                    converter.start()
                                else: converter.reset(cycleID=self.cycle_ID(), msg = 'SoR command, resetting DataConverter')
                                converter.run_number = run_number
                                #send special SoR header
                                self.special_header['frameTime'] = 0xFF005C01
                                self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                                #start pybar trigger scan
                                scan_thread = RunManager.run_run(ThresholdScan, use_thread=True, catch_exception=True)
#                                 scan_thread = runmngr.run_run(ExtTriggerScanSHiP, run_conf={'scan_timeout': 86400, 'max_triggers':0, 
#                                                                                             'no_data_timeout':0, 'ship_run_number': run_number}, 
#                                                                                             use_thread=True) # TODO: how start pyBAR in thread?
                                scan_thread(0.1)
                                self.ch_com.send_done('SoR',self.partitionID, self.status ) 
                            elif self.command == 'EoR': # stop existing pyBAR ExtTriggerScanShiP
                                logging.info('Recieved EoR command')
                                if RunManager.current_run.__class__.__name__ == 'ThresholdScan':
#                                     scan_thread(timeout = 0.01)
                                    RunManager.current_run.stop(msg='ExtTriggerScanSHiP') # TODO: check how to properly stop pyBAR RunManager
                                else:
                                    logging.error('Recieved EoR command, but no ExtTriggerScanSHiP running')
                                if converter.is_alive():
                                    converter.reset(cycleID = self.cycle_ID(), msg='EoR command, resetting DataConverter') # reset interpreter and event counter
                                else:
                                    logging.error('Recieved EoR command to reset converter, but no converter running')
                                # send special EoR header
                                self.special_header['frameTime'] = 0xFF005C02
                                self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                                self.ch_com.send_done('EoR',self.partitionID, self.status)
                            elif self.command == 'SoS': # new spill. trigger counter will be reset by hardware signal. The software command triggers an empty header
                                if len(self.cmd) > 1:
                                    cycleID = np.uint64(self.cmd[1])
                                else:
                                    cycleID = 0 #self.cycle_ID()
                                converter.cycle_ID = cycleID
                                logging.info('Recieved SoS header, cycleID = %s' % cycleID)
                                self.special_header['frameTime'] = 0xFF005C03
                                self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                                self.ch_com.send_done('SoS',self.partitionID, converter.total_events) # TODO: make sure send done is called after last event is converted
                            elif self.command == 'EoS': # trigger EoS header, sent after last event
                                logging.info('recieved EoS, local cycleID:%s' % self.cycle_ID())
                                self.special_header['frameTime'] = 0xFF005C04 # TODO: send EoS header after last event from spill
                                self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
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
                scan_thread(0.01)
            converter.stop()
            print "hello"
            RunManager.abort()
            logging.error('Loop exited')
        except Exception as e:
            logging.error('Exception, terminating')
            print e.__class__.__name__ + ": " + str(e)
            


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
    
    rec.receive()

    
#     rec = threading.Thread(name = 'reciever', target = run_control('127.0.0.1', '/home/niko/git/pyBAR/pybar/configuration.yaml'))
#     rec.start()
    
    
        
    
    