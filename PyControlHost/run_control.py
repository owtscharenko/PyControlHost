# import __future__
import os, sys
import time
import signal
import logging
import datetime
from threading import current_thread, _MainThread, Thread
from optparse import OptionParser
from inspect import getmembers, isclass, getargspec
import multiprocessing

import numpy as np
from PyControlHost import control_host_coms as ch
# import control_host_coms as ch
from pybar import *
import ship_data_converter
from  PyControlHost.ControlHost import CHostInterface, FrHeader, CHostReceiveHeader
#from bdaq53_send_data import transfer_file
# from signal import SIGINT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
logger = logging.getLogger('RunControl')


class RunControl(object):
    
    def __init__(self,dispatcher_addr, converter_addr, ports, configuration, partitionID, bcids):    
           
        self.status = 0
        self.enabled = True
        self.disp_addr = dispatcher_addr
        self.converter_socket_addr = converter_addr
        self.ports = ports
        self.pybar_conf = configuration
        self.commands = {'SoR','EoR','SoS','EoS','Enable','Disable','Stop'}
        self.partitionID = int(partitionID,16) # '0X0802' from 0800 to 0802
        self.partIDhex = partitionID[2:]
        self.DetName = 'Pixels' + partitionID[5:] + '_LocDaq_' + partitionID[2:]
        self.bcids = bcids
        
        self.ch_com = CHostInterface()
        self.connect_CH(self.disp_addr,self.DetName)
        self._stop = False
        self._run = True
        self.EoR_rec = False
        self.SoR_rec = False
        self.EoS_rec = False
        self.SoS_rec = False
        self.Enable_rec = False
        self.error_printed = False
        self.converter_started = False
        self.terminated = False
        self.cmd = []
        self.run_number = 0
        self.cycleID = 0
        self.command = 'none'
        self.scan_status = None
        self.special_header = np.empty(shape=(1,), dtype= FrHeader)
        self.special_header['size'] = 16
        self.special_header['partID'] = self.partitionID
        self.special_header['timeExtent'] = 0
        self.special_header['flags'] = 0
        
        
    def _signal_handler(self, signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # setting default handler... pressing Ctrl-C a second time will kill application
        self.disconnect_CH(self.disp_addr)
        
    
    def cycle_ID(self):
        ''' counts in 0.2s steps from 08. April 2015 '''
        start_date = datetime.datetime(2015, 04, 8, 00, 00)
        return int((datetime.datetime.now() - start_date).total_seconds()*5)
    
    
    def connect_CH(self,disp_addr,DetName):
        ''' connect to dispatcher, DetName needed for identification. only needed at initial connection'''
        self.ch_com.init_link(disp_addr, subscriber=None)
        if self.ch_com.status < 0:
            logger.error('Could not connect to host %s' % disp_addr)
        elif self.ch_com.status >= 0 :
            self.ch_com.subscribe(DetName)
        if self.ch_com.status < 0:
            logger.error('Could not subscribe with name=%s to host %s' % (DetName, disp_addr))
        elif self.ch_com.status >= 0 :
            self.ch_com.send_me_always()
            
    
    def disconnect_CH(self, disp_addr):
        logger.info('Dropping connection to %s' % disp_addr)
        self.ch_com.drop_conn()
        self.CH_head_reciever.stop()
        self._run = False
        logger.info("setting self._run to False")
    
    def receive(self):
        ''' 
        main loop for reception and execution of commands.
        starts threads corresponding to the command recieved.
        Every command is acknowledged to the dispatcher first, upon successful execution of the command a done message is sent.
        '''
        
        if isinstance(current_thread(), _MainThread):
            logger.info('Press Ctrl-C twice to terminate RunControl')
            signal_handler = self._signal_handler
            signal.signal(signal.SIGINT, signal_handler)
#             signal.signal(signal.SIGTERM, signal_handler)

        self.converter = ship_data_converter.DataConverter(pybar_addr = self.converter_socket_addr, ports = self.ports, partitionID = self.partitionID)
        self.converter.name = 'DataConverter'
#             converter.daemon = True
        self.mngr = RunManager(self.pybar_conf)
        self.join_scan_thread = None
        recv_end, send_end = multiprocessing.Pipe(False)
        self.CH_head_reciever = CHostReceiveHeader(send_end)
        self.CH_head_reciever.name = 'CHostHeadReciever'
        self.CH_head_reciever.daemon = True
        
        
        while True:
            try:
                time.sleep(0.001) # TODO: why does this work? sleep should lock the global interpreter?
    #                 if self.status >=0 and self.ch_com.status >=0 :
    #                     self.status = ch.get_head_wait('DAQCMD', self.ch_com.cmdsize)
    #                     if self.status >=0 and self.ch_com.status >=0 :
    #                         self.cmd = self.ch_com.get_cmd() # recieved command contains command word [0] and additional info [1]... different for each case
                if not self.CH_head_reciever.is_alive() and self._run :
                    self.CH_head_reciever.start() # TODO: CH receiver does not terminate upon 2 x ctrl+c
                if not self.converter.is_alive() and self.converter_started:
                    logging.error('\n\n\n         DataConverter was started but is not alive anymore \n\n')
                    self.converter_started = False
                    raise RuntimeWarning
                if recv_end.poll(): # look whether there is a command in pipe
                    self.cmd = recv_end.recv()
#                     if len(self.cmd) > 1:
#                         self.command = self.cmd[0]
#                     elif len(self.cmd)==0 or len(self.cmd) ==1 :
                    self.command = self.cmd[0]
                    if self.command in self.commands and self.ch_com.status >=0:
                        self.ch_com.send_ack(tag='DAQACK',msg = '%s %04X %s' %(self.command, self.partitionID, self.disp_addr)) # acknowledge command
                        
                        self.react() # holds reactions to the DAQ commands
                        
                    else:
                        logger.error('Command=%s could not be identified' % self.cmd)
                elif self.command in self.commands and self.CH_head_reciever.status.value >=0:
                    self.error_printed = False
                    if self.command =='Enable' and self.Enable_rec:
                        self.ch_com.send_done('Enable',self.partitionID, self.status)
                        self.Enable_rec = False
                    if self.command == 'SoR' and self.SoR_rec: 
#                         self.scan_status = self.join_scan_thread(0.001) # TODO: self.join_scan_thread only defined after start of scan, bad practice?
                        if self.scan_status == 'RUNNING' :
                            self.special_header['frameTime'] = 0xFF005C01
                            self.special_header['cycleID'] = self.cycleID
                            self.ch_com.send_data(tag = 'RAW_' + partIDhex, header = self.special_header, hits=None)
                            self.ch_com.send_done('SoR',self.partitionID, self.status)
                            self.SoR_rec = False
                    elif self.command == 'EoR' and self.EoR_rec:
                        if self.scan_status == 'FINISHED' or self.scan_status == 'ABORTED' or self.scan_status == 'STOPPED':
                            self.special_header['frameTime'] = 0xFF005C02
                            self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                            self.ch_com.send_done('EoR',self.partitionID, self.status)
                        elif self.scan_status == 'CRASHED':
                            self.special_header['frameTime'] = 0xFF005C02
                            self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                            self.ch_com.send_done('EoR',self.partitionID, self.status)
                        self.EoR_rec = False
                    elif self.converter.all_workers_finished.wait(0.01) and self.SoS_rec: # can not check for command = SoS because SoS done msg can only be sent after EoS (SoS is "done" after buffering last event. This is triggered bei EoS signal.)
                        self.ch_com.send_done('SoS',self.partitionID, self.status) # send done is called after last trigger is read out
                        self.SoS_rec = False
                    elif self.command == 'EoS' and self.converter.EoS_data_flag.wait(0.01) and self.EoS_rec:
                        self.special_header['frameTime'] = 0xFF005C04
                        self.special_header['cycleID'] = self.cycleID
                        self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
                        self.ch_com.send_done('EoS', self.partitionID, self.status)
                        self.converter.EoS_reset() # TODO: bad practice, reset should not be here but in react method
                        self.EoS_rec = False
                elif self.CH_head_reciever.status.value < 0 :
                    if not self.error_printed:
                        logger.error('Header could not be recieved')
                        self.error_printed = True
                if self._stop == True:
                    break
                else:
                    continue
            except RuntimeWarning, RuntimeError:
                if self.converter_started == False:
                    print "DataConverter alive? ", self.converter.is_alive()
                    self.converter.stop()
                    logger.info('Restarting DataConverter.')
                    if not self.converter.is_alive():
                        self.converter.start()
                        if self.converter.is_alive():
                            self.converter_started = True
                    self.disconnect_CH(self.disp_addr)
                    continue
                else:
                    break
#         self.scan_status = self.join_scan_thread(0.01)
#         logger.info('scan status : %s' % self.scan_status)
        if self._stop == True:    
            self.CH_head_reciever.stop()
            self.converter.stop()
            if self.scan_status:
                self.join_scan_thread(timeout = 0.01)
                self.mngr.current_run.abort()
        self.disconnect_CH(self.disp_addr)
        logger.error('Loop exited')
        
#             
#             else:
#                 logger.error('RuntimeWarning, terminating')
            

            
#             logger.error('Exception, terminating')
#             print e.__class__.__name__ + ": " + str(e)
            

    def react(self):
        ''' 
        for each DAQ command a specific reaction is executed.
        
        Enable:    The DataConverter will be started and send data to the dispatcher
        
        Disable:    The Detector will continue taking data but the DataConverter will be stopped.
                    in this way no data will be sent to the dispatcher.
        
        SoR:        Starts a new Run, this means a new "ExtTriggerScanSHiP". If the DataConverter is not alive it will be started.
        
        EoR:        Ends the run, the "ExtTriggerScanSHiP" will be stopped and the DataConverter will be reset.
        
        SoS:        Determinates the next spill, the cycleID for saving the Data is updated, and Workers are set to receive data again
        
        EoS:        Tells the loop when to start sending data. Starts conversion of collected hits. When all events are sent out, the DataConverter
                    will be reset and wait for the next SoS.
                    
        Stop:       All processes will be terminated. Realized with a flag which will stop the "receive" loop and all processes
        '''
        
        if self.command == 'Enable': # enable detector partition
            self.enabled = True
            self.Enable_rec = True
#             if not self.converter.is_alive():
#                 self.converter.start()
#                 self.converter_started = True
#                 logger.info('Received Enable, starting DataConverter')
#             else:
            logger.info('Received Enable.')
        elif self.command == 'Disable': #disable detector partition
            self.enabled = False
    #             if self.converter.is_alive():
#                 self.converter.stop()
#                 logger.info('Received Disable, stopping DataConverter')
#             else:
            logger.info('Received Disable.')
        elif self.command == 'SoR': # Start new pyBAR ExtTriggerScanShiP.
            self.SoR_rec = True
            if len(self.cmd) > 1 :
                self.run_number = int(self.cmd[1])
            else:
                self.run_number = 999
            if not os.path.exists("./RUN_%03d/" % self.run_number):
                os.makedirs("./RUN_%03d/" % self.run_number)
            if not self.converter.is_alive():
                self.converter.start()
                self.converter.reset(cycleID=self.cycle_ID(), msg = 'SoR command, resetting DataConverter')
                self.converter_started = True
            else:
                self.converter.reset(cycleID=self.cycle_ID(), msg = 'SoR command, resetting DataConverter')
            self.converter.SoR_flag.set()
            with multiprocessing.Lock():
                self.converter.run_number.value = self.run_number
            #start pybar trigger scan
#             self.join_scan_thread = self.mngr.run_run(ThresholdScan, use_thread=True, catch_exception=False)
            
            
            self.join_scan_thread = self.mngr.run_run(ExtTriggerScanSHiP, run_conf={'scan_timeout': 86400,# 'max_triggers':0, 
                                                                        'trig_count':self.bcids, 'ship_run_number': self.run_number},
                                                                        use_thread=True ,catch_exception=False)
            self.scan_status = self.join_scan_thread(10)
#             transfer_file('/media/silab/data/98_module_0_ext_trigger_scan_s_hi_p.h5',#'/media/data/SHiP/charm_exp_2018/test_data_converter/elsa_testbeam_data/take_data/module_0/96_module_0_ext_trigger_scan.h5',
#                            self.converter_socket_addr[:-4] + ports[0])
            logger.info('Scan started, status = %s' % self.scan_status)
#             if self.scan_status == 'RUNNING': 
#                 self.ch_com.send_done('SoR',self.partitionID, self.status)
        elif self.command == 'EoR': # stop existing pyBAR ExtTriggerScanShiP
            self.EoR_rec = True
            if self.mngr.current_run.__class__.__name__ == 'ExtTriggerScanSHiP':
#                 self.join_scan_thread(timeout = 0.01)
                self.mngr.current_run.stop(msg='ExtTriggerScanSHiP')
                self.scan_status = self.join_scan_thread() # TODO: join after current_run.stop ?
            else:
                logger.error('Recieved EoR command, but no ExtTriggerScanSHiP running. Last scan status : %s' % self.scan_status)
            if self.converter.is_alive():
                self.converter.EoR_flag.set()
                self.converter.reset(cycleID = self.cycle_ID(), msg='EoR command, resetting DataConverter') # reset interpreter and event counter
            else:
                logger.error('Recieved EoR command to reset DataConverter, but no converter running')
        elif self.command == 'SoS': # new spill. Trigger counter will be reset by hardware signal. The software command triggers an empty header and resets of converter functions
            self.SoS_rec = True
            self.converter.SoS_flag.set()
            if len(self.cmd) > 1:
                cycleID = np.uint64(self.cmd[1])
            else:
                cycleID = self.cycle_ID() # create own cycleID if SoS command does not provide one
            self.converter.cycle_ID.value = cycleID
            logger.info('Recieved SoS header, cycleID = %s' % cycleID)
            self.converter.SoS_reset()
#             transfer_file('/media/silab/data/98_module_0_ext_trigger_scan_s_hi_p.h5',self.converter_socket_addr[:-4] + ports[0])
            self.special_header['frameTime'] = 0xFF005C03
            self.ch_com.send_data(tag = 'RAW_0802', header = self.special_header, hits=None)
        elif self.command == 'EoS': # trigger EoS header, sent after last event
            self.EoS_rec = True
            self.converter.EoS_flag.set()
            logger.info('Set EoS flag, local cycleID:%s' % self.cycle_ID())
        elif self.command == 'Stop':
            logger.info('Recieved Stop! Leaving loop, aborting all functions')
            self._stop = True


            

if __name__ == '__main__':
    
    usage = "Usage: %prog dispatcher_addr converter_addr configuration.yaml partitionID"
    description = "dispatcher_addr: Remote address of the sender (default: 127.0.0.1). converter_addr: local address of the SHiP data converter (default: tcp://127.0.0.1:5678). configuration: absolute path of pyBAR configuration.yaml. partitionID : ID of detector for dispatcher (default: 0X0802)"
    parser = OptionParser(usage, description=description)
    options, args = parser.parse_args()
    if len(args) == 1 and not args[0].find('configuration')==-1 :
        dispatcher_addr = '127.0.0.1'
        converter_addr = 'tcp://127.0.0.1'
        configuration = args[0]
        partitionID = '0X0802'
    elif len(args) == 4:
        dispatcher_addr = args[0]
        converter_addr = args[1]
        configuration = args[2]
        partitionID = args[3]
        
    else:
        parser.error("incorrect number of arguments")
    ports = ['5011','5012','5013','5014','5015','5016','5017','5018']
    bcids = 8
    rec = RunControl(dispatcher_addr,
                      converter_addr,
                      ports,
                      configuration,
                      partitionID,
                      bcids)
    
    rec.receive()

    
    
        
    
    
