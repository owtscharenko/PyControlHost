# Python test code for sending local DAQ data
# PG 14.06.2018 v.2.0 (extensible frame buffer)

from ctypes import *

#lib="/afs/cern.ch/user/p/petr/daq/ControlHost/lib/libconthost.so"

lib="/usr/local/lib/libconthost.so"
libContHost = CDLL(lib)
#libc = CDLL("/usr/lib64/libc.so.6")
 
libContHost.connect()

# define the Dispatcher host 
Dispatcher="local"   

rc = libContHost.init_disp_link(Dispatcher,"a DAQCMD a PYTHON")
print "init_disp_link: rc=", rc
 
rc = libContHost.my_id("Python_test")
print "my_id : rc= ", rc 

rc = libContHost.send_me_always()
print "send_me_always : rc=", rc

#tag=create_string_buffer(10)
#cmd=create_string_buffer(20)
#
#name = input("Check dispstat, enter \"go\" to continue (then send some test with \"PYTHON\" tag  ")
#sz=c_int()
#rc=libContHost.wait_head(tag,byref(sz))
#print "wait_head : ", rc, repr(tag.value), repr(sz.value)
#rc=libContHost.get_string(cmd,c_int(20))
#print "get_string : ", rc, repr(cmd.value)

#reserve sufficient space for a maximum possible # of hits 

MAX_HITS=50
HITS_INC=50

class Hit(Structure):
     _fields_=[("chID",  c_ushort),
               ("hData" ,c_ushort)]

def make_frame(nhits) :
  class Frame(Structure):
    _fields_=[("size",   c_ushort),
              ("partId", c_ushort),
              ("cycleId",c_int),
              ("frTime", c_int),
              ("timeEx", c_ushort),
              ("flags",  c_ushort),
              ("hits",   Hit*MAX_HITS)]  
  print "New Frame class instance for up to ",nhits, "hits"
  return Frame()


# instantiate the frame object only when the hit space is exceeded 
frame   =  make_frame(MAX_HITS)
partId  = 0x0801
cycleId = 0xcafe0000
frTime  = 0xdead0000
timeEx  = 0
flags   = 0xface

# loop over triggers (events)
for ev in range(1,100):

  # 20 events per cycle
  if ev%20 == 0:
     cycleId = cycleId + 0x100
     timeEx = 0
     print "New cycleId id=",format(cycleId,'08x') 

  Nhits = 20+ev*3  # normally, you get # of hits from the DAQ code....

  # re-allocate "frame" with more hit space
  if Nhits > MAX_HITS:
     del frame
     MAX_HITS = MAX_HITS + HITS_INC
     frame=make_frame(MAX_HITS)

  frTime = frTime + 16 # dummy time increment
  timeEx = timeEx + 1  # ev counter in spill
  #flags = ...

  frame.partId  = partId
  frame.cycleId = cycleId
  frame.frTime  = frTime
  frame.timeEx  = timeEx
  frame.flags   = flags

  frame.size    = 16+4*Nhits

  i=0
  for k in range(Nhits):
    h = frame.hits[k]
    i = i+1

    # for each hits, retrieve hit data from the DAQ code
    #h.chID = column<<9 ^ row
    #h.hData = tot <<12 ^ moduleID <<8 ^ flag<<4 ^ rel_time

    h.chID =  2*i + ev  # this are just dummy values, well visible in hexdump
    h.hData = 16*i + ev 
#    print (i,h.chID, h.hData)
#--- end of loop over hits

  libContHost.put_fulldata("RAW_0801", byref(frame), frame.size)
  print "put_fulldata : ", rc
#--- end of loop over events

name = input("Enter \"go\" to continue before drop_conn ")

rc=libContHost.drop_connection()

