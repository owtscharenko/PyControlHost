import time, sys
import PyControlHost.control_host_coms as ch
import numpy as np

# host = "131.220.165.153"
host = "localhost"
ch.say_hello_to('ControlHost')
ch.init_disp(host,"a TAG")

'''all values must be given in int32, then the package size in byte (nybtes) is calculated and everything is sent.
BUT: the LSB is sent first, the MSB last.
'''

hit = np.array([26880,16],dtype = np.int32)

partition_id = 2050 # 0x0802
cycle_id = 442368000 # 0x1a5e0000
frame_time = 0 # 0x00000000
time_extent = 15 # 0x000f
flags = 1 # 0x0001
size = 32
header = np.array([size, partition_id,cycle_id, frame_time, time_extent, flags],dtype = np.int32) 

send_buf = np.concatenate([header,hit])

nbytes = send_buf.shape[0] * 4
print nbytes

for i in range(0,1):
    time.sleep(0.1)
    ch.send_fulldata("SOMETAG1",hit,nbytes)
#     ch.send_fullstring("SOMETAG1", send_buf)
print "sent %i data packages to %s" %(i+1, host)
