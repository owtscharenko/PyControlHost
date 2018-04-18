# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
from numpy cimport ndarray
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t


import numpy as np
cimport numpy as cnp

# Global variables    
cdef extern uint32_t CycleID;  # centrally distributed cycleID
cdef extern uint32_t TrigTime; # ideal trigger time (us)
    
ctypedef struct DataFrameHeader:
    uint16_t size
    uint16_t partitionID
    uint32_t cycleIdentifier
    uint32_t frameTime
    uint16_t timeExtent
    uint16_t flags

ctypedef struct RawDataHit:
    uint16_t channelId
    uint16_t hitTime
    uint16_t extraData

ctypedef struct DataFrame:
    DataFrameHeader header   
    RawDataHit hits[]


