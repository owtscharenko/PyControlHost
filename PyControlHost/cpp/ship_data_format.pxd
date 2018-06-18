# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
from numpy cimport ndarray
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t


import numpy as np
from numba.tests.cache_usecases import packed_arr
cimport numpy as cnp

# Global variables    
# cdef extern uint32_t CycleID;  # centrally distributed cycleID
# cdef extern uint32_t TrigTime; # ideal trigger time (us)
    
cdef packed struct np_DataFrameHeader:
    uint16_t size
    uint16_t partitionID
    uint32_t cycleID
    uint32_t frameTime
    uint16_t timeExtent
    uint16_t flags

cdef packed struct np_Hit:
    uint16_t channelId
    uint16_t hit_Data
    
ctypedef struct DataFrameHeader:
    uint16_t size
    uint16_t partitionID
    uint32_t cycleID
    uint32_t frameTime
    uint16_t timeExtent
    uint16_t flags

ctypedef struct Hit:
    uint16_t channelId
    uint16_t hit_Data
    
    