import tables as tb
#from tables import descr_from_dtype
import numpy as np


class DataFrameHeader(tb.IsDescription):
    size = tb.UInt16Col(pos=0)
    partitionID = tb.UInt16Col(pos=1)
    cycleID = tb.UInt32Col(pos=2)
    frameTime = tb.UInt32Col(pos=3)
    timeExtent = tb.UInt16Col(pos=4)
    flags = tb.UInt16Col(pos=5)


class Hit(tb.IsDescription):
    channelID = tb.UInt16Col(pos=0)
    hit_data = tb.UInt16Col(pos=1)