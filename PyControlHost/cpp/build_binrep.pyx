# cython: profile=True
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
cimport numpy as np
from bitarray import bitarray
# from numpy cimport ndarray
np.import_array()  # if array is used it has to be imported, otherwise possible runtime error


_lookup = {
    '0': b'0000',
    '1': b'0001',
    '2': b'0010',
    '3': b'0011',
    '4': b'0100',
    '5': b'0101',
    '6': b'0110',
    '7': b'0111',
    '8': b'1000',
    '9': b'1001',
    'a': b'1010',
    'b': b'1011',
    'c': b'1100',
    'd': b'1101',
    'e': b'1110',
    'f': b'1111',
    'A': b'1010',
    'B': b'1011',
    'C': b'1100',
    'D': b'1101',
    'E': b'1110',
    'F': b'1111',
    'L': b'',
}


cdef binary_repr(num, width=None):
    """binary_repr(num, width=None)
    Converts an integer number to its binary representation.
    For negative numbers, if width is not given, a minus sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.
    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string if `num` is positive, the length of
        the two's complement if `num` is negative.
    Returns
    -------
    out : str
        Binary representation of `num` or two's complement of `num`.
    Examples
    --------
    >>> binary_repr(3)
    '11'
    >>> binary_repr(-3)
    '-11'
    >>> binary_repr(3, width=4)
    '0011'
    The two's complement is returned when the input number is negative and
    width is specified:
    >>> binary_repr(-3, width=4)
    '1101'
    """
    # ' <-- unbreak Emacs fontification
    sign = b''
    if num < 0:
        if width is None:
            sign = b'-'
            num = -num
        else:
            # replace num with its 2-complement
            num = 2**width + num
    elif num == 0:
        return b'0'*(width or 1)
    ostr = hex(num)
    out = b''.join(_lookup[chr] for chr in ostr[2:])
    out = out.lstrip(b'0')
    if width is not None:
        out = out.zfill(width)
    return sign + out

cdef extern from "binrep.h":
    unsigned int shift_left(unsigned int number,unsigned int positions)
    unsigned int bit_or(unsigned int number1, unsigned int number2)
    unsigned int bit_or4(unsigned int number1, unsigned int number2, unsigned int number3, unsigned int number4)
    
cdef packed struct ch_hit_info:
    np.uint16_t row  # row value (unsigned short int: 0 to 65.535)
    np.uint8_t column  # column value (unsigned char: 0 to 255)
    np.uint8_t tot  # ToT value (unsigned char: 0 to 255)
    np.uint8_t relative_BCID  # relative BCID value (unsigned char: 0 to 255)


def hit_to_binary(np.ndarray[ch_hit_info,ndim=1] data_array, int moduleID):
    ch_hit_data = []
    for i in range(data_array.shape[0]):
        channelID = bitarray()
        second_word = bitarray()
#         print data_array['row'][i], data_array['column'][i]
        channelID.extend([binary_repr(bit_or(shift_left(data_array['row'][i],7), data_array['column'][i]))])
#         print binary_repr(shift_left(data_array['row'][i],7))
#         print bit_or(shift_left(data_array['row'][i],7), data_array['column'][i])
#         print binary_repr(bit_or(shift_left(data_array['row'][i],7), data_array['column'][i]),width=16)
        second_word.extend([binary_repr(bit_or4(shift_left(data_array['tot'][i],12),
                                                shift_left(moduleID,8) ,
                                                shift_left(0, 4),
                                                data_array['relative_BCID'][i]))])
        ch_hit_data.extend([channelID, second_word])
    return ch_hit_data

