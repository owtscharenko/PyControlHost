# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
cimport numpy as cnp
from numpy cimport ndarray
# cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error

cdef extern from "getdata.c":
    int init_disp_link(const char *host, const char *subscr)
    int init_2disp_link(const char *host, const char *subscrdata, const char *subscrcmd)
    int check_head(char *tag, int *size)
    int put_data(const char *tag, const void *buf, int size, int *pos)
    int put_fulldata(const char *tag, const void *buf, int size)
    int get_data(void *buf, int lim)
    int put_fullstring(const char *tag, const char *string)
    int wait_head(char *tag, int *size)
    int get_string(char *buf, int lim)
    int my_id(const char *id)
    int send_me_always()
    void drop_connection()
    
cdef extern from 'utils.h':
    cdef cppclass DataFrameHeader:
        DataFrameHeader()
    cdef cppclass Hit:
        Hit()
    unsigned short* build_EvtFrame(DataFrameHeader*& header, Hit*& hits, const unsigned int head_bytes, const unsigned int hits_bytes)
    unsigned short* build_special_header(DataFrameHeader*& header, const unsigned int head_length)
    
cdef int buf(cnp.ndarray a):
    return 6


def init_disp(const char *host, const char *subscr):
    return init_disp_link(host, subscr)

def init_disp_2way(const char *host, const char *subscrdata, const char *subscrcmd):
    return init_2disp_link(host, subscrdata, subscrcmd)

def subscribe(const char *id):
    return my_id(id)

def get_head(char *tag, cnp.ndarray[cnp.int8_t, ndim=1] size):
    return check_head(tag, <int*>size.data)

def get_head_wait(char *tag, cnp.ndarray[cnp.int8_t, ndim=1] size):
    return wait_head(tag, <int*>size.data)
 
def send_data(const char *tag, cnp.ndarray[cnp.int8_t, ndim=1] buf, int size, cnp.ndarray[cnp.int8_t, ndim=1] pos): # remove casting of numpy array!
    return put_data(tag, <const void *>buf.data, size, <int*>pos.data)

# def send_fulldata(const char *tag, cnp.ndarray[cnp.int32_t, ndim=1] buf, int size):
#     return put_fulldata(tag, <const void *>buf.data, size)
def send_fulldata(const char *tag, buf, int size):
    return put_fulldata(tag, <const void *>buf.data, size)

def send_fulldata_numpy(const char *tag, ndarray head, ndarray hits):
    cdef int head_bytes = head.nbytes
    cdef int hits_bytes = hits.nbytes # 4* hits.shape[0]
    cdef int size = head_bytes + hits_bytes
    cdef unsigned short *frame = build_EvtFrame(<DataFrameHeader*&> head.data, <Hit*&>  hits.data, head_bytes, hits_bytes)
    return put_fulldata(tag, <const void *>frame, size)

def send_header_numpy(const char *tag, ndarray head):
    cdef int head_size = head.nbytes
    cdef unsigned short *frame = build_special_header(<DataFrameHeader*&> head.data, head_size)
    return put_fulldata(tag, <const void *>frame, head_size)

def rec_cmd():
    cdef char cmd_str[200]
    cdef int lim=200
    return get_string(cmd_str, lim), cmd_str
   
def rec_data(buf, lim): # makes memcpy to buf of size lim
    return get_data(<void *>buf.data, lim), buf

def drop_ch_connection():
    return drop_connection()


def send_fullstring(const char *tag, const char *string):
    return put_fullstring(<const char *>tag, <const char *>string)
   
def accept_at_all_times():
    return send_me_always()   

def say_hello_to(name):
    print("Hello %s!" % name)