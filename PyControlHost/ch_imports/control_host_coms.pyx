# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
cimport numpy as cnp
# from numpy cimport ndarray
cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error

cdef extern from "getdata.c":
    int init_disp_link(const char *host, const char *subscr)
    int check_head(char *tag, int *size)
    int put_data(const char *tag, const void *buf, int size, int *pos)
    int put_fulldata(const char *tag, const void *buf, int size)
    int get_data(void *buf, int lim)
    int put_fullstring(const char *tag, const char *string)
    
cdef int buf(cnp.ndarray a):
    return 6


def init_disp(const char *host, const char *subscr):
    return init_disp_link(host, subscr)

def check_head(char *tag, cnp.ndarray[cnp.int8_t, ndim=1] size):
    return check_head(tag, size)
 
def send_data(const char *tag, cnp.ndarray[cnp.int8_t, ndim=1] buf, int size, cnp.ndarray[cnp.int8_t, ndim=1] pos):
    return put_data(tag, <const void *>buf.data, size, <int*>pos.data)

# def send_fulldata(const char *tag, cnp.ndarray[cnp.int32_t, ndim=1] buf, int size):
#     return put_fulldata(tag, <const void *>buf.data, size)
def send_fulldata(const char *tag, buf, int size):
    return put_fulldata(tag, <const void *>buf.data, size)
   
def rec_data(buf, lim): # makes memcpy to buf of size lim
    return get_data(<void *>buf.data, lim), buf

def send_fullstring(const char *tag, const char *string):
    return put_fullstring(<const char *>tag, <const char *>string)
   
def say_hello_to(name):
    print("Hello %s!" % name)