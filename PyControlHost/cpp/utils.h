#include <stdint.h>
//#include <iostream>
//#include <iomanip>
//#include <string>
//#include <ctime>
//#include <cmath>


typedef struct DataFrameHeader{
    uint16_t size;
    uint16_t partitionID;
    uint32_t cycleID;
    uint32_t frameTime;
    uint16_t timeExtent;
    uint16_t flags;
	}DataFrameHeader;

typedef struct Hit{
	uint16_t channelId;
	uint16_t hit_Data;
	}Hit;


//unsigned int shift_left(unsigned int number,unsigned int positions)
//	{
//	unsigned int res;
//	res = number << positions;
//	return res;
//	}
//
//unsigned int bit_or(unsigned int number1, unsigned int number2)
//	{
//	unsigned int res;
//	res = number1 ^ number2;
//	return res;
//	}
//
//unsigned int bit_or4(unsigned int number1, unsigned int number2, unsigned int number3, unsigned int number4)
//	{
//	unsigned int res;
//	res = number1 ^ number2 ^ number3 ^ number4 ;
//	return res;
//	}

unsigned short *build_EvtFrame(DataFrameHeader*& header, Hit*& hits, const unsigned int& head_bytes, const unsigned int& hits_bytes)
	{
//  head_lenght and hits_length are size in byte of the respective array
//  join header and hits array by memcopy, then send as one memory block
//  allocate contiguous part of memory of correct size

	unsigned short *path;
	path = (unsigned short *)malloc(head_bytes + hits_bytes);
	//  now: copy header to beginning of path
	memcpy(path, header, head_bytes);
	//  copy hits to end of header
//	printf("end of header in memcopy %i" , head_bytes/sizeof(unsigned short));
	memcpy(&path[head_bytes/sizeof(unsigned short)],hits, hits_bytes);

//	printf("size of unsigned short %i\n" , sizeof(unsigned short));
//    printf("header: %p %p %p %p %p %p %p %p\n", path[0],path[1],path[2],path[3],path[4],path[5],path[6],path[7]);
//    printf("first hit: %p\n", % path[head_length])
//    printf("memcp size: %i Byte\n",head_length + hits_length);
	return path;
	}

unsigned short *build_special_header(DataFrameHeader*& header, const unsigned int& head_length)
	{
//  build header for reaction to commands (SoR, EoR, SoS, EoS...)
//  allocate contiguous part of memory of correct size
	unsigned short *path;
	path = (unsigned short *)malloc(head_length);
	//  now: copy header to beginning of path
	memcpy(path, header, head_length);

	return path;
	}
