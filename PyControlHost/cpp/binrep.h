#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <ctime>
#include <cmath>

unsigned int shift_left(unsigned int number,unsigned int positions)
	{
	unsigned int res;
	res = number << positions;
	return res;
	}

unsigned int bit_or(unsigned int number1, unsigned int number2)
	{
	unsigned int res;
	res = number1 ^ number2;
	return res;
	}

unsigned int bit_or4(unsigned int number1, unsigned int number2, unsigned int number3, unsigned int number4)
	{
	unsigned int res;
	res = number1 ^ number2 ^ number3 ^ number4 ;
	return res;
	}
