===============================================
Introduction
===============================================

This repository is providing scripts necessary for communication of pyBAR with the DAQ (dispatcher, run control) of the SHiP Charm experiment

Installation
============
You have to have Python 2/3 with the following modules installed:
  - cython
  - tables
  - numba
  - numpy
  - zmq
 
You also have to have installed ControlHost in the same folder as the top project folder, and compiled a dynamic library libconthost.so

The installation procedure for ControlHost is as follows:
	1. Download and untar archive to emtpy directory, e.g. "ControlHost".
	2. cd ControlHost/src/
	3. add the following lines to Makefile.project:

	.. code-block:: bash
		include $(VPATH)/Options.Makefile  # after the first non-commented statement
		
		CFLAGS_COMMON := -fPIC $(CFLAGS_COMMON)
		CXXFLAGS_COMMON := -fPIC $(CXXFLAGS_COMMON)


	4. make # (ignore compiler warnings)
	5. copy the static libary "libconthost.a" to an temporary empty directory
	6. in this directory perform the following commands one by one
	
	.. code-block:: bash
		ar -x libconthost.a
		gcc -shared *.o -o libconthost_shared.so
		rm -f *.o
	
	
	6. copy conthost_shared.so to the "ControlHost" folder.
	7. `cd ../bin; ls -l`  # check the presence of the system binary files dispatch, dispstat, stopdisp and the application executables tst1, tstsnd, tstrcv

	
If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change example files according to your needs you should install the module with

.. code-block:: bash

   python setup.py develop


Example usage
==============
tbd


