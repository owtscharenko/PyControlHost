===============================================
Introduction
===============================================

This repository is providing scripts necessary for communication of pyBAR with the DAQ (dispatcher, run control) of the SHiP Charm experiment.
Hit data from FE-I4 modules (provided by pyBAR) is converted to SHiP data format and sent to DAQ. Also run_control commands are recieved and executed.

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
- Download and untar archive to emtpy directory, e.g. "ControlHost". Go to this directory:

.. code-block:: bash

	cd ControlHost/src/
	
- add the following lines to Makefile.project:

.. code-block:: bash

	include $(VPATH)/Options.Makefile  # after the first non-commented statement
	
	CFLAGS_COMMON := -fPIC $(CFLAGS_COMMON)
	CXXFLAGS_COMMON := -fPIC $(CXXFLAGS_COMMON)

- build the project:

.. code-block:: bash
	
	make

- copy the static libary "libconthost.a" to an temporary empty directory
- in this directory perform the following commands one by one
	
.. code-block:: bash

	ar -x libconthost.a
	gcc -shared *.o -o libconthost_shared.so
	rm -f *.o
	
- copy conthost_shared.so to the "ControlHost" folder.
- check the presence of the system binary files dispatch, dispstat, stopdisp and the application executables tst1, tstsnd, tstrcv:

.. code-block:: bash

	cd ../bin; ls -l
	
If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change example files according to your needs you should install the module with

.. code-block:: bash

   python setup.py develop


Example usage
==============
tbd


