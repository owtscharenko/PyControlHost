# does gcc -Wall pat_c_send_and_cmd.c -o test -I/home/niko/ControlHost/include -L/home/niko/ControlHost/bin -lconthost
# libconthost_shared.so needs to be in pyproject folder regardless of what is written in library_dirs
from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize

extensions = [
    Extension("control_host_coms", ["control_host_coms.pyx"],
        include_dirs = ['/home/niko/ControlHost/include'],
        libraries = ['conthost_shared'],
        library_dirs = ['/home/niko/ControlHost/bin'])
    ]

setup(
    name = "PyControlHost",
    ext_modules = cythonize(extensions),
)