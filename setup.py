# does gcc -Wall pat_c_send_and_cmd.c -o test -I/home/niko/ControlHost/include -L/home/niko/ControlHost/bin -lconthost
# libconthost_shared.so needs to be in pyproject folder or in /usr/lib
# This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

version = '0.1.0'

extensions = [
    Extension("PyControlHost.control_host_coms", ["PyControlHost/cpp/control_host_coms.pyx"],
        include_dirs = ['../ControlHost/include', '../ControlHost/src'],
        libraries = ['conthost_shared']), # library libconthost_shared.so must be in /usr/lib and accessible for user (chmod 0755)
    Extension("PyControlHost.build_binrep", ["PyControlHost/cpp/build_binrep.pyx"],)
#         library_dirs = ['/home/niko/git/ControlHost/bin'])
    ]

setup(
    name = "PyControlHost",
    version = version,
    packages = find_packages(),
    include_dirs = [np.get_include()], # needed on some systems to provided numpy c-library to compiler
    ext_modules = cythonize(extensions),
)