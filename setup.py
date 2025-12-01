from setuptools import setup, Extension
import numpy

module = Extension('fast_net',
                    sources=['fast_sockets.c'],
                    include_dirs=[numpy.get_include()])

setup(name='fast_net',
      version='1.0',
      description='Fast C socket operations for Numpy',
      ext_modules=[module])