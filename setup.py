from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import petsc4py
import os
import subprocess

petsc_dir = os.environ.get('PETSC_DIR')
petsc_arch = os.environ.get('PETSC_ARCH')

include_dirs = [numpy.get_include(), petsc4py.get_include()]
library_dirs = []
libraries = ['petsc']
extra_link_args = []

if petsc_dir:
    include_dirs.append(os.path.join(petsc_dir, 'include'))
    if petsc_arch:
        include_dirs.append(os.path.join(petsc_dir, petsc_arch, 'include'))
        lib_dir = os.path.join(petsc_dir, petsc_arch, 'lib')
        library_dirs.append(lib_dir)
        extra_link_args.append(f'-Wl,-rpath,{lib_dir}')

# Add MPI includes
try:
    mpi_includes = subprocess.check_output(['mpicc', '--showme:incdirs']).decode('utf-8').strip().split()
    include_dirs.extend(mpi_includes)
    mpi_libdirs = subprocess.check_output(['mpicc', '--showme:libdirs']).decode('utf-8').strip().split()
    library_dirs.extend(mpi_libdirs)
    mpi_libs = subprocess.check_output(['mpicc', '--showme:libs']).decode('utf-8').strip().split()
    libraries.extend([l.replace('-l', '') for l in mpi_libs])
except:
    pass

setup(
    name='viamr',
    version='0.1',
    author='Stefano Fochesatto, Ed Bueler',
    author_email='elbueler@alaska.edu',
    description='AMR for VIs',
    long_description='Free boundary oriented adaptive mesh refinement for variational inequalities.',
    packages=['viamr'],
    ext_modules=cythonize([
        Extension("viamr.viamrcore",
                  sources=["viamr/viamrcore.pyx"],
                  include_dirs=include_dirs,
                  library_dirs=library_dirs,
                  libraries=libraries,
                  extra_link_args=extra_link_args)
    ]),
    zip_safe=False,
)
