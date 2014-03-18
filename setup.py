
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='masque',
    packages=['masque'],
    # ext_modules=cythonize('masque/cutils.pyx'),
)