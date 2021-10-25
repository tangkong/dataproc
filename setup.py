import sys
import os
import pip

from setuptools import setup, find_packages

#pip.main(['install',  'pyzmq', '--install-option=--zmq=bundled'])

setup(name='dataproc',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'dataproc': 'dataproc'},
    #package_data={'xrdc': ['inputs/*']},
#    scripts = [
#        'scripts/mecana.py', 'scripts/logbooksync.py'
#    ],
    #install_requires = ['gpflow', 'bayesian-optimization'],
    zip_safe = False,
    )

#pip.main(['install', 'git+https://github.com/uqfoundation/pathos.git'])
#pip.main(['install', 'pymongo'])
#pip.main(['install', 'pytest'])

#print  "Packages are: ", find_packages('.')
