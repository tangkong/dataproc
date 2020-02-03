#-----------------------------------------------------------------------------
# :author:    Robert Tang-Kong
# :email:     roberttk@slac.stanford.edu
# :copyright: (c) 2019-2020, Stanford Linear Accelerator Center
#
# Licensing? 
#-----------------------------------------------------------------------------

__project__     = u'dataproc'
__version__     = '0.0.1'
__description__ = u"Basic dumb data processing workflow"
__copyright__   = u'2019-2020 SLAC National Accelerator Laboratory'
__authors__     = [u'Robert Tang-Kong', ]
__author__      = ', '.join(__authors__)
__institution__ = u"Stanford Synchrotron Radiation Lightsource, SLAC"
__author_email__= u"roberttk@slac.stanford.edu"
__license__     = u"(c) " + __copyright__
__license__     += u" (see LICENSE.txt file for details)"
__platforms__   = 'any'
__zip_safe__    = False
__exclude_project_dirs__ = "tests fstore".split()
__python_version_required__ = '>3.6'
__install_requires__ = [
                        'pyFAI', 'fabio', 'pyopencl',
                        'numpy', 'scipy', 'pathlib', 'pandas', 'matplotlib',
                       ]

__package_name__ = __project__
__long_description__ = __description__

__classifiers__ = [
    'Development Status :: 1 - Planning',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: Freely Distributable',
    'License :: Public Domain',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
    'Topic :: Utilities',
]   