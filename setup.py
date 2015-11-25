#!/usr/bin/python2

import os
from os.path import join, exists, abspath, dirname
import subprocess
from distutils.core import setup
from distutils.core import Extension
import pip.download
from pip.req import parse_requirements

BASE_DIR = abspath(dirname(__file__))
PROTO_DIR = join(BASE_DIR, 'src/caffe/proto')
SRC_DIR = join(BASE_DIR, 'src')
INC_DIR = join(BASE_DIR, 'include')
SRC_GEN = [join(SRC_DIR, 'caffe/proto/caffe.pb.h'),
           join(SRC_DIR, 'caffe/proto/caffe.pb.cc')]

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('python/requirements.txt',
                                  session=pip.download.PipSession())
reqs = [str(ir.req) for ir in install_reqs]

# Create the proto module in python/caffe, from the caffe proto buffer
module_dir = join(BASE_DIR, 'python/caffe/proto')

try:
    os.makedirs(module_dir)
except OSError:
    pass

# Makes an empty __init__ file for the caffe.proto module
if not exists(join(module_dir, '__init__.py')):
    with open(join(module_dir, '__init__.py'), 'w') as f:
        f.write('')

if not exists(join(module_dir, 'caffe_pb2.py')):
    # Converts the caffe.proto protocol buffer into a python format
    subprocess.call(['protoc',
                     join(PROTO_DIR, 'caffe.proto'),
                     '--proto_path', PROTO_DIR,
                     '--python_out', join(BASE_DIR, 'python/caffe/proto')])

# Find source files in src/* and python/*
sources = []

# Generates cc files from proto buffer
if (not exists(join(PROTO_DIR, 'caffe.pb.cc')) or
        not exists(join(PROTO_DIR, 'caffe.pb.h'))):
    # Converts the caffe.proto protocol buffer into a python format
    subprocess.call(['protoc',
                     join(PROTO_DIR, 'caffe.proto'),
                     '--proto_path', PROTO_DIR,
                     '--cpp_out', PROTO_DIR])

for dirName, subdirList, fileList in os.walk(SRC_DIR):
    if os.path.basename(dirName) in ('test'):
        # Skip tests, utils
        continue
    for fname in fileList:
        if fname.endswith('.cpp') or fname.endswith('.cc'):
            sources.append(join(dirName, fname))

for dirName, subdirList, fileList in os.walk('python'):
    for fname in fileList:
        if fname.endswith('.cpp') or fname.endswith('.cc'):
            sources.append(join(dirName, fname))


caffe_module = Extension(
    'caffe/_caffe',
    define_macros = [('CPU_ONLY', '1')],
    libraries = ['cblas', 'blas', 'boost_thread', 'glog', 'gflags', 'protobuf',
                 'boost_python', 'boost_system', 'boost_filesystem', 'm',
                 'hdf5_hl', 'hdf5'],
    include_dirs = [
        SRC_DIR,
        INC_DIR,
        '/usr/include/python2.7',
        '/usr/lib/python2.7/dist-packages/numpy/core/include'
    ],
    sources = sources,
    extra_compile_args = ['-Wno-sign-compare'],
)

setup(
    name = 'caffe',
    version = '1.0rc2',
    description = ('Caffe is a deep learning framework made with expression, '
                    'speed, and modularity in mind.'),
    author = 'BVLC members and the open-source community',
    url = 'https://github.com/BVLC/caffe',
    license = 'BSD',
    ext_modules = [caffe_module],
    package_dir = {'': 'python'},
    packages = ['caffe', 'caffe/proto'],
    scripts = ['python/classify.py', 'python/detect.py', 'python/draw_net.py'],
    platforms = ['Linux', 'MacOS X', 'Windows'],
    long_description = ('Caffe is a deep learning framework made with '
                        'expression,  speed, and modularity in mind. It is '
                        'developed by the Berkeley Vision and Learning '
                        'Center (BVLC) and community contributors.'),
    install_requires = reqs,
    keywords = ['caffe', 'deep learning'],
    download_url = 'https://github.com/BVLC/caffe',
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Software Development',
                   'Topic :: Utilities'],
    provides = ['caffe'],
)

