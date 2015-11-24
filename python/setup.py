import os
import subprocess
from distutils.core import setup
from distutils.core import Extension
import pip.download
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt',
                                  session=pip.download.PipSession())

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src', 'caffe')
INC_DIR = os.path.join(BASE_DIR, 'include')

subprocess.call(['protoc', '--proto_path', os.path.join(SRC_DIR, 'proto'),
                 os.path.join(SRC_DIR, 'proto', 'caffe.proto'),
                 '--cpp_out', SRC_DIR])

def get_sources():
    sources = []
    for dirName, subdirList, fileList in os.walk(SRC_DIR):
        for fname in fileList:
            if fname.endswith('.cpp'):
                sources.append(os.path.join(dirName, fname))

    return sources

print(get_sources())
caffe_module = Extension(
    'caffe',
    define_macros = [('CPU_ONLY', '1')],
    libraries = ['openblas'],
    include_dirs = [
        INC_DIR,
        SRC_DIR,
        '/usr/include/python2.7',
        '/usr/lib/python2.7/dist-packages/numpy/core/include'
    ],
    sources = get_sources(),
)

setup(
    name = 'caffe',
    version = '1.0rc2',
    description = ('Caffe is a deep learning framework made with expression, '
                    'speed, and modularity in mind.'),
    author = 'BVLC members and the open-source community',
    url = 'https://github.com/BVLC/caffe',
    long_description = '''
    This is really just a demo package.
    ''',
    license = 'BSD',
    ext_modules = [caffe_module]
)

#print(caffe_module)
#config = {
#    'name': 'caffe',
#    'version': '0.1.0',
#    'author': 'BVLC members and the open-source community',
#    'packages': find_packages(),
#    'scripts': ['classify.py', 'detect.py',
#                'draw_net.py'],
#    'ext_modules': [caffe_module],
#    'platforms': ['Linux', 'MacOS X', 'Windows'],
#    'long_description': ('Caffe is a deep learning framework made with '
#                         'expression,  speed, and modularity in mind. It is '
#                         'developed by the Berkeley Vision and Learning '
#                         'Center (BVLC) and community contributors.'),
#    'install_requires': reqs,
#    'keywords': ['caffe', 'deep learning'],
#    'download_url': 'https://github.com/BVLC/caffe',
#    'classifiers': ['Development Status :: 3 - Alpha',
#                    'Environment :: Console',
#                    'Intended Audience :: Developers',
#                    'Intended Audience :: Science/Research',
#                    'License :: OSI Approved :: BSD License',
#                    'Natural Language :: English',
#                    'Programming Language :: Python :: 2.7',
#                    'Programming Language :: Python :: 3',
#                    'Programming Language :: Python :: 3.3',
#                    'Programming Language :: Python :: 3.4',
#                    'Topic :: Scientific/Engineering :: Artificial Intelligence',
#                    'Topic :: Software Development',
#                    'Topic :: Utilities'],
#    'zip_safe': False
#}
#
#setup(**config)
