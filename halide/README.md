# Halide for Caffe

## Why use Halide?

[Halide](http://halide-lang.org/) is described as a language for image
processing and computational photography. It works by de-coupling the
definition of the algorithm from the definition from scheduling of the
computation. This conceptual separation allows for concise definition of the
algorithm and the ability to easily define different schedules, for different
architectures for example.

In the context of caffe the Halide layer can be used to bridge the gap between
python layers which are easy to implement and hand coded CUDA layers. This
works by Halide compiling a computation to a CUDA kernel which can then
accessed from caffe using the Halide layer. Halide can compile to any
of a number of different architectures such as: x86/SSE, CUDA, OpenCL etc.


## How is Halide called?

Calling halide from caffe requires running the Halide compiler, this makes
the build process a bit more complicated. The next section describes the
steps the build system needs to perform in order to produce a halide layer.

To explain what is happening I will first define a few terms. The *halide
function* is the piece of halide code that specifies the computation. This
function is then registered to a *halide generator* which, when run, will
generate a header file and a *halide object* file. This object will then
be linked to a (caffe) *wrapper* which is compiled as a shared library.


## How does the build work?

To use Halide with caffe we first have to re-install it with the `BUILD_halide`
variable activated.

To better explain the build process lets look at the halide function supplied
as part of the unit tests. First copy these to the `./halide` directory.

```
cp ./test/generator .
cp ./test/wrapper .
```

The *halide generator class* , which contain the *halide function* can be found
in `./generator/register_gen.cpp`. These files will be compiled to an *halide
generator executable. The way the build process is set up in `./CMakeLists.txt`
all *.cpp files in the `./generator` directory are included in this generator
executable, so each *halide generator class* should register with different
names.

The next step is to run the generator executable. Given the name which a
generator class has been registered and a target architecture this will
generate the specified objects* and their header files.

As the registered name is needed this step is a bit difficult for the build
system. ( It cannot easily look at the source code to find these names. )
To tell the build system to generate a particular object create an empty
file that is named `<registered name>.gen`

```
touch ./generator/testfunc.gen
```

The build system will generate all of these objects.

The next step is to combine halide objects into a wrapper. These are located in
the ./wrappers folder. The build system is configured to turn each .cpp file in
this folder into a wrapper shared library. Each of these libraries are linked
with all halide objects generated previously.

Now re-compile caffe with BUILD_halide on and install. You should now have
the usable halide library file:

```
cd ../build
cmake .. -DBUILD_halide=ON
make install
file ./install/lib/halide/libtestfunc_wrapper.so
```

As a final step we can test if this objects works, just set your install dir.

```
import tempfile
import numpy as np
import caffe
from caffe import layers as L

INSTALL_DIR = XXX

def create_neural_net(batch_size=1):
    library = INSTALL_DIR + "/lib/halide/libtestfunc_wrapper.so"
    netspec = caffe.NetSpec()
    netspec.data = L.DummyData(shape=[dict(dim=[batch_size, 1, 3, 5])], ntop=1)
    netspec.halide = L.Halide(netspec.data, halide_param=dict(library=library))
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(str(netspec.to_proto()))
        f.flush()
        net = caffe.Net(f.name, caffe.TEST)
    return net

if __name__=='__main__':
    caffe.set_mode_gpu()
    net = create_neural_net()
    net.blobs['data'].data[...]  = np.array( [[1,2,5,2,3],[9,4,1,4,8],[1,2,5,2,3] ] )
    net.forward()
    print(net.blobs["halide"].data)


```

## How do I compile a module by hand?
In bash run the following:

```
HALIDE_DIR=../../halide
CAFFE_DIR=../build/install
NAME=testfunc
g++ ./wrappers/testfunc_wrapper.cpp ../build/halide/testfunc.o -I../build/halide/ -shared -g -std=c++11 -fPIC -fno-rtti -I${HALIDE_DIR}/include -I${CAFFE_DIR}/include -I/opt/cuda/include -L ${HALIDE_DIR}/bin -lHalide -L${CAFFE_DIR}/lib  -lpthread -ldl  -lpthread -lz -lglog -lgflags -lboost_system-mt -lcaffe -o testfunc_wrapper.so

```


## How do I change halide targets?
Look at the CMakeList.txt file and change target=cuda to something different.

## How do I change target per object?
Append the info to the .gen file and extract it using cmake.
