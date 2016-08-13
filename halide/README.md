Halide for Caffe

How to get started?

We want to use halide layer shipped for the unit tests. As the build
process is quite complicated we have included a cmake file that should
build this layer, and install it to the installation directory

Copy the contents of the ./test folder to the curren  directory, but
don't overwrite the CMakeLists.txt file already there.

cp ./test/generator .
cp ./test/wrapper .

Now re-compile caffe with BUILD_halide on and install. You should now have
the usable halide library file:
./install/lib/halide/libplip_wrapper.so


How does the build process work?

In summary:
halide generator -> halide object -> halide wrapper

We first compile a halide generator, to which halide functions have been
registered, see ./generator/register_gen.cpp. You can append to this file
or add arbitrary .cpp files to this directory.

Secondly we need to inform the make-system which halide objects to generate
the halide objects. As these are named in .cpp files they are not easily accessible to the build system, as a workaround just add a empty <generator_name>.gen file, eg:

touch ./generator/plip.gen

Third compile a wrapper layer. These are located in the ./wrappers folder.
A new wrapper is created for each .cpp file in the ./wrappers folder. Each
of these are linked with all previously specified halide objects. So use
different names for all objects. So just add any new .cpp file similar to
plip_wrapper.cpp.


FAQ:

How do I change halide targets? 
Look at the CMakeList.txt file and change target=cuda to something different.

How do I change target per object?
Append the info to the .gen file and extract it using cmake.
