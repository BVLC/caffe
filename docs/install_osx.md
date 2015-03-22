---
title: Installation: OS X
---

# OS X Installation

We highly recommend using the [Homebrew](http://brew.sh/) package manager.
Ideally you could start from a clean `/usr/local` to avoid conflicts.
In the following, we assume that you're using Anaconda Python and Homebrew.

**CUDA**: Install via the NVIDIA package that includes both CUDA and the bundled driver. **CUDA 7 is strongly suggested.** Older CUDA require `libstdc++` while clang++ is the default compiler and `libc++` the default standard library on OS X 10.9+. This disagreement makes it necessary to change the compilation settings for each of the dependencies. This is prone to error.

**Library Path**: We find that everything compiles successfully if `$LD_LIBRARY_PATH` is not set at all, and `$DYLD_FALLBACK_LIBRARY_PATH` is set to to provide CUDA, Python, and other relevant libraries (e.g. `/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib`).
In other `ENV` settings, things may not work as expected.

**General dependencies**

    brew install --fresh -vd snappy leveldb gflags glog szip lmdb
    # need the homebrew science source for OpenCV and hdf5
    brew tap homebrew/science
    hdf5 opencv

If using Anaconda Python, a modification to the OpenCV formula might be needed
Do `brew edit opencv` and change the lines that look like the two lines below to exactly the two lines below.

      -DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib
      -DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7

If using Anaconda Python, HDF5 is bundled and the `hdf5` formula can be skipped.

**Remaining dependencies, with / without Python**

    # with Python pycaffe needs dependencies built from source
    brew install --build-from-source --with-python --fresh -vd protobuf
    brew install --build-from-source --fresh -vd boost boost-python
    # without Python the usual installation suffices
    brew install protobuf boost

**BLAS**: already installed as the [Accelerate / vecLib Framework](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man7/Accelerate.7.html). OpenBLAS and MKL are alternatives for faster CPU computation.

**Python** (optional): Anaconda is the preferred Python.
If you decide against it, please use Homebrew.
Check that Caffe and dependencies are linking against the same, desired Python.

Continue with [compilation](installation.html#compilation).

## libstdc++ installation

This route is not for the faint of heart.
For OS X 10.10 and 10.9 you should install CUDA 7 and follow the instructions above.
If that is not an option, take a deep breath and carry on.

In OS X 10.9+, clang++ is the default C++ compiler and uses `libc++` as the standard library.
However, NVIDIA CUDA (even version 6.0) currently links only with `libstdc++`.
This makes it necessary to change the compilation settings for each of the dependencies.

We do this by modifying the Homebrew formulae before installing any packages.
Make sure that Homebrew doesn't install any software dependencies in the background; all packages must be linked to `libstdc++`.

The prerequisite Homebrew formulae are

    boost snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv

For each of these formulas, `brew edit FORMULA`, and add the ENV definitions as shown:

      def install
          # ADD THE FOLLOWING:
          ENV.append "CXXFLAGS", "-stdlib=libstdc++"
          ENV.append "CFLAGS", "-stdlib=libstdc++"
          ENV.append "LDFLAGS", "-stdlib=libstdc++ -lstdc++"
          # The following is necessary because libtool likes to strip LDFLAGS:
          ENV["CXX"] = "/usr/bin/clang++ -stdlib=libstdc++"
          ...

To edit the formulae in turn, run

    for x in snappy leveldb protobuf gflags glog szip boost boost-python lmdb homebrew/science/opencv; do brew edit $x; done

After this, run

    for x in snappy leveldb gflags glog szip lmdb homebrew/science/opencv; do brew uninstall $x; brew install --build-from-source --fresh -vd $x; done
    brew uninstall protobuf; brew install --build-from-source --with-python --fresh -vd protobuf
    brew install --build-from-source --fresh -vd boost boost-python

If this is not done exactly right then linking errors will trouble you.

**Homebrew versioning** that Homebrew maintains itself as a separate git repository and making the above `brew edit FORMULA` changes will change files in your local copy of homebrew's master branch. By default, this will prevent you from updating Homebrew using `brew update`, as you will get an error message like the following:

    $ brew update
    error: Your local changes to the following files would be overwritten by merge:
      Library/Formula/lmdb.rb
    Please, commit your changes or stash them before you can merge.
    Aborting
    Error: Failure while executing: git pull -q origin refs/heads/master:refs/remotes/origin/master

One solution is to commit your changes to a separate Homebrew branch, run `brew update`, and rebase your changes onto the updated master. You'll have to do this both for the main Homebrew repository in `/usr/local/` and the Homebrew science repository that contains OpenCV in  `/usr/local/Library/Taps/homebrew/homebrew-science`, as follows:

    cd /usr/local
    git checkout -b caffe
    git add .
    git commit -m "Update Caffe dependencies to use libstdc++"
    cd /usr/local/Library/Taps/homebrew/homebrew-science
    git checkout -b caffe
    git add .
    git commit -m "Update Caffe dependencies"

Then, whenever you want to update homebrew, switch back to the master branches, do the update, rebase the caffe branches onto master and fix any conflicts:

    # Switch batch to homebrew master branches
    cd /usr/local
    git checkout master
    cd /usr/local/Library/Taps/homebrew/homebrew-science
    git checkout master

    # Update homebrew; hopefully this works without errors!
    brew update

    # Switch back to the caffe branches with the formulae that you modified earlier
    cd /usr/local
    git rebase master caffe
    # Fix any merge conflicts and commit to caffe branch
    cd /usr/local/Library/Taps/homebrew/homebrew-science
    git rebase master caffe
    # Fix any merge conflicts and commit to caffe branch

    # Done!

At this point, you should be running the latest Homebrew packages and your Caffe-related modifications will remain in place.
