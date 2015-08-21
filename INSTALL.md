# Installation

See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

Check the issue tracker in case you need help:
https://github.com/BVLC/caffe/issues


# Testing
To run all tests type

  $ make runtest

It is possible to separate phases of compilation and running. To compile use

  $ make test.testbin

This creates a runner, which is a Google test binary. Google test framework
which provides few useful options. The most used is to filter tests by name. For
example we want to run all tests with CPU in the name

  $ test/test.testbin --gtest_filter='*CPU*'

Please note that files, used in tests, are available on relative path to your
CWD in the moment of running tests.