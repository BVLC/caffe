---
title: "Installation: Debian"
---

# Debian Installation

Caffe packages are available for several Debian versions, as shown in the
following chart

```
Your Distro     |  CPU_ONLY  |  CUDA  |     Alias
----------------+------------+--------+-------------------
Debian/stable   |     ✘      |   ✘    | Debian Jessie
Debian/testing  |     ✔      |   ☐    | Debian Stretch/Sid
Debian/unstable |     ✔      |   ✔    | Debian Sid
```

* `✘ ` You should take a look at [Ubuntu installation instruction](install_apt.html).

* `✔ ` You can install caffe with a single command line following this guide.

* `☐ ` The same with `✔ `. However it will not work any more when Debian/Stretch becomes the stable branch.

Last update: 2017-01-05

## Binary installation with APT

Apart from the installation methods based on source, Debian/unstable
and Debian/testing users can install pre-compiled Caffe packages via the official archive.

Make sure that there is something like the follows in your `/etc/apt/sources.list`:
```
deb http://MIRROR/debian CODENAME main contrib non-free
```
where `MIRROR` is your favorate Debian mirror, and `CODENAME ∈ {testing,stretch,sid}`.

Then we update APT cache and directly install Caffe. Note, the cpu version and
the cuda version cannot be installed at the same time.
```
# apt update
# apt install [ caffe-cpu | caffe-cuda ]
# caffe                                              # command line interface working
# python3 -c 'import caffe; print(caffe.__path__)'   # python3 interface working
```
It should work out of box.

#### Customizing caffe packages

Some users may need to customize the Caffe package. The way to customize
the package is beyond this guide. Here is only a brief guide of producing
the customized `.deb` packages. 

Make sure that there is something like this in your `/etc/apt/sources.list`:
```
deb http://ftp2.cn.debian.org/debian sid main contrib non-free
deb-src http://ftp2.cn.debian.org/debian sid main contrib non-free
```

Then we build caffe deb files with the following commands:
```
$ sudo apt update
$ sudo apt install build-essential debhelper devscripts    # standard package building tools
$ sudo apt build-dep [ caffe-cpu | caffe-cuda ]            # the most elegant way to pull caffe build dependencies
$ apt source [ caffe-cpu | caffe-cuda ]               # download the source tarball and extract
$ cd caffe-XXXX
[ ... optional, customize caffe code/build ... ]
$ dch -llocal "Modified XXX in order to XXX"          # write your one-line changelog
$ debuild -B -j4                                      # build caffe with 4 parallel jobs (similar to make -j4)
[ ... building ...]
$ debc                                                # optional, if you want to check the package contents
$ sudo debi                                           # optional, install the generated packages
```
The resulting deb packages can be found under the parent directory of the source tree.

Note, the `dch ...` command line above is for bumping the package version number
and adding an entry to the package changelog. If you would like to write
more than one changelog entry, use subsequent `dch` command (see `man 1 dch`)
instead of manually modifing `debian/changelog` unless you know how to keep its format correct.
The changelog will be installed at e.g. `/usr/share/doc/caffe-cpu/changelog.Debian.gz`.

## Source installation

Source installation under Debian/unstable is similar to that of Ubuntu, but
here is a more elegant way to pull caffe build dependencies:
```
$ sudo apt build-dep [ caffe-cpu | caffe-cuda ]
```
Note, this requires a `deb-src` entry in your `/etc/apt/sources.list`.

#### Compiler Combinations

Some users may find their favorate compiler doesn't work well with CUDA.
```
CXX compiler |  CUDA 7.5  |  CUDA 8.0  |
-------------+------------+------------+-
GCC-7        |     ?      |     ?      |
GCC-6        |     ✘      |     ✘      |
GCC-5        |     ✔ [1]  |     ✔      |
CLANG-4.0    |     ?      |     ?      |
CLANG-3.9    |     ✘      |     ✘      |
CLANG-3.8    |     ?      |     ✔      |
```

`[1]` CUDA 7.5 's `host_config.h` must be patched before working with GCC-5.

BTW, please forget the GCC-4.X series, since its `libstdc++` ABI is not compatible with GCC-5's.
You may encounter failure linking GCC-4.X object files against GCC-5 libraries.
(See https://wiki.debian.org/GCC5 )

## Notes

* Consider re-compiling OpenBLAS locally with optimization flags for sake of
performance. This is highly recommended for any kind of production use, including
academic research.

* If you are installing `caffe-cuda`, APT will automatically pull some of the
CUDA packages and the nvidia driver packages. Please be careful if you have
manually installed or hacked nvidia driver or CUDA toolkit or any other
related stuff, because in this case APT may fail.

* Additionally, a manpage (`man caffe`) and a bash complementation script
(`caffe <TAB><TAB>`, `caffe train <TAB><TAB>`) are provided.
Both of the two files are still not merged into caffe master.

* The python interface is Python 3 version: `python3-caffe-{cpu,cuda}`.
No plan to support python2.

* If you encountered any problem related to the packaging system (e.g. failed to install `caffe-*`),
please report bug to Debian via Debian's bug tracking system. See https://www.debian.org/Bugs/ .
Patches and suggestions are also welcome.

## FAQ

* where is caffe-cudnn?

CUDNN library seems not redistributable currently. If you really want the
caffe-cudnn deb packages, the workaround is to install cudnn by yourself,
and hack the packaging scripts, then build your customized package.

* I installed the CPU version. How can I switch to the CUDA version?

`sudo apt install caffe-cuda`, apt's dependency resolver is smart enough to deal with this.

* Where are the examples, the models and other documentation stuff?

```
sudo apt install caffe-doc
dpkg -L caffe-doc
```

* Where can I find the Debian package status?

https://tracker.debian.org/pkg/caffe  (for the CPU_ONLY version)

https://tracker.debian.org/pkg/caffe-contrib  (for the CUDA version)
