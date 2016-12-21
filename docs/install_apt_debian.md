---
title: "Installation: Debian"
---

# Debian Installation

Caffe packages are available for `Debian/unstable`. Debian/stable users
should take a look at Ubuntu installation instruction.  

Only experienced linux users are recommended to try Debian/unstable (Sid).  

Last update: Dec.21 2016  

## Debian/unstable

Apart from the installation methods based on source, Debian/unstable
users can install pre-compiled Caffe packages via the official archive.

### Binary installation

Make sure that there is something like the follows in your `/etc/apt/sources.list`:
```
deb http://ftp2.cn.debian.org/debian sid main contrib non-free
```
Then we update APT cache and directly install Caffe. Note, the cpu version and
the cuda version cannot be installed at the same time.
```
# apt update
# apt install [ caffe-cpu | caffe-cuda ]
```
It should work out of box.

#### Customizing caffe packages

Some users may need to customize the Caffe package. Here is a brief
guide of producing the customized `.deb` packages.

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
$ debuild -B -j4                                      # build caffe with 4 parallel jobs (similar to make -j4)
[ ... building ...]
$ debc                                                # optional, if you want to check the package contents
$ sudo debi                                           # optional, install the generated packages
```
The resulting deb packages can be found under the parent directory of the source tree.

### Source installation

Source installation under Debian/unstable is similar to that of Ubuntu, but
here is a more elegant way to pull caffe build dependencies:
```
$ sudo apt build-dep [ caffe-cpu | caffe-cuda ]
```
Note, this requires a `deb-src` entry in your `/etc/apt/sources.list`.

### Notes

* Consider re-compiling OpenBLAS locally with optimization flags for sake of
performance. This is highly recommended if you are writing a paper.

* If you are installing `caffe-cuda`, APT will automatically pull some of the
CUDA packages and the nvidia driver packages. Please take care if you have
manually installed or hacked nvidia driver or CUDA toolkit or any other
related stuff, because in this case it may fail.

* If you encountered any problem when installing `caffe-*`, please report bug
to Debian via Debian's bug tracking system. See https://www.debian.org/Bugs/ .

* Additionally, a manpage (`man caffe`) and a bash complementation script
(`caffe <TAB><TAB>`, `caffe train <TAB><TAB>`) are provided.
Both of the two files are still not merged into caffe master.

* The python interface is Python 3 version: `python3-caffe-{cpu,cuda}`.
No plan to support python2.

## FAQ

* where is caffe-cudnn?

CUDNN library seems not redistributable currently. If you really want the
caffe-cudnn deb packages, the workaround is to install cudnn by yourself,
and hack the packaging scripts, then build your customized package.

* I installed the CPU version, How can I switch to the CUDA version?

`sudo apt install caffe-cuda`, apt's dependency resolver is smart enough to deal with this.

* Where is the examples, the models and other documentation stuff?

```
sudo apt install caffe-doc
dpkg -L caffe-doc
```
