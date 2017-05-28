### Running an official image

You can run one of the automatic [builds](https://hub.docker.com/r/bvlc/caffe). E.g. for the CPU version:

`docker run -ti bvlc/caffe:cpu caffe --version`

or for GPU support (You need a CUDA 8.0 capable driver and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)):

`nvidia-docker run -ti bvlc/caffe:gpu caffe --version`

You might see an error about libdc1394, ignore it.

### Docker run options

By default caffe runs as root, thus any output files, e.g. snapshots, will be owned
by root. It also runs by default in a container-private folder.

You can change this using flags, like user (-u), current directory, and volumes (-w and -v).
E.g. this behaves like the usual caffe executable:

`docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu caffe train --solver=example_solver.prototxt`

Containers can also be used interactively, specifying e.g. `bash` or `ipython`
instead of `caffe`.

```
docker run -ti bvlc/caffe:cpu ipython
import caffe
...
```

The caffe build requirements are included in the container, so this can be used to
build and run custom versions of caffe. Also, `caffe/python` is in PATH, so python
utilities can be used directly, e.g. `draw_net.py`, `classify.py`, or `detect.py`.

### Building images yourself

Examples:

`docker build -t caffe:cpu cpu`

`docker build -t caffe:gpu gpu`

You can also build Caffe and run the tests in the image:

`docker run -ti caffe:cpu bash -c "cd /opt/caffe/build; make runtest"`
