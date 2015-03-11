from caffe import pycuda_util

if pycuda_util.pycuda_available:
    import unittest
    import tempfile
    import os

    import numpy as np

    import caffe
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

    caffe_include_dirs = pycuda_util.caffe_include_dirs

    kernel_set_value = """
#include <caffe/util/device_alternate.hpp>
__global__ void set_value(float *x, float value, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        x[i] = value;
    }
}
    """
    kernel_axpb = """
#include <caffe/util/device_alternate.hpp>
__global__ void axpb(float a, float *x, float b, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        x[i] = a * x[i] + b;
    }
}
    """

    def python_net_file():
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write("""name: 'blobonly'
            input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
            """)
            return f.name

    def get_test_gpuid_from_makefile():
        path_makefile = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..', '..', '..', 'Makefile.config'))
        try:
            test_gpuid = filter(
                lambda x: x.startswith("TEST_GPUID"),
                open(path_makefile, 'r'))[0]
            test_gpuid = int(test_gpuid.split(' ')[2])
            return test_gpuid
        except IndexError:
            pass
        except ValueError:
            pass
        return 0

    class TestPyCuda(unittest.TestCase):

        def setUp(self):
            test_gpuid = get_test_gpuid_from_makefile()
            caffe.set_mode_gpu()
            caffe.set_device(test_gpuid)
            net_proto = python_net_file()
            net = caffe.Net(net_proto, caffe.TRAIN)
            self.blob = net.blobs["data"]
            os.remove(net_proto)
            self.rng = np.random.RandomState(313)

        def test_blob_to_gpuarray(self):
            with pycuda_util.caffe_cuda_context():
                self.blob.data_as_pycuda_gpuarray()
                self.blob.diff_as_pycuda_gpuarray()

        def test_pycuda_set_data(self):
            with pycuda_util.caffe_cuda_context():
                mod = SourceModule(
                    kernel_set_value, include_dirs=caffe_include_dirs)
                set_value = mod.get_function("set_value")
                set_value(
                    self.blob.data_as_pycuda_gpuarray(), np.float32(5),
                    np.int32(self.blob.count),
                    ** pycuda_util.block_and_grid(self.blob.count))
            for v in self.blob.data.flat:
                self.assertEqual(v, np.float32(5))

        def test_pycuda_set_diff(self):
            a = np.float32(5.0)
            with pycuda_util.caffe_cuda_context():
                mod = SourceModule(
                    kernel_set_value, include_dirs=caffe_include_dirs)
                set_value = mod.get_function("set_value")
                set_value(
                    self.blob.diff_as_pycuda_gpuarray(), a,
                    np.int32(self.blob.count),
                    **pycuda_util.block_and_grid(self.blob.count))
            for v in self.blob.diff.flat:
                self.assertEqual(v, np.float32(5))

        def test_pycuda_axpb(self):
            a = np.float32(-2.0)
            b = np.float32(3.5)
            with pycuda_util.caffe_cuda_context():
                mod = SourceModule(
                    kernel_axpb, include_dirs=caffe_include_dirs)
                axpb = mod.get_function("axpb")
                base = self.rng.rand(*self.blob.shape)
                self.blob.data[...] = base
                axpb(
                    a, self.blob.data_as_pycuda_gpuarray(),
                    b, np.int32(self.blob.count),
                    **pycuda_util.block_and_grid(self.blob.count))
            for v, w in zip(self.blob.data.flat, base.flat):
                self.assertEqual(v, a * np.float32(w) + b)

        def test_scikits_cuda_cublas_sgemm(self):
            try:
                import scikits.cuda.cublas as cublas
            except ImportError:
                return
            foo = self.rng.randn(10, 9, 8)
            self.blob.data[...] = foo
            with pycuda_util.caffe_cuda_context():
                h = caffe.cublas_handle()
                a = self.blob.data_as_pycuda_gpuarray()
                a = a.reshape(10, 9 * 8)
                b = gpuarray.to_gpu(
                    self.rng.randn(9 * 8, 2).astype(np.float32))
                c = gpuarray.zeros((10, 2), np.float32)
                cublas.cublasSgemm(h, 'N', 'N', 2, 10, 9 * 8,
                    1.0, b.gpudata, 2, a.gpudata, 9*8, 0., c.gpudata, 2)
                a = a.get()
                b = b.get()
                c = c.get()
            c2 = np.dot(a, b)
            assert np.allclose(c, c2, rtol=0, atol=1e-5)
