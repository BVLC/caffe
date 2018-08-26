import os
import unittest

from google.protobuf import text_format

import caffe.draw
from caffe.proto import caffe_pb2

def getFilenames():
    """Yields files in the source tree which are Net prototxts."""
    result = []

    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..'))
    assert os.path.exists(root_dir)

    for dirname in ('models', 'examples'):
        dirname = os.path.join(root_dir, dirname)
        assert os.path.exists(dirname)
        for cwd, _, filenames in os.walk(dirname):
            for filename in filenames:
                filename = os.path.join(cwd, filename)
                if filename.endswith('.prototxt') and 'solver' not in filename:
                    yield os.path.join(dirname, filename)


class TestDraw(unittest.TestCase):
    def test_draw_net(self):
        for filename in getFilenames():
            net = caffe_pb2.NetParameter()
            with open(filename) as infile:
                text_format.Merge(infile.read(), net)
            caffe.draw.draw_net(net, 'LR')


if __name__ == "__main__":
    unittest.main()
