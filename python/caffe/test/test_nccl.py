import sys
import unittest

import caffe


class TestNCCL(unittest.TestCase):

    def test_newuid(self):
        """
        Test that NCCL uids are of the proper type
        according to python version
        """
        if caffe.has_nccl():
            uid = caffe.NCCL.new_uid()
            if sys.version_info.major >= 3:
                self.assertTrue(isinstance(uid, bytes))
            else:
                self.assertTrue(isinstance(uid, str))
