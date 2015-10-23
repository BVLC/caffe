__author__ = 'pittnuts'
import caffe
from google.protobuf import text_format

class CaffeProtoParser:
    def readProtoSolverFile(self):
        solver_config = caffe.proto.caffe_pb2.SolverParameter()
        #TODO how to read proto file?
        return self._readProtoTxtFile(self.filepath, solver_config)
    #enddef

    def readProtoNetFile(self):
        net_config = caffe.proto.caffe_pb2.NetParameter()
        #TODO how to read proto file?
        return self._readProtoTxtFile(self.filepath, net_config)
    #enddef

    def readBlobProto(self):
        blob = caffe.proto.caffe_pb2.BlobProto()
        #TODO how to read proto file?
        return self._readProtoBinFile(self.filepath, blob)
    #enddef

    def _readProtoTxtFile(self, filepath, parser_object):

        file = open(filepath, "r")

        if not file:
            raise self.ProcessException("ERROR (" + filepath + ")!")


        text_format.Merge(str(file.read()), parser_object)
        file.close()
        return parser_object

    def _readProtoBinFile(self, filepath, parser_object):

        file = open(filepath, "rb")

        if not file:
            raise self.ProcessException("ERROR (" + filepath + ")!")


        parser_object.ParseFromString(file.read())
        file.close()
        return parser_object

    def __init__(self,filepath):
        self.filepath = filepath
