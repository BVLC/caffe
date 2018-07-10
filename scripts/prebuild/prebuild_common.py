variant_types  = ['int8_t', 'int16_t', 'int32_t', 'int64_t',
                  'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'half_fp',
                  'float', 'double']

float_types = ['half_fp', 'float', 'double']
float_types_no_half = ['float', 'double']
int_types = ['int8_t', 'int16_t', 'int32_t', 'int64_t'
             'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t']
unsigned_int_types = ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t']

pointer_variant_types = variant_types + ['bool', 'char', 'void']

variant_enable_flags = dict()

#variant_enable_flags['pac_bin'] = 'USE_PACKED_BINARY'
variant_enable_flags['int8_t'] = 'USE_INT_QUANT_8'
variant_enable_flags['int16_t'] = 'USE_INT_QUANT_16'
variant_enable_flags['int32_t'] = 'USE_INT_QUANT_32'
variant_enable_flags['int64_t'] = 'USE_INT_QUANT_64'
variant_enable_flags['uint8_t'] = 'USE_INT_QUANT_8'
variant_enable_flags['uint16_t'] = 'USE_INT_QUANT_16'
variant_enable_flags['uint32_t'] = 'USE_INT_QUANT_32'
variant_enable_flags['uint64_t'] = 'USE_INT_QUANT_64'
variant_enable_flags['half_fp'] = 'USE_HALF'
variant_enable_flags['float'] = 'USE_SINGLE'
variant_enable_flags['double'] = 'USE_DOUBLE'

proto_types = dict()

#proto_types['pac_bin'] = 'PACKED_BINARY'
proto_types['half_fp'] = 'CAFFE_HALF'
proto_types['float'] = 'CAFFE_FLOAT'
proto_types['double'] = 'CAFFE_DOUBLE'
proto_types['uint8_t'] = 'CAFFE_INT8_QUANTIZED'
proto_types['uint16_t'] = 'CAFFE_INT16_QUANTIZED'
proto_types['uint32_t'] = 'CAFFE_INT32_QUANTIZED'
proto_types['uint64_t'] = 'CAFFE_INT64_QUANTIZED'
