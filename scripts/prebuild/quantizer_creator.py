import sys
from prebuild_common import variant_types, pointer_variant_types, variant_enable_flags, proto_types

path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/quantizer_creator.hpp', 'w')

header.write('// Automatically generated file, DO NOT EDIT!\n')
header.write('#include "caffe/common.hpp"\n')
header.write('#include "caffe/proto/caffe.pb.h"\n')
header.write('#include "caffe/quantizer.hpp"\n')

# Inline helper function for creating Quantizers
header.write('namespace caffe {\n')

header.write('inline shared_ptr<QuantizerBase> CreateQuantizer(QuantizerParameter quant_param) {\n')
header.write('switch (quant_param.input_data_type()) {\n')

var_types = list(proto_types.keys())

for i in range(0, len(var_types)):
    if var_types[i] in list(variant_enable_flags.keys()):
        header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
    header.write('case ' + proto_types[var_types[i]] + ': {\n')
    header.write('switch (quant_param.output_data_type()) {\n')
    for j in range(0, len(var_types)):
        if var_types[j] in list(variant_enable_flags.keys()):
            header.write('#if defined(' + variant_enable_flags[var_types[j]] + ')\n')
        header.write('case ' + proto_types[var_types[j]] + ': {\n')
        header.write('return make_shared<Quantizer<' + var_types[i]  + ', ' + var_types[j] + '> >(quant_param);\n')
        header.write('}\n')  # case
        if var_types[j] in list(variant_enable_flags.keys()):
            header.write('#endif\n')
    header.write('default: { LOG(FATAL) << "Data types not enabled or supported"; }\n')
    header.write('}\n')  # switch
    header.write('}\n')  # case
    if var_types[i] in list(variant_enable_flags.keys()):
        header.write('#endif\n')
header.write('default: { LOG(FATAL) << "Data types not enabled or supported"; }\n')
header.write('}\n')  # switch
header.write('return nullptr;\n')
header.write('}\n') # CreateQuantizer

header.write('}  // namespace caffe\n')

header.close()