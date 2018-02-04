import sys
from prebuild_common import variant_types, pointer_variant_types, variant_enable_flags, proto_types

path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/blob_creator.hpp', 'w')

header.write('// Automatically generated file, DO NOT EDIT!\n')
header.write('#include "caffe/common.hpp"\n')
header.write('#include "caffe/blob.hpp"\n')
header.write('#include "caffe/proto/caffe.pb.h"\n')

# Inline helper function for creating layers
header.write('namespace caffe {\n')

header.write('inline shared_ptr<BlobBase> CreateBlob(Device* dev, DataType data_type) {\n')
header.write('switch(data_type) {\n')

var_types = list(proto_types.keys())

for i in range(0, len(var_types)):
    if var_types[i] in list(variant_enable_flags.keys()):
        header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
    header.write('case ' + proto_types[var_types[i]] + ': {\n')
    header.write('return make_shared<Blob<' + var_types[i] + '> >(dev);\n')
    header.write('}\n')  # case
    if var_types[i] in list(variant_enable_flags.keys()):
        header.write('#endif\n')
header.write('default: { LOG(FATAL) << "Data types not enabled or supported"; }\n')
header.write('}\n') # switch
header.write('return nullptr;\n')
header.write('}\n') # CreateLayer

header.write('}  // namespace caffe\n')

header.close()