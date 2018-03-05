import sys
from prebuild_common import variant_types, pointer_variant_types, variant_enable_flags, proto_types, float_types, float_types_no_half, unsigned_int_types

path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/test_macros.hpp', 'w')

var_types = list(proto_types.keys())

header.write('namespace caffe {\n')

header.write('template<typename TypeParam> class CPUDevice;\n')
header.write('template<typename TypeParam> class GPUDevice;\n')

# TestDtypes
header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in variant_enable_flags.keys():
        header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
    if not li == 0:
        header.write(',')
    header.write(var_types[i] + '\n')
    li += 1
    if var_types[i] in variant_enable_flags.keys():
        header.write('#endif\n')
header.write('> TestDtypes;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in float_types:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write(var_types[i] + '\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesFloat;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in float_types_no_half:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write(var_types[i] + '\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesFloatNoHalf;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in unsigned_int_types:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write(var_types[i] + '\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesInteger;\n')

# TestDtypesAndDevices
header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in variant_enable_flags.keys():
        header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
    if not li == 0:
        header.write(',')
    header.write('CPUDevice<' + var_types[i] + '>\n')
    header.write('#ifndef CPU_ONLY\n')
    header.write(',GPUDevice<' + var_types[i] + '>\n')
    header.write('#endif  // CPU_ONLY\n')
    li += 1
    if var_types[i] in variant_enable_flags.keys():
        header.write('#endif\n')
header.write('> TestDtypesAndDevices;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in float_types:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write('CPUDevice<' + var_types[i] + '>\n')
        header.write('#ifndef CPU_ONLY\n')
        header.write(',GPUDevice<' + var_types[i] + '>\n')
        header.write('#endif  // CPU_ONLY\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesFloatAndDevices;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in float_types_no_half:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write('CPUDevice<' + var_types[i] + '>\n')
        header.write('#ifndef CPU_ONLY\n')
        header.write(',GPUDevice<' + var_types[i] + '>\n')
        header.write('#endif  // CPU_ONLY\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesFloatNoHalfAndDevices;\n')

header.write('typedef ::testing::Types<\n')
li = 0
for i in range(0, len(var_types)):
    if var_types[i] in unsigned_int_types:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write('CPUDevice<' + var_types[i] + '>\n')
        header.write('#ifndef CPU_ONLY\n')
        header.write(',GPUDevice<' + var_types[i] + '>\n')
        header.write('#endif  // CPU_ONLY\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesIntegerAndDevices;\n')

header.write('}  // namespace caffe\n')
