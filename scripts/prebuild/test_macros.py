import sys
from prebuild_common import variant_types, pointer_variant_types, variant_enable_flags, proto_types, float_types, float_types_no_half, int_types

path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/test_macros.hpp', 'w')

var_types = proto_types.keys()

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
    if var_types[i] in float_types_no_half:
        if var_types[i] in variant_enable_flags.keys():
            header.write('#if defined(' + variant_enable_flags[var_types[i]] + ')\n')
        if not li == 0:
            header.write(',')
        header.write(var_types[i] + '\n')
        li += 1
        if var_types[i] in variant_enable_flags.keys():
            header.write('#endif\n')
header.write('> TestDtypesInteger;\n')