import sys

path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/macros.hpp', 'w')

variant_types  = ['bool', 'char', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
                  'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'half_fp',
                  'float', 'double']

pointer_variant_types = variant_types + ['void']

variant_enable_flags = dict()

variant_enable_flags['int8_t'] = 'USE_GPU_INT_QUANT_8'
variant_enable_flags['int16_t'] = 'USE_GPU_INT_QUANT_16'
variant_enable_flags['int32_t'] = 'USE_GPU_INT_QUANT_32'
variant_enable_flags['int64_t'] = 'USE_GPU_INT_QUANT_64'
variant_enable_flags['uint8_t'] = 'USE_GPU_INT_QUANT_8'
variant_enable_flags['uint16_t'] = 'USE_GPU_INT_QUANT_16'
variant_enable_flags['uint32_t'] = 'USE_GPU_INT_QUANT_32'
variant_enable_flags['uint64_t'] = 'USE_GPU_INT_QUANT_64'
variant_enable_flags['half_fp'] = 'USE_GPU_HALF'
variant_enable_flags['float'] = 'USE_GPU_SINGLE'
variant_enable_flags['double'] = 'USE_GPU_DOUBLE'

header.write('// Automatically generated file, DO NOT EDIT!\n')
header.write('#include <boost/preprocessor/seq/for_each.hpp>\n')
header.write('#include <boost/preprocessor/seq/enum.hpp>\n')
header.write('#include <boost/preprocessor/seq/elem.hpp>\n')
header.write('#include <boost/preprocessor/seq/for_each_product.hpp>\n')
header.write('#include <boost/preprocessor/seq/to_tuple.hpp>\n')
header.write('#include <boost/preprocessor/tuple/to_seq.hpp>\n')

header.write('#define PP_EMPTY(...)\n')
header.write('#define PP_DEFER(...) __VA_ARGS__ PP_EMPTY()\n')
header.write('#define PP_OBSTRUCT(...) __VA_ARGS__ PP_DEFER(PP_EMPTY)()\n')
header.write('#define PP_EXPAND(...) __VA_ARGS__\n')

header.write('#define VARIANT_TYPES\\\n')
for i in range(0, len(variant_types)):
    header.write('  (' + variant_types[i] + ')')
    if (i == len(variant_types) - 1):
        header.write('\n')
    else:
        header.write('\\\n')
    

# 1 template class instantiation
for var_type_1 in variant_types:
    flags = []
    if var_type_1 in variant_enable_flags.keys():
        flag = variant_enable_flags[var_type_1]
        if not flag in flags:
            flags.append(flag)
    if len(flags) > 0:
        header.write('#if ')
    for i in range(0, len(flags)):
        header.write('defined(' + flags[i] + ')')
        if (i == len(flags) - 1):
            header.write('\n')
        else:
            header.write(' && ')
    header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '(CLASSNAME)\\\n')
    header.write('  template class CLASSNAME<' + var_type_1 + '>;\n')
    if len(flags) > 0:
        header.write('#else\n')
        header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '(CLASSNAME)\n')
        header.write('#endif\n')

# 2 template class instantiation
for var_type_1 in variant_types:
    for var_type_2 in variant_types:
        flags = []
        if var_type_1 in variant_enable_flags.keys():
            flag = variant_enable_flags[var_type_1]
            if not flag in flags:
                flags.append(flag)
        if var_type_2 in variant_enable_flags.keys():
            flag = variant_enable_flags[var_type_2]
            if not flag in flags:
                flags.append(flag)
        if len(flags) > 0:
            header.write('#if ')
        for i in range(0, len(flags)):
            header.write('defined(' + flags[i] + ')')
            if (i == len(flags) - 1):
                header.write('\n')
            else:
                header.write(' && ')
        header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\\\n')
        header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        if len(flags) > 0:
            header.write('#else\n')
            header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\n')
            header.write('#endif\n')

# 3 template class instantiation
for var_type_1 in variant_types:
    for var_type_2 in variant_types:
        for var_type_3 in variant_types:
            flags = []
            if var_type_1 in variant_enable_flags.keys():
                flag = variant_enable_flags[var_type_1]
                if not flag in flags:
                    flags.append(flag)
            if var_type_2 in variant_enable_flags.keys():
                flag = variant_enable_flags[var_type_2]
                if not flag in flags:
                    flags.append(flag)
            if var_type_3 in variant_enable_flags.keys():
                flag = variant_enable_flags[var_type_3]
                if not flag in flags:
                    flags.append(flag)
            if len(flags) > 0:
                header.write('#if ')
            for i in range(0, len(flags)):
                header.write('defined(' + flags[i] + ')')
                if (i == len(flags) - 1):
                    header.write('\n')
                else:
                    header.write(' && ')
            header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define INSTANTIATE_LAYER_GPU_FORWARD_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  template void CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>::Forward_gpu(\\\n')
            header.write('      const vector<Blob<' + var_type_2 + '>*>& bottom,\\\n')
            header.write('      const vector<Blob<' + var_type_3 + '>*>& top);\n')
            header.write('#define INSTANTIATE_LAYER_GPU_BACKWARD_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  template void CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>::Backward_gpu(\\\n')
            header.write('      const vector<Blob<' + var_type_3 + '>*>& top,\\\n')
            header.write('      const vector<bool>& propagate_down,\\\n')
            header.write('      const vector<Blob<' + var_type_2 + '>*>& bottom);\n')
            header.write('#define REGISTER_LAYER_CREATOR_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(TYPE, CREATOR)\\\n')
            header.write('  static LayerRegisterer<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '> ')
            header.write('g_creator_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '_##TYPE(#TYPE, CREATOR<'+ var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>);\n')
            if len(flags) > 0:
                header.write('#else\n')
                header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\n')
                header.write('#define INSTANTIATE_LAYER_GPU_FORWARD_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\n')
                header.write('#define INSTANTIATE_LAYER_GPU_BACKWARD_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\n')
                header.write('#define REGISTER_LAYER_CREATOR_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(TYPE, CREATOR)\n')
                header.write('#endif\n')


header.write('#define CART_PROD_JOIN_US_2T(R, SEQ_X)\\\n')
header.write('  (BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_SEQ_ELEM(0,SEQ_X),_),BOOST_PP_SEQ_ELEM(1,SEQ_X)))\n')

header.write('#define CART_SET_JOIN_US_2T(T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH_PRODUCT(CART_PROD_JOIN_US_2T, (T1) (T2))\n')

header.write('#define CART_PROD_JOIN_US_3T(R, SEQ_X)\\\n')
header.write('  (BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_SEQ_ELEM(0,SEQ_X),_),BOOST_PP_CAT(BOOST_PP_SEQ_ELEM(1,SEQ_X),_)),BOOST_PP_SEQ_ELEM(2,SEQ_X)))\n')

header.write('#define CART_SET_JOIN_US_3T(T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH_PRODUCT(CART_PROD_JOIN_US_3T, (T1) (T2) (T3))\n')

# 1 template class instantiation
header.write('#define INSTANTIATE_CLASS_1T_HELPER(R, CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_, T1)(CLASSNAME)\n')

header.write('#define INSTANTIATE_CLASS_1T(CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_1T_HELPER, CLASSNAME, T1)\n')

# 2 template class instantiation
header.write('#define INSTANTIATE_CLASS_2T_HELPER(R, CLASSNAME, T1T2)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_, T1T2)(CLASSNAME)\n')

header.write('#define INSTANTIATE_CLASS_2T(CLASSNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_2T_HELPER, CLASSNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

# 3 template class instantiation
header.write('#define INSTANTIATE_CLASS_3T_HELPER(R, CLASSNAME, T1T2T3)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_, T1T2T3)(CLASSNAME)\n')

header.write('#define INSTANTIATE_CLASS_3T(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_1T_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

# Instantiate pointer class
header.write('#define INSTANTIATE_POINTER_CLASS(CLASSNAME)\\\n')
header.write('  char gInstantiationGuard##CLASSNAME;\\\n')
for i in range(0, len(pointer_variant_types)):
    var_type = pointer_variant_types[i]
    header.write('  template class CLASSNAME<' + var_type + '>;\\\n')
    header.write('  template class CLASSNAME<const ' + var_type + '>;')
    if (i == len(pointer_variant_types) - 1):
        header.write('\n')
    else:
        header.write('\\\n')

# 3 template GPU functions
header.write('#define INSTANTIATE_LAYER_GPU_FORWARD_HELPER(R, CLASSNAME, T1T2T3)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_LAYER_GPU_FORWARD_, T1T2T3)(CLASSNAME)\n')

header.write('#define INSTANTIATE_LAYER_GPU_FORWARD(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_LAYER_GPU_FORWARD_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_LAYER_GPU_BACKWARD_HELPER(R, CLASSNAME, T1T2T3)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_LAYER_GPU_BACKWARD_, T1T2T3)(CLASSNAME)\n')

header.write('#define INSTANTIATE_LAYER_GPU_BACKWARD(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_LAYER_GPU_BACKWARD_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_LAYER_GPU_FUNCS(CLASSNAME, T1, T2, T3)\\\n')
header.write('  INSTANTIATE_LAYER_GPU_FORWARD(CLASSNAME, T1, T2, T3);\\\n')
header.write('  INSTANTIATE_LAYER_GPU_BACKWARD(CLASSNAME, T1, T2, T3);\n')

# 3 template layer register / creator
header.write('#define REGISTER_LAYER_CLASS_INST_HELPER(R, TYPE, T1T2T3)\\\n')
header.write('  BOOST_PP_CAT(REGISTER_LAYER_CREATOR_, T1T2T3)(TYPE, Creator_##TYPE##Layer)\n')

header.write('#define REGISTER_LAYER_CLASS(TYPE)\\\n')
header.write('  template<typename Dtype, typename MItype, typename MOtype>\\\n')
header.write('  shared_ptr<Layer<Dtype, MItype, MOtype> >\\\n')
header.write('    Creator_##TYPE##Layer(const LayerParameter& param)\\\n')
header.write('  { return shared_ptr<Layer<Dtype, MItype, MOtype> > (new TYPE##Layer<Dtype, MItype, MOtype>(param)); }\n')

header.write('#define REGISTER_LAYER_CLASS_INST(TYPE, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(REGISTER_LAYER_CLASS_INST_HELPER, TYPE, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define REGISTER_LAYER_CREATOR_HELPER(R, TYPE_CREATOR, T1T2T3)\\\n')
header.write('  PP_DEFER(BOOST_PP_CAT(REGISTER_LAYER_CREATOR_, T1T2T3))(BOOST_PP_SEQ_ELEM(0, TYPE_CREATOR), BOOST_PP_SEQ_ELEM(1, TYPE_CREATOR))\n')

header.write('#define REGISTER_LAYER_CREATOR(TYPE, CREATOR, T1, T2, T3)\\\n')
header.write('  PP_EXPAND(BOOST_PP_SEQ_FOR_EACH(REGISTER_LAYER_CREATOR_HELPER, (TYPE)(CREATOR), CART_SET_JOIN_US_3T(T1, T2, T3)))\n')

# Stub GPU
header.write('#ifdef CPU_ONLY  // CPU-only Caffe.\n')
header.write('#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."\n')

header.write('#define STUB_GPU(CLASSNAME)\\\n')
header.write('  template <typename Dtype, typename MItype, typename MOtype>\\\n')
header.write('  void CLASSNAME<Dtype, MItype, MOtype>::Forward_gpu(\\\n')
header.write('    const vector<Blob<MItype>*>& bottom,\\\n')
header.write('    const vector<Blob<MOtype>*>& top) { NO_GPU; }\\\n')
header.write('  template <typename Dtype, typename MItype, typename MOtype>\\\n')
header.write('  void CLASSNAME<Dtype, MItype, MOtype>::Backward_gpu(\\\n')
header.write('    const vector<Blob<MOtype>*>& top,\\\n')
header.write('    const vector<bool>& propagate_down,\\\n')
header.write('    const vector<Blob<MItype>*>& bottom) { NO_GPU; }\n')

header.write('#define STUB_GPU_FORWARD(CLASSNAME, FUNCNAME)\\\n')
header.write('  template <typename Dtype, typename MItype, typename MOtype>\\\n')
header.write('  void CLASSNAME<Dtype, MItype, MOtype>::FUNCNAME##_##gpu(\\\n')
header.write('    const vector<Blob<MItype>*>& bottom,\\\n')
header.write('    const vector<Blob<MOtype>*>& top) { NO_GPU; }\n')

header.write('#define STUB_GPU_BACKWARD(CLASSNAME, FUNCNAME)\\\n')
header.write('  template <typename Dtype, typename MItype, typename MOtype>\\\n')
header.write('  void CLASSNAME<Dtype, MItype, MOtype>::FUNCNAME##_##gpu(\\\n')
header.write('  const vector<Blob<MOtype>*>& top,\\\n')
header.write('  const vector<bool>& propagate_down,\\\n')
header.write('  const vector<Blob<MItype>*>& bottom) { NO_GPU; }\n')
header.write('#endif\n')

header.close() 
