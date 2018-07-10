import sys
from prebuild_common import variant_types, pointer_variant_types, variant_enable_flags, proto_types


path = ''
if (len(sys.argv) > 1):
    path = sys.argv[1]

header = open(path + '/macros.hpp', 'w')

header.write('// Automatically generated file, DO NOT EDIT!\n')
header.write('#include <boost/preprocessor/seq/for_each.hpp>\n')
header.write('#include <boost/preprocessor/seq/enum.hpp>\n')
header.write('#include <boost/preprocessor/seq/elem.hpp>\n')
header.write('#include <boost/preprocessor/seq/for_each_product.hpp>\n')
header.write('#include <boost/preprocessor/seq/to_tuple.hpp>\n')
header.write('#include <boost/preprocessor/tuple/to_seq.hpp>\n')

# Helpers for template class function instantiation
header.write('template<class Pmf>\n')
header.write('struct pmf_sig{};\n')
header.write('template<class Pmf>\n')
header.write('using pmf_sig_t=typename pmf_sig<Pmf>::type;\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...)>{\n')
header.write('  using type=R(Args...);\n')
header.write('};\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...) const>{\n')
header.write('  using type=R(Args...) const;\n')
header.write('};\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...) const&>{\n')
header.write('  using type=R(Args...) const&;\n')
header.write('};\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...) const&&>{\n')
header.write('  using type=R(Args...) const&&;\n')
header.write('};\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...) &&>{\n')
header.write('  using type=R(Args...) &&;\n')
header.write('};\n')
header.write('template<class R, class T, class...Args>\n')
header.write('struct pmf_sig<R(T::*)(Args...) &>{\n')
header.write('  using type=R(Args...) &;\n')
header.write('};\n')

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
   
header.write('#define PROTO_TYPES\\\n')
for i in range(0, len(list(proto_types.keys()))):
    header.write('  (' + list(proto_types.keys())[i] + ')')
    if (i == len(list(proto_types.keys())) - 1):
        header.write('\n')
    else:
        header.write('\\\n')
             
header.write('#define POINTER_VARIANT_TYPES\\\n')
for i in range(0, len(pointer_variant_types)):
    header.write('  (' + pointer_variant_types[i] + ')')
    if (i == len(pointer_variant_types) - 1):
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
    header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '(CLASSNAME)\\\n')
    header.write('  template class CLASSNAME<' + var_type_1 + '>;\n')
    header.write('#define EXTERN_CLASS_' + var_type_1 + '(CLASSNAME)\\\n')
    header.write('  extern template class CLASSNAME<' + var_type_1 + '>;\n')
    header.write('#define INSTANTIATE_FUNC_' + var_type_1 + '(FUNCNAME)\\\n')
    header.write('  template decltype(FUNCNAME<' + var_type_1 + '>) FUNCNAME<' + var_type_1 + '>;\n')
    header.write('#define INSTANTIATE_CLASST_FUNC_' + var_type_1 + '(CLASSNAME, FUNCNAME)\\\n')
    header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + '>::FUNCNAME)> CLASSNAME<' + var_type_1 + '>::FUNCNAME;\n')
    header.write('#define INSTANTIATE_CLASS_FUNCT_' + var_type_1 + '(CLASSNAME, FUNCNAME)\\\n')
    header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + '>)> CLASSNAME::FUNCNAME<' + var_type_1 + '>;\n')
    if len(flags) > 0:
        header.write('#if ')
    for i in range(0, len(flags)):
        header.write('defined(' + flags[i] + ')')
        if (i == len(flags) - 1):
            header.write('\n')
        else:
            header.write(' && ')
    header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '(CLASSNAME)\\\n')
    header.write('  template class CLASSNAME<' + var_type_1 + '>;\n')
    header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '(CLASSNAME)\\\n')
    header.write('  extern template class CLASSNAME<' + var_type_1 + '>;\n')
    header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '(FUNCNAME)\\\n')
    header.write('  template decltype(FUNCNAME<' + var_type_1 + '>) FUNCNAME<' + var_type_1 + '>;\n')
    header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '(CLASSNAME, FUNCNAME)\\\n')
    header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + '>::FUNCNAME)> CLASSNAME<' + var_type_1 + '>::FUNCNAME;\n')
    header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '(CLASSNAME, FUNCNAME)\\\n')
    header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + '>)> CLASSNAME::FUNCNAME<' + var_type_1 + '>;\n')
    header.write('#define REGISTER_SOLVER_CREATOR_' + var_type_1 + '(TYPE, CREATOR)\\\n')
    header.write('  static SolverRegisterer<' + var_type_1 + '> ')
    header.write('g_creator_' + var_type_1 + '_##TYPE(#TYPE, CREATOR<' + var_type_1 + '>);\n')
    if len(flags) > 0:
        header.write('#else\n')
        header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '(CLASSNAME)\n')
        header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '(CLASSNAME)\n')
        header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '(FUNCNAME)\n')
        header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '(CLASSNAME, FUNCNAME)\n')
        header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '(CLASSNAME, FUNCNAME)\n')
        header.write('#define REGISTER_SOLVER_CREATOR_' + var_type_1 + '(TYPE, CREATOR)\n')
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
        header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\\\n')
        header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define EXTERN_CLASS_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\\\n')
        header.write('  extern template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define INSTANTIATE_FUNC_' + var_type_1 + '_' + var_type_2 + '(FUNCNAME)\\\n')
        header.write('  template decltype(FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>) FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define INSTANTIATE_CLASST_FUNC_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\\\n')
        header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>::FUNCNAME)> CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>::FUNCNAME;\n')
        header.write('#define INSTANTIATE_CLASS_FUNCT_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\\\n')
        header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>)> CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        if len(flags) > 0:
            header.write('#if ')
        for i in range(0, len(flags)):
            header.write('defined(' + flags[i] + ')')
            if (i == len(flags) - 1):
                header.write('\n')
            else:
                header.write(' && ')
        header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\\\n')
        header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\\\n')
        header.write('  extern template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '(FUNCNAME)\\\n')
        header.write('  template decltype(FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>) FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\\\n')
        header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>::FUNCNAME)> CLASSNAME<' + var_type_1 + ',' + var_type_2 + '>::FUNCNAME;\n')
        header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\\\n')
        header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>)> CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + '>;\n')
        if len(flags) > 0:
            header.write('#else\n')
            header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\n')
            header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME)\n')
            header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '(FUNCNAME)\n')
            header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\n')
            header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '_' + var_type_2 + '(CLASSNAME, FUNCNAME)\n')
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
            header.write('#define INSTANTIATE_CLASS_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define EXTERN_CLASS_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  extern template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define INSTANTIATE_FUNC_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(FUNCNAME)\\\n')
            header.write('  template decltype(FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>) FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define INSTANTIATE_CLASST_FUNC_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\\\n')
            header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>::FUNCNAME)> CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>::FUNCNAME;\n')
            header.write('#define INSTANTIATE_CLASS_FUNCT_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\\\n')
            header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>)> CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>;\n')
            if len(flags) > 0:
                header.write('#if ')
            for i in range(0, len(flags)):
                header.write('defined(' + flags[i] + ')')
                if (i == len(flags) - 1):
                    header.write('\n')
                else:
                    header.write(' && ')
            header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\\\n')
            header.write('  extern template class CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(FUNCNAME)\\\n')
            header.write('  template decltype(FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>) FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>;\n')
            header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\\\n')
            header.write('  template pmf_sig_t<decltype(&CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>::FUNCNAME)> CLASSNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>::FUNCNAME;\n')
            header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\\\n')
            header.write('  template pmf_sig_t<decltype(&CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>)> CLASSNAME::FUNCNAME<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 +'>;\n')
            header.write('#define REGISTER_LAYER_CREATOR_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(TYPE, CREATOR)\\\n')
            header.write('  static LayerRegisterer<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '> ')
            header.write('g_creator_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '_##TYPE(#TYPE, CREATOR<' + var_type_1 + ',' + var_type_2 + ',' + var_type_3 + '>);\n')
            if len(flags) > 0:
                header.write('#else\n')
                header.write('#define INSTANTIATE_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\n')
                header.write('#define EXTERN_CLASS_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME)\n')
                header.write('#define INSTANTIATE_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(FUNCNAME)\n')
                header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\n')
                header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_' + var_type_1 + '_' + var_type_2 + '_' + var_type_3 + '(CLASSNAME, FUNCNAME)\n')
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

# template class instantiation
header.write('#define INSTANTIATE_CLASS_HELPER(R, CLASSNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_, T)(CLASSNAME)\n')

header.write('#define INSTANTIATE_CLASS_GUARDED_HELPER(R, CLASSNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_GUARDED_, T)(CLASSNAME)\n')

# 1 template class instantiation
header.write('#define INSTANTIATE_CLASS_1T(CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_HELPER, CLASSNAME, T1)\n')

header.write('#define INSTANTIATE_CLASS_1T_GUARDED(CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_GUARDED_HELPER, CLASSNAME, T1)\n')

# 2 template class instantiation
header.write('#define INSTANTIATE_CLASS_2T(CLASSNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_HELPER, CLASSNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

header.write('#define INSTANTIATE_CLASS_2T_GUARDED(CLASSNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_GUARDED_HELPER, CLASSNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

# 3 template class instantiation
header.write('#define INSTANTIATE_CLASS_3T(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_CLASS_3T_GUARDED(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_GUARDED_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')


# template class extern
header.write('#define EXTERN_CLASS_HELPER(R, CLASSNAME, T)\\\n')
header.write('  BOOST_PP_CAT(EXTERN_CLASS_, T)(CLASSNAME)\n')

header.write('#define EXTERN_CLASS_GUARDED_HELPER(R, CLASSNAME, T)\\\n')
header.write('  BOOST_PP_CAT(EXTERN_CLASS_GUARDED_, T)(CLASSNAME)\n')

# 1 template class extern
header.write('#define EXTERN_CLASS_1T(CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_HELPER, CLASSNAME, T1)\n')

header.write('#define EXTERN_CLASS_1T_GUARDED(CLASSNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_GUARDED_HELPER, CLASSNAME, T1)\n')

# 2 template class extern
header.write('#define EXTERN_CLASS_2T(CLASSNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_HELPER, CLASSNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

header.write('#define EXTERN_CLASS_2T_GUARDED(CLASSNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_GUARDED_HELPER, CLASSNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

# 3 template class extern
header.write('#define EXTERN_CLASS_3T(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define EXTERN_CLASS_3T_GUARDED(CLASSNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(EXTERN_CLASS_GUARDED_HELPER, CLASSNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')


# template function instantiation
header.write('#define INSTANTIATE_FUNC_HELPER(R, FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_FUNC_, T)(FUNCNAME)\n')

header.write('#define INSTANTIATE_FUNC_GUARDED_HELPER(R, FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_FUNC_GUARDED_, T)(FUNCNAME)\n')

# 1 template function instantiation
header.write('#define INSTANTIATE_FUNC_1T(FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_HELPER, FUNCNAME, T1)\n')

header.write('#define INSTANTIATE_FUNC_1T_GUARDED(FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_GUARDED_HELPER, FUNCNAME, T1)\n')

# 2 template function instantiation
header.write('#define INSTANTIATE_FUNC_2T(FUNCNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_HELPER, FUNCNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

header.write('#define INSTANTIATE_FUNC_2T_GUARDED(FUNCNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_GUARDED_HELPER, FUNCNAME, CART_SET_JOIN_US_2T(T1, T2))\n')

# 3 template function instantiation
header.write('#define INSTANTIATE_FUNC_3T(FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_HELPER, FUNCNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_FUNC_3T_GUARDED(FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC_GUARDED_HELPER, FUNCNAME, CART_SET_JOIN_US_3T(T1, T2, T3))\n')

# template class function instantiation
header.write('#define INSTANTIATE_CLASST_FUNC_HELPER(R, CLASSNAME_FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASST_FUNC_, T)(BOOST_PP_SEQ_ELEM(0,CLASSNAME_FUNCNAME),BOOST_PP_SEQ_ELEM(1,CLASSNAME_FUNCNAME))\n')

header.write('#define INSTANTIATE_CLASST_FUNC_GUARDED_HELPER(R, CLASSNAME_FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASST_FUNC_GUARDED_, T)(BOOST_PP_SEQ_ELEM(0,CLASSNAME_FUNCNAME),BOOST_PP_SEQ_ELEM(1,CLASSNAME_FUNCNAME))\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_HELPER(R, CLASSNAME_FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_FUNCT_, T)(BOOST_PP_SEQ_ELEM(0,CLASSNAME_FUNCNAME),BOOST_PP_SEQ_ELEM(1,CLASSNAME_FUNCNAME))\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_GUARDED_HELPER(R, CLASSNAME_FUNCNAME, T)\\\n')
header.write('  BOOST_PP_CAT(INSTANTIATE_CLASS_FUNCT_GUARDED_, T)(BOOST_PP_SEQ_ELEM(0,CLASSNAME_FUNCNAME),BOOST_PP_SEQ_ELEM(1,CLASSNAME_FUNCNAME))\n')

# 1 template class function instantiation
header.write('#define INSTANTIATE_CLASST_FUNC_1T(CLASSNAME, FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_HELPER, (CLASSNAME)(FUNCNAME), T1)\n')

header.write('#define INSTANTIATE_CLASST_FUNC_1T_GUARDED(CLASSNAME, FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_GUARDED_HELPER, (CLASSNAME)(FUNCNAME), T1)\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_1T(CLASSNAME, FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_FUNCT_HELPER, (CLASSNAME)(FUNCNAME), T1)\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_1T_GUARDED(CLASSNAME, FUNCNAME, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_FUNCT_GUARDED_HELPER, (CLASSNAME)(FUNCNAME), T1)\n')

# 2 template class function instantiation
header.write('#define INSTANTIATE_CLASST_FUNC_2T(CLASSNAME, FUNCNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_2T(T1, T2))\n')

header.write('#define INSTANTIATE_CLASST_FUNC_2T_GUARDED(CLASSNAME, FUNCNAME, T1, T2)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_GUARDED_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_2T(T1, T2))\n')

# 3 template class function instantiation
header.write('#define INSTANTIATE_CLASST_FUNC_3T(CLASSNAME, FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_CLASST_FUNC_3T_GUARDED(CLASSNAME, FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASST_FUNC_GUARDED_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_3T(CLASSNAME, FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_FUNCT_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_3T(T1, T2, T3))\n')

header.write('#define INSTANTIATE_CLASS_FUNCT_3T_GUARDED(CLASSNAME, FUNCNAME, T1, T2, T3)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS_FUNCT_GUARDED_HELPER, (CLASSNAME)(FUNCNAME), CART_SET_JOIN_US_3T(T1, T2, T3))\n')

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

# 1 template solver register / creator
header.write('#define REGISTER_SOLVER_CLASS_INST_HELPER(R, TYPE, T1)\\\n')
header.write('  BOOST_PP_CAT(REGISTER_SOLVER_CREATOR_, T1)(TYPE, Creator_##TYPE##Solver)\n')

header.write('#define REGISTER_SOLVER_CLASS(TYPE)\\\n')
header.write('  template<typename Dtype>\\\n')
header.write('  Solver<Dtype>* Creator_##TYPE##Solver(const SolverParameter& param, Device* dev)\\\n')
header.write('  { return new TYPE##Solver<Dtype>(param, dev); }\n')

header.write('#define REGISTER_SOLVER_CLASS_INST(TYPE, T1)\\\n')
header.write('  BOOST_PP_SEQ_FOR_EACH(REGISTER_SOLVER_CLASS_INST_HELPER, TYPE, T1)\n')

header.write('#define REGISTER_SOLVER_CREATOR_HELPER(R, TYPE_CREATOR, T1)\\\n')
header.write('  PP_DEFER(BOOST_PP_CAT(REGISTER_SOLVER_CREATOR_, T1))(BOOST_PP_SEQ_ELEM(0, TYPE_CREATOR), BOOST_PP_SEQ_ELEM(1, TYPE_CREATOR))\n')

header.write('#define REGISTER_SOLVER_CREATOR(TYPE, CREATOR, T1)\\\n')
header.write('  PP_EXPAND(BOOST_PP_SEQ_FOR_EACH(REGISTER_SOLVER_CREATOR_HELPER, (TYPE)(CREATOR), T1))\n')

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
