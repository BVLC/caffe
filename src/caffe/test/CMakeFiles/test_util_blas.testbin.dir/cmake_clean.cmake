FILE(REMOVE_RECURSE
  "../../../test/test_util_blas.testbin.pdb"
  "../../../test/test_util_blas.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_util_blas.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
