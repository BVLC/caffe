FILE(REMOVE_RECURSE
  "../../../test/test_im2col_kernel.testbin.pdb"
  "../../../test/test_im2col_kernel.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_im2col_kernel.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
