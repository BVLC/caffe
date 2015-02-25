FILE(REMOVE_RECURSE
  "../../../test/test_math_functions.testbin.pdb"
  "../../../test/test_math_functions.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_math_functions.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
