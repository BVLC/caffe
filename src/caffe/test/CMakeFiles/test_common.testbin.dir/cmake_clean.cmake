FILE(REMOVE_RECURSE
  "../../../test/test_common.testbin.pdb"
  "../../../test/test_common.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_common.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
