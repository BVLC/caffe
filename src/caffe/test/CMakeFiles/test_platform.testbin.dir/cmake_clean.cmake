FILE(REMOVE_RECURSE
  "../../../test/test_platform.testbin.pdb"
  "../../../test/test_platform.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_platform.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
