FILE(REMOVE_RECURSE
  "../../../test/test_internal_thread.testbin.pdb"
  "../../../test/test_internal_thread.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_internal_thread.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
