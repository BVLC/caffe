FILE(REMOVE_RECURSE
  "../../../test/test_syncedmem.testbin.pdb"
  "../../../test/test_syncedmem.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_syncedmem.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
