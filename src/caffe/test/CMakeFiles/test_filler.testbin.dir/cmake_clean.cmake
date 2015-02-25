FILE(REMOVE_RECURSE
  "../../../test/test_filler.testbin.pdb"
  "../../../test/test_filler.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_filler.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
