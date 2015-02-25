FILE(REMOVE_RECURSE
  "../../../test/test.testbin.pdb"
  "../../../test/test.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
