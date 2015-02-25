FILE(REMOVE_RECURSE
  "../../../test/test_net.testbin.pdb"
  "../../../test/test_net.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_net.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
