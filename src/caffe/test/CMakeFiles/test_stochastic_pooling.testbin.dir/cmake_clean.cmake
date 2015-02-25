FILE(REMOVE_RECURSE
  "../../../test/test_stochastic_pooling.testbin.pdb"
  "../../../test/test_stochastic_pooling.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_stochastic_pooling.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
