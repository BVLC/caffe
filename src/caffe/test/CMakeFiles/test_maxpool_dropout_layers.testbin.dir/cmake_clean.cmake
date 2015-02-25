FILE(REMOVE_RECURSE
  "../../../test/test_maxpool_dropout_layers.testbin.pdb"
  "../../../test/test_maxpool_dropout_layers.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_maxpool_dropout_layers.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
