FILE(REMOVE_RECURSE
  "../../../test/test_data_layer.testbin.pdb"
  "../../../test/test_data_layer.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_data_layer.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
