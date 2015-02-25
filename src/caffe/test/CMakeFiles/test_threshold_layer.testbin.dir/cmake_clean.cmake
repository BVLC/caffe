FILE(REMOVE_RECURSE
  "../../../test/test_threshold_layer.testbin.pdb"
  "../../../test/test_threshold_layer.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_threshold_layer.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
