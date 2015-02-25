FILE(REMOVE_RECURSE
  "../../../test/test_protobuf.testbin.pdb"
  "../../../test/test_protobuf.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_protobuf.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
