FILE(REMOVE_RECURSE
  "../../../test/test_upgrade_proto.testbin.pdb"
  "../../../test/test_upgrade_proto.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_upgrade_proto.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
