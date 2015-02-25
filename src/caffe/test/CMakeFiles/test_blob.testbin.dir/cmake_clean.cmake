FILE(REMOVE_RECURSE
  "../../../test/test_blob.testbin.pdb"
  "../../../test/test_blob.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_blob.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
