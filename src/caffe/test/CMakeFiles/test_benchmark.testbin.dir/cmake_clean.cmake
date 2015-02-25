FILE(REMOVE_RECURSE
  "../../../test/test_benchmark.testbin.pdb"
  "../../../test/test_benchmark.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_benchmark.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
