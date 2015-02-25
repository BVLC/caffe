FILE(REMOVE_RECURSE
  "../../../test/test_random_number_generator.testbin.pdb"
  "../../../test/test_random_number_generator.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_random_number_generator.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
