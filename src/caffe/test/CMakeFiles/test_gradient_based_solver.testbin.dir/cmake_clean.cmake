FILE(REMOVE_RECURSE
  "../../../test/test_gradient_based_solver.testbin.pdb"
  "../../../test/test_gradient_based_solver.testbin"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test_gradient_based_solver.testbin.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
