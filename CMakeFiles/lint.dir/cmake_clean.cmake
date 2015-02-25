FILE(REMOVE_RECURSE
  "CMakeFiles/lint"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/lint.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
