# copy *.prototxt
file(GLOB_RECURSE models_prototxt "${PROJECT_SOURCE_DIR}/models/*.prototxt")
foreach(prototxt_file ${models_prototxt})
  # get file name
  get_filename_component(name ${prototxt_file} NAME)

  # get folder name
  get_filename_component(path ${prototxt_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${prototxt_file} DESTINATION models/${folder}/)
endforeach()

# copy *.md
file(GLOB_RECURSE models_md "${PROJECT_SOURCE_DIR}/models/*.md")
foreach(md_file ${models_md})
  # get file name
  get_filename_component(name ${md_file} NAME)

  # get folder name
  get_filename_component(path ${md_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${md_file} DESTINATION models/${folder}/)
endforeach()

