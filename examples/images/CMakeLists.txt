# copy *.jpg
file(GLOB_RECURSE files "${PROJECT_SOURCE_DIR}/examples/images/*.jpg")
foreach(file ${files})
  # get file name
  get_filename_component(name ${file} NAME)

  # get folder name
  get_filename_component(path ${file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${file} DESTINATION examples/images)
endforeach()
