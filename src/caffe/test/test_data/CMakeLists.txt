# copy *.py
file(GLOB_RECURSE files "${PROJECT_SOURCE_DIR}/src/caffe/test/test_data/*.py")
foreach(file ${files})
  # get file name
  get_filename_component(name ${file} NAME)

  # get folder name
  get_filename_component(path ${file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${file} DESTINATION src/caffe/test/test_data/)
endforeach()

# copy *.h5
file(GLOB_RECURSE files "${PROJECT_SOURCE_DIR}/src/caffe/test/test_data/*.h5")
foreach(file ${files})
  # get file name
  get_filename_component(name ${file} NAME)

  # get folder name
  get_filename_component(path ${file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${file} DESTINATION src/caffe/test/test_data/)
endforeach()

# copy *.txt
file(GLOB_RECURSE files "${PROJECT_SOURCE_DIR}/src/caffe/test/test_data/*.txt")
foreach(file ${files})
  # get file name
  get_filename_component(name ${file} NAME)

  # get folder name
  get_filename_component(path ${file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${file} DESTINATION src/caffe/test/test_data/)
endforeach()
