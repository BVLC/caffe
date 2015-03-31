# copy *.cl
file(GLOB_RECURSE OpenCL_sources "${PROJECT_SOURCE_DIR}/src/caffe/layers/OpenCL/*.cl")
foreach(cl_file ${OpenCL_sources})
  # get file name
  get_filename_component(name ${cl_file} NAME)

  # get folder name
  get_filename_component(path ${cl_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${cl_file} DESTINATION src/caffe/layers/OpenCL/)
endforeach()
