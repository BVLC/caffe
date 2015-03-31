file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
    
  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
    
  add_executable(${name} ${source_file})
  target_link_libraries(${name} ${Caffe_LINK})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION bin)

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standartaized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${CAffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()

# copy shell scripts
file(GLOB_RECURSE examples_sh "${PROJECT_SOURCE_DIR}/examples/*.sh")
foreach(sh_file ${examples_sh})
  # get file name
  get_filename_component(name ${sh_file} NAME)

  # get folder name
  get_filename_component(path ${sh_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(PROGRAMS ${sh_file} DESTINATION examples/${folder}/)
endforeach()

# copy *.prototxt
file(GLOB_RECURSE examples_prototxt "${PROJECT_SOURCE_DIR}/examples/*.prototxt")
foreach(prototxt_file ${examples_prototxt})
  # get file name
  get_filename_component(name ${prototxt_file} NAME)

  # get folder name
  get_filename_component(path ${prototxt_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${prototxt_file} DESTINATION examples/${folder}/)
endforeach()
