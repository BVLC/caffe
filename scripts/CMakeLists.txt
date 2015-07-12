# copy *.sh
file(GLOB scripts_sh "${PROJECT_SOURCE_DIR}/scripts/*.sh")
foreach(sh_file ${scripts_sh})
  # get file name
  get_filename_component(name ${sh_file} NAME)

  # get folder name
  get_filename_component(path ${sh_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${sh_file} DESTINATION scripts/)
endforeach()

# copy *.sh
file(GLOB scripts_sh "${PROJECT_SOURCE_DIR}/scripts/travis/*.sh")
foreach(sh_file ${scripts_sh})
  # get file name
  get_filename_component(name ${sh_file} NAME)

  # get folder name
  get_filename_component(path ${sh_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${sh_file} DESTINATION scripts/${folder}/)
endforeach()

# copy *.py
file(GLOB scripts_py "${PROJECT_SOURCE_DIR}/scripts/*.py")
foreach(py_file ${scripts_py})
  # get file name
  get_filename_component(name ${py_file} NAME)

  # get folder name
  get_filename_component(path ${py_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  install(FILES ${py_file} DESTINATION scripts/)
endforeach()
