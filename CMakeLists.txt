# Building docs script
# Requirements:
#   sudo apt-get install doxygen texlive ruby-dev
#   sudo gem install jekyll execjs therubyracer

if(NOT BUILD_docs OR NOT DOXYGEN_FOUND)
  return()
endif()

#################################################################################################
# Gather docs from <root>/examples/**/readme.md
function(gather_readmes_as_prebuild_cmd target gathered_dir root)
  set(full_gathered_dir ${root}/${gathered_dir})

  file(GLOB_RECURSE readmes ${root}/examples/readme.md ${root}/examples/README.md)
  foreach(file ${readmes})
    # Only use file if it is to be included in docs.
    file(STRINGS ${file} file_lines REGEX "include_in_docs: true")

    if(file_lines)
      # Since everything is called readme.md, rename it by its dirname.
      file(RELATIVE_PATH file ${root} ${file})
      get_filename_component(folder ${file} PATH)
      set(new_filename ${full_gathered_dir}/${folder}.md)

      # folder value might be like <subfolder>/readme.md. That's why make directory.
      get_filename_component(new_folder ${new_filename} PATH)
      add_custom_command(TARGET ${target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${new_folder}
        COMMAND ln -sf ${root}/${file} ${new_filename}
        COMMENT "Creating symlink ${new_filename} -> ${root}/${file}"
        WORKING_DIRECTORY ${root} VERBATIM)
    endif()
  endforeach()
endfunction()

################################################################################################
# Gather docs from examples/*.ipynb and add YAML front-matter.
function(gather_notebooks_as_prebuild_cmd target gathered_dir root)
  set(full_gathered_dir ${root}/${gathered_dir})

  if(NOT PYTHON_EXECUTABLE)
    message(STATUS "Python interpeter is not found. Can't include *.ipynb files in docs. Skipping...")
    return()
  endif()

  file(GLOB_RECURSE notebooks ${root}/examples/*.ipynb)
  foreach(file ${notebooks})
    file(RELATIVE_PATH file ${root} ${file})
    set(new_filename ${full_gathered_dir}/${file})

    get_filename_component(new_folder ${new_filename} PATH)
    add_custom_command(TARGET ${target} PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${new_folder}
      COMMAND ${PYTHON_EXECUTABLE} scripts/copy_notebook.py ${file} ${new_filename}
      COMMENT "Copying notebook ${file} to ${new_filename}"
      WORKING_DIRECTORY ${root} VERBATIM)
  endforeach()

  set(${outputs_var} ${outputs} PARENT_SCOPE)
endfunction()

################################################################################################
########################## [ Non macro part ] ##################################################

# Gathering is done at each 'make doc'
file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/docs/gathered)

# Doxygen config file path
set(DOXYGEN_config_file ${PROJECT_SOURCE_DIR}/.Doxyfile CACHE FILEPATH "Doxygen config file")

# Adding docs target
add_custom_target(docs COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_config_file}
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                       COMMENT "Launching doxygen..." VERBATIM)

# Gathering examples into docs subfolder
gather_notebooks_as_prebuild_cmd(docs docs/gathered ${PROJECT_SOURCE_DIR})
gather_readmes_as_prebuild_cmd(docs docs/gathered  ${PROJECT_SOURCE_DIR})

# Auto detect output directory
file(STRINGS ${DOXYGEN_config_file} config_line REGEX "OUTPUT_DIRECTORY[ \t]+=[^=].*")
if(config_line)
  string(REGEX MATCH "OUTPUT_DIRECTORY[ \t]+=([^=].*)" __ver_check "${config_line}")
  string(STRIP ${CMAKE_MATCH_1} output_dir)
  message(STATUS "Detected Doxygen OUTPUT_DIRECTORY: ${output_dir}")
else()
  set(output_dir ./doxygen/)
  message(STATUS "Can't find OUTPUT_DIRECTORY in doxygen config file. Try to use default: ${output_dir}")
endif()

if(NOT IS_ABSOLUTE ${output_dir})
  set(output_dir ${PROJECT_SOURCE_DIR}/${output_dir})
  get_filename_component(output_dir ${output_dir} ABSOLUTE)
endif()

# creates symlink in docs subfolder to code documentation built by doxygen
add_custom_command(TARGET docs POST_BUILD VERBATIM
                   COMMAND ln -sfn "${output_dir}/html" doxygen
                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
                   COMMENT "Creating symlink ${PROJECT_SOURCE_DIR}/docs/doxygen -> ${output_dir}/html")

# for quick launch of jekyll
add_custom_target(jekyll COMMAND jekyll serve -w -s . -d _site --port=4000
                         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
                         COMMENT "Launching jekyll..." VERBATIM)
