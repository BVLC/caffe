add_executable(pretune_convert pretune_convert.cpp)
target_link_libraries(pretune_convert "-lOpenCL")

set(PRETUNE_CONVERT ${CMAKE_CURRENT_BINARY_DIR}/pretune_convert PARENT_SCOPE)