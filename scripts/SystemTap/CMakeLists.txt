# Installation/customization of system taps scripts
if(USE_SYSTEMTAP)
  # system Tap fails with OpenCV (no idea why)
  if(USE_OPENCV)
    message(FATAL "SystemTap is not working when OpenCV is supported!")
  endif()

  # Copy profile.stp and put absolute path to binary and nn lib
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/profile_selected_function.stp DESTINATION ${CMAKE_BINARY_DIR})
  file(READ ${CMAKE_BINARY_DIR}/profile_selected_function.stp PROFILE_STP_STRING)
  string(REPLACE "[DIRNAME_OF_CAFFE_TOOL_BINARY]" "\"${CMAKE_BINARY_DIR}/tools/caffe\"" PROFILE_STP_STRING ${PROFILE_STP_STRING})
  string(REPLACE "[DIRNAME_OF_LIBCAFFE]" "\"${CMAKE_BINARY_DIR}/lib/libcaffe.so\"" PROFILE_STP_STRING ${PROFILE_STP_STRING})
  file(WRITE ${CMAKE_BINARY_DIR}/profile_selected_function.stp ${PROFILE_STP_STRING})
endif()
