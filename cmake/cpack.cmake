INCLUDE(InstallRequiredSystemLibraries)

if("${CMAKE_SYSTEM}" MATCHES "Linux")
  find_program(DPKG_PROGRAM dpkg)
  if(EXISTS ${DPKG_PROGRAM})
    list(APPEND CPACK_GENERATOR "DEB")
		#------------------------------------------------------------------------------
		# General config
		#------------------------------------------------------------------------------
		SET(CPACK_PACKAGE_NAME "caffe-dev")
		SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Caffe development files")
		SET(CPACK_PACKAGE_VENDOR "Suat Gedikli")
		SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
		SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
		SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CPack_test")
		SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/debian/postinst")
    SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libgoogle-glog0 (>=0.3.3-1)")
#    SET(CPACK_DEBIAN_PACKAGE_PREDEPENDS "libgoogle-glog0 (>=0.3.3-1), libboost-thread1.54.0 (>=1.54.0-4), libhdf5-7 (>=1.8.11-5), liblmdb0 (>=0.9.10-1), libleveldb1 (>=1.15.0-2), libopencv-core2.4 (>=2.4.8), libopencv-highgui2.4 (>=2.4.8), libatlas3-base (>=3.10.1-4), libboost-python1.54.0 (>=1.54.0-4), libgflags-dev")
    SET(CPACK_DEBIAN_PACKAGE_PREDEPENDS "libgoogle-glog-dev (>=0.3.3-1), libprotobuf-dev (>=2.5.0-9), libboost-thread1.54.0 (>=1.54.0-4), libhdf5-dev (>=1.8.11-5), liblmdb-dev (>=0.9.10-1), libleveldb-dev (>=1.15.0-2), libopencv-core2.4 (>=2.4.8), libhighgui-dev (>=2.4.8), libatlas-base-dev (>=3.10.1-4), libboost-python1.54-dev (>=1.54.0-4), libgflags-dev (>=2.0-1), libsnappy-dev (>=1.1.0-1)")

		#SET(CPACK_STRIP_FILES "bin/MyExecutable")
		#SET(CPACK_SOURCE_STRIP_FILES "")
		#SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
		#SET(CPACK_COMPONENTS_ALL "${CPACK_COMPONENTS_ALL} doc")

		#------------------------------------------------------------------------------
		# assemble package name and version number
		#------------------------------------------------------------------------------
		SET(CPACK_BUILD_NO "" CACHE STRING "The build id")
		if(CPACK_BUILD_NO)
		#  message("CPACK_BUILD_NO: Build number '${CPACK_BUILD_NO}' specified.")
			SET(CPACK_VERSION ${CAFFE_VERSION}~b${CPACK_BUILD_NO})
		else()
		#  message("CPACK_BUILD_NO: No CPACK_BUILD_NO specified. Treating as personal build.")
			SET(CPACK_VERSION ${CAFFE_VERSION}~$ENV{USER})
		endif()
		
		SET(CPACK_PACKAGE_VERSION ${CPACK_VERSION})
    SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Suat Gedikli (suat.gedikli@gmail.com)")
		SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
		SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Suat Gedikli (suat.gedikli@gmail.com)")
		SET(CPACK_DEBIAN_PACKAGE_PRIORITY "extra")
		SET(CPACK_DEBIAN_PACKAGE_SECTION "Other")
		SET(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_INTERNAL_CONFIG_REQUIRES}")
		SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}~${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
		message("CPACK_PACKAGE_FILE_NAME: '${CPACK_PACKAGE_FILE_NAME}'")
		INCLUDE(CPack)
  endif(EXISTS ${DPKG_PROGRAM})
endif()




