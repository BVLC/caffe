FIND_PACKAGE( PackageHandleStandardArgs )

set(CUDA_NVRTC_HEADERS, "")

find_path(TEST_PATH2 utility)

message("****TEST2_PATH****")
message("${TEST2_PATH}")

file(GLOB_RECURSE CUDA_NVRTC_HEADERS /usr/include/*/utility)

message("****CUDA_NVRTC_HEADERS****")
message("${CUDA_NVRTC_HEADERS}")
message("${CUDA_NVRTC_HEADERS_SEARCH_PATHS}")



message("****CUDA_NVRTC_HEADERS****")
message("${CUDA_NVRTC_HEADERS}")


MARK_AS_ADVANCED(
  CUDA_NVRTC_HEADERS
)