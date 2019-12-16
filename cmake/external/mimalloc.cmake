
set(mimalloc_root_dir ${CMAKE_CURRENT_LIST_DIR}/mimalloc)

add_definitions(-DUSE_MIMALLOC) # used in ONNXRuntime
include_directories(${mimalloc_root_dir}/include)

option(MI_OVERRIDE "" OFF)
option(MI_BUILD_TESTS "" OFF)

add_subdirectory(${mimalloc_root_dir} EXCLUDE_FROM_ALL)
set_target_properties(mimalloc-static PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
  set_target_properties(mimalloc-static PROPERTIES COMPILE_FLAGS "/wd4389 /wd4201 /wd4244 /wd4565")
endif()
