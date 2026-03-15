cmake_minimum_required(VERSION 3.16)
project(ort_concorde_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED DynamoRIO_DIR)
  set(DynamoRIO_DIR "/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/cmake" CACHE PATH "DynamoRIO cmake directory")
endif()

find_package(DynamoRIO REQUIRED)
find_package(Threads REQUIRED)

configure_DynamoRIO_global(ON ON)
set(DynamoRIO_RPATH OFF)

if(NOT TARGET snappy AND EXISTS "/usr/lib64/libsnappy.so.1")
  add_library(snappy SHARED IMPORTED GLOBAL)
  set_target_properties(snappy PROPERTIES IMPORTED_LOCATION "/usr/lib64/libsnappy.so.1")
endif()

if(NOT TARGET lz4 AND EXISTS "/usr/lib64/liblz4.so.1")
  add_library(lz4 SHARED IMPORTED GLOBAL)
  set_target_properties(lz4 PROPERTIES IMPORTED_LOCATION "/usr/lib64/liblz4.so.1")
endif()

get_target_property(DRMEMTRACE_ANALYZER_LIBS drmemtrace_analyzer INTERFACE_LINK_LIBRARIES)
if(DRMEMTRACE_ANALYZER_LIBS)
  string(REPLACE "/usr/aarch64-linux-gnu/lib/libpthread.so" "Threads::Threads" DRMEMTRACE_ANALYZER_LIBS "${DRMEMTRACE_ANALYZER_LIBS}")
  string(REPLACE "/usr/aarch64-linux-gnu/lib/libz.a" "/usr/lib64/libz.a" DRMEMTRACE_ANALYZER_LIBS "${DRMEMTRACE_ANALYZER_LIBS}")
  set_target_properties(drmemtrace_analyzer PROPERTIES INTERFACE_LINK_LIBRARIES "${DRMEMTRACE_ANALYZER_LIBS}")
endif()

add_executable(drmemtrace_pseudoview drmemtrace_pseudoview.cpp)
configure_DynamoRIO_standalone(drmemtrace_pseudoview)
use_DynamoRIO_drmemtrace(drmemtrace_pseudoview)
target_link_libraries(drmemtrace_pseudoview drmemtrace_analyzer drdecode)

add_executable(concorde_native_analyzer concorde_native_analyzer.cpp)
configure_DynamoRIO_standalone(concorde_native_analyzer)
use_DynamoRIO_drmemtrace(concorde_native_analyzer)
target_link_libraries(concorde_native_analyzer drmemtrace_analyzer drdecode)
