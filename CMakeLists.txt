cmake_minimum_required(VERSION 3.10)
SET(CMAKE_CXX_COMPILER mpicxx)   # use CC for okeanos

project(CA3DMM CXX)

SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")

include_directories(.)

add_executable(ca3dmm
		densematgen.cpp
		densematgen.h
		utils.h
        ca3dmm.cpp)

target_link_libraries(ca3dmm ${MPI_C_LIBRARIES})
target_compile_options(ca3dmm PRIVATE ${MPI_C_COMPILE_FLAGS})
