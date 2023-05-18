cmake_minimum_required(VERSION 3.10)
SET(CMAKE_CXX_COMPILER mpicxx)   # use CC for okeanos

project(CA3DMM_TESTS CXX)

SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14  -g3 -Wall -O0")

include_directories(.)
include_directories(../)

add_executable(generate_data
		../densematgen.cpp
		../densematgen.h
		../utils.h
    generate_data.cpp)

