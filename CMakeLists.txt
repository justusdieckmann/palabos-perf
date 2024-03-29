cmake_minimum_required(VERSION 3.20)
project(palabos_perf)

set(CMAKE_CXX_STANDARD 20)

set(BUILD_HDF5 OFF)


# Compiler flags
# Append flags: set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ...")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    message("GCC.")
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wnon-virtual-dtor")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DPLB_DEBUG -O0")
    set(CMAKE_CXX_FLAGS_TEST "-g -DPLB_DEBUG -DPLB_REGRESSION -O0 -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_LINKER_FLAGS_TEST "${CMAKE_CXX_LINKER_FLAGS_TEST} -fno-omit-frame-pointer -fsanitize=address")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message("Clang.")
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall  -Wextra -Wnon-virtual-dtor")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DPLB_DEBUG -O0")
    set(CMAKE_CXX_FLAGS_TEST "-g -DPLB_DEBUG -DPLB_REGRESSION -O0 -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_LINKER_FLAGS_TEST "${CMAKE_CXX_LINKER_FLAGS_TEST} -fno-omit-frame-pointer -fsanitize=address")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    message("MSVC.")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /Ot /GS- /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/DPLB_DEBUG")
    set(CMAKE_CXX_FLAGS_TEST "/Ox /Ot /GS- /GL /DPLB_DEBUG /DPLB_REGRESSION")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message( FATAL_ERROR "CXX compiler not recognized. CMake will quit." )
endif()

option(ENABLE_MPI "Enable MPI" ON)
if(ENABLE_MPI)
    message("Enabling MPI")
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        #set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
        include_directories(${MPI_CXX_INCLUDE_PATH})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
        add_definitions(-DPLB_MPI_PARALLEL)
    endif()
endif()

if(WIN32)
    option(ENABLE_POSIX "Enable POSIX" OFF)
else()
    option(ENABLE_POSIX "Enable POSIX" ON)
endif()

if(ENABLE_POSIX)
    message("Enabling POSIX")
    add_definitions(-DPLB_USE_POSIX)
endif()

if(APPLE)
    add_definitions(-DPLB_MAC_OS_X)
endif()

if(WIN32 OR CYGWIN)
    add_definitions(-DPLB_WINDOWS)
endif()

###############################################################################
# Palabos Library
###############################################################################

include_directories("palabos/src")
include_directories("palabos/externalLibraries")
include_directories("palabos/externalLibraries/Eigen3")

file(GLOB_RECURSE PALABOS_SRC "palabos/src/*.cpp")
file(GLOB_RECURSE EXT_SRC "palabos/externalLibraries/tinyxml/*.cpp")

add_library(palabos STATIC ${PALABOS_SRC} ${EXT_SRC})

###############################################################################

add_executable(palabos_perf main.cpp)

# Link with the following libraries
target_link_libraries(palabos_perf palabos)
if(ENABLE_MPI)
    target_link_libraries(palabos_perf ${MPI_CXX_LIBRARIES})
endif()