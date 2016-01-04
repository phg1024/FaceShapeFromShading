cmake_minimum_required(VERSION 2.8.11)

project(FaceShapeFromShading)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

set(CMAKE_AUTOUIC ON)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Boost
find_package(Boost COMPONENTS filesystem timer program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES} -lboost_filesystem -lboost_system)

# OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

# Eigen
set(EIGEN_INCLUDE_DIR /usr/include/eigen3)
include_directories(${EIGEN_INCLUDE_DIR})

# MKL
set(MKL_INCLUDE_DIRS "/home/phg/SDKs/intel/composer_xe/include")
message([] ${MKL_INCLUDE_DIRS} )
include_directories(${MKL_INCLUDE_DIRS})
set(MKLROOT "/home/phg/SDKs/intel/composer_xe_2015/mkl")
set(MKLLIBS_DIRS "${MKLROOT}/lib/intel64/")
link_directories(${MKLLIBS_DIRS})
set(MKLLIBS "-Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group -liomp5 -ldl -lpthread -lm")

# PhGLib
include_directories("/home/phg/SDKs/PhGLib/include")
link_directories("/home/phg/SDKs/PhGLib/lib")
set(PhGLib "-lPhGLib")

# Qt5
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)

# Ceres solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
link_libraries(${CERES_LIBRARIES})

# Face shape from shading program
add_executable(FaceShapeFromShading faceshapefromshading.cpp common.h OffscreenMeshVisualizer.cpp OffscreenMeshVisualizer.h utils.h)
target_link_libraries(FaceShapeFromShading
                      multilinearmodel
                      basicmesh
                      tensor
                      Qt5::Core
                      Qt5::Widgets
                      Qt5::OpenGL
                      ${MKLLIBS}
                      ${PhGLib})
link_directories(MultilinearReconstruction)

add_subdirectory(MultilinearReconstruction)
add_subdirectory(tests)
