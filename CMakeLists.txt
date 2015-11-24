cmake_minimum_required(VERSION 2.8)

project(aam)

if (WIN32)
	#set(OpenCV_STATIC true)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS")
endif()

set(EIGEN_INCLUDE_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR} "inc")

# Library

add_library(aam
    inc/aam/fwd.h
    inc/aam/types.h
	inc/aam/map.h
	inc/aam/pca.h
	inc/aam/aam.h
	inc/aam/io.h
	inc/aam/show.h
	src/pca.cpp	
	src/io.cpp
	src/show.cpp
)
	
target_link_libraries(aam ${OpenCV_LIBRARIES})
	
# Samples

add_executable(aam_shapemorph examples/shapemorph.cpp)
target_link_libraries(aam_shapemorph aam ${OpenCV_LIBRARIES})

add_executable(aam_showtrainingdata examples/showtrainingdata.cpp)
target_link_libraries(aam_showtrainingdata aam ${OpenCV_LIBRARIES})

# Tests

add_executable(tests
    tests/catch.hpp
	tests/random_distributions.h
    tests/pca.cpp
	tests/map.cpp
)
target_link_libraries(tests aam ${OpenCV_LIBRARIES})