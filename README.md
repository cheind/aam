# About this library

**Active Appearance Models** is a small C++ library providing implementations for training, fitting and tracking active appearance models.

# Active Appearance Models

Implementation is based on [1].

# Building from source
**Active Appearance Models** requires the following pre-requisites
 - [CMake](www.cmake.org) - for generating cross platform build files
 - [OpenCV](www.opencv.org) - for image processing related functions
 - [Eigen](eigen.tuxfamily.org/) - for sparse linear system solving
 
To build from source
 - Point CMake to the cloned git repository
 - Click CMake Configure
 - Point EIGEN_INCLUDE_DIR to the location of Eigen header directory
 - Point OpenCV_DIR to the directory containing the file ´OpenCVConfig.cmake´
 - Click CMake Generate
 
Although **Active Appearance Models** should build across multiple platforms and architectures, tests are carried out on these systems
 - Windows 8/10 MSVC10 x86
 - OS X 10.10 XCode 6.x

If the build should fail for a specific platform, don't hesitate to create an issue. 

# Test Databases
**Active Appearance Models** can be trained on the following test databases

 - IMM Face Database [2] http://www.imm.dtu.dk/~aam/datasets/datasets.html

# References

[1] Matthews, Iain, and Simon Baker. "Active appearance models revisited." International Journal of Computer Vision 60.2 (2004): 135-164.

[2] Nordstrøm, Michael M., et al. The IMM face database-an annotated dataset of 240 face images. Technical University of Denmark, DTU Informatics, Building 321, 2004.

# License
```
This file is part of Active Appearance Models (AAM).

Copyright Christoph Heindl 2015
Copyright Sebastian Zambal 2015

AAM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AAM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AAM.  If not, see <http://www.gnu.org/licenses/>.
```
