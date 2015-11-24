/**
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
*/

#ifndef AAM_TYPES_H
#define AAM_TYPES_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <vector>

namespace aam {

    /** Type traits for mapping matrices */
    template<class Scalar>
    struct EigenMatrixMapTraits {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
        typedef Eigen::Map< MatrixType, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > MapType;
    };
    
    /** Precision */
    typedef float Scalar;
    
    /** Generic MxN matrix set to storage order compatible with OpenCV matrices. */
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixX;

    /** Mapped MxN matrix with external storage. */
    typedef EigenMatrixMapTraits<Scalar>::MapType MapMatrixX;

    /** Generic 2x2 matrix set to storage order compatible with OpenCV matrices. */
    typedef Eigen::Matrix<Scalar, 2, 2, Eigen::RowMajor> Matrix2;
    
    /** Generic 1xM row vector. */
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVectorX;

    /** Generic 1x2 row vector. */
    typedef Eigen::Matrix<Scalar, 1, 2> RowVector2;

    /** Generic 1x3 row vector. */
    typedef Eigen::Matrix<Scalar, 1, 3> RowVector3;


    struct TrainingSet {
    public:
        std::vector<cv::Mat> images;       // training images
        cv::Mat shapes;    // NxM matrix with N (nb. rows) = number of training examples, M (nb. cols) = number of coordinates per training shape
        cv::Mat contour;  // optional: contours defined on the object (this data is just for visualization, not needed for actual AAM)
    };

    struct ActiveAppearanceModel {
        //mean
        //eigenVectors
        //eigenvalues
        cv::Mat1i triangleIndices;  // Nx3 Matrix of landmark points connectivity (i.e. triangles from Delaunay triangulation)
    };

}

#endif