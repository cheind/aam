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

#include <aam/traits.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <vector>

namespace aam {
   
    /** Precision */
    typedef float Scalar;
    
    /** Generic MxN matrix set to storage order compatible with OpenCV matrices. */
    typedef AamMatrixTraits<Scalar>::MatrixType MatrixX;

    /** Mapped MxN matrix with external storage. */
    typedef AamMatrixTraits<Scalar>::MatrixMapType MapMatrixX;

    /** Generic 2x2 matrix set to storage order compatible with OpenCV matrices. */
    typedef AamMatrixTraits<Scalar, 2, 2>::MatrixType Matrix2;
    
    /** Generic 1xM row vector. */
    typedef AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixType RowVectorX;

    /** Generic 1xM row vector of integer. */
    typedef AamMatrixTraits<int, 1, Eigen::Dynamic>::MatrixType RowVectorXi;

    /** Generic 1x2 row vector. */
    typedef AamMatrixTraits<Scalar, 1, 2>::MatrixType RowVector2;

    /** Generic 1x3 row vector. */
    typedef AamMatrixTraits<Scalar, 1, 3>::MatrixType RowVector3;


    struct TrainingSet {
    public:
        std::vector<cv::Mat> images;       // training images
        cv::Mat shapes;    // NxM matrix with N (nb. rows) = number of training examples, M (nb. cols) = number of coordinates per training shape
        cv::Mat contour;  // optional: contours defined on the object (this data is just for visualization, not needed for actual AAM)
    };

}

#endif