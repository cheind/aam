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

#ifndef AAM_MAP_H
#define AAM_MAP_H

#include <aam/types.h>
#include <opencv2/core/core.hpp>

namespace aam {

    /** Convert Eigen matrix to OpenCV without copying */
    inline cv::Mat toOpenCVHeader(Eigen::Ref<MatrixX> m)
    {
        int depth = cv::DataType<Scalar>::depth;
        return cv::Mat(m.rows(), m.cols(), CV_MAKETYPE(depth, 1), m.data());
    }

    /** Convert Eigen matrix to OpenCV without copying */
    template<class Scalar>
    inline cv::Mat_<Scalar> toOpenCVHeader(Eigen::Ref< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > m)
    {
        int depth = cv::DataType<Scalar>::depth;
        return cv::Mat(m.rows(), m.cols(), CV_MAKETYPE(depth, 1), m.data());
    }

    /** Convert OpenCV matrix to Eigen without copying. */
    inline MapMatrixX toEigenHeader(cv::Mat m)
    {
        // Only single channel matrices make sense in Eigen.

        cv::Mat m1 = m.reshape(1);
        int depth = cv::DataType<Scalar>::depth;
        CV_Assert(m1.type() == CV_MAKETYPE(depth, 1));

        int outerStride = m1.step[0] / sizeof(Scalar);

        return MapMatrixX(m1.ptr<Scalar>(), m1.rows, m1.cols, Eigen::Stride < Eigen::Dynamic, Eigen::Dynamic>(outerStride, 1));
    }

    /** Convert OpenCV matrix to Eigen without copying. */
    template<class Scalar>
    inline typename EigenMatrixMapTraits<Scalar>::MapType toEigenHeader(cv::Mat_<Scalar> m)
    {
        // Only single channel matrices make sense in Eigen.
        cv::Mat m1 = m.reshape(1);

        int depth = cv::DataType<Scalar>::depth;
        CV_Assert(m1.type() == CV_MAKETYPE(depth, 1));

        int outerStride = m1.step[0] / sizeof(Scalar);

        return EigenMatrixMapTraits<Scalar>::MapType(m1.ptr<Scalar>(), m1.rows, m1.cols, Eigen::Stride < Eigen::Dynamic, Eigen::Dynamic>(outerStride, 1));
    }



}

#endif