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
    template<class Scalar>
    inline cv::Mat_<Scalar> toOpenCVHeader(typename AamMatrixTraits<Scalar>::MatrixRefType m)
    {
        int depth = cv::DataType<Scalar>::depth;
        return cv::Mat(m.rows(), m.cols(), CV_MAKETYPE(depth, 1), m.data());
    }

    /** Convert OpenCV matrix to Eigen without copying. */
    template<class Scalar>
    inline typename AamMatrixTraits<Scalar>::MatrixMapType toEigenHeader(cv::Mat_<Scalar> m)
    {
        // Convert to single channel
        cv::Mat m1 = m.reshape(1);

        int depth = cv::DataType<Scalar>::depth;
        CV_Assert(m1.type() == CV_MAKETYPE(depth, 1));

        int outerStride = m1.step[0] / sizeof(Scalar);

        return typename AamMatrixTraits<Scalar>::MatrixMapType(m1.ptr<Scalar>(), m1.rows, m1.cols, Eigen::Stride < Eigen::Dynamic, 1>(outerStride, 1));
    }

    /** Convert from interleaved representation to row style */
    template<class Scalar>
    inline typename AamMatrixTraits<Scalar>::MatrixType fromInterleaved(typename AamMatrixTraits<Scalar>::ConstMatrixRefType ibased, int dims = 2)
    {
        typedef typename AamMatrixTraits<Scalar>::MatrixType ReturnType;            

        const ReturnType::Index nRowsInput = ibased.rows();
        const ReturnType::Index nRowsOutput = ibased.cols() / dims;
        const ReturnType::Index nColsOutput = nRowsInput * dims;

        ReturnType r(nRowsOutput, nColsOutput);
        
        for (int y = 0; y < nRowsInput; ++y) {

            auto irow = ibased.row(y);
            auto block = r.block(0, y * dims, nRowsOutput, dims);

            for (int x = 0; x < nRowsOutput; ++x) {
                block.row(x) = irow.segment(x * dims, dims);
            }

        }

        return r;

    }

    /** Convert from per-row to interleaved */
    template<class Scalar>
    inline typename AamMatrixTraits<Scalar>::MatrixType toInterleaved(typename AamMatrixTraits<Scalar>::ConstMatrixRefType rbased, int dims = 2)
    {
        typedef typename AamMatrixTraits<Scalar>::MatrixType ReturnType;

        const ReturnType::Index nObjects = rbased.cols() / dims;
        const ReturnType::Index nRowsOutput = nObjects;
        const ReturnType::Index nColsOutput = rbased.rows() * dims;

        ReturnType r(nRowsOutput, nColsOutput);

        for (ReturnType::Index y = 0; y < nObjects; ++y) {

            auto rowOutput = r.row(y);
            auto blockInput = rbased.block(0, y * dims, rbased.rows(), dims);

            for (ReturnType::Index x = 0; x < blockInput.rows(); ++x) {
                rowOutput.segment(x * dims, dims) = blockInput.row(x);
            }
        }

        return r;
    }
}

#endif